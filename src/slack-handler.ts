import { App } from '@slack/bolt';
import { ClaudeHandler } from './claude-handler';
import { SDKMessage } from '@anthropic-ai/claude-agent-sdk';
import { Logger } from './logger';
import { WorkingDirectoryManager } from './working-directory-manager';
import { FileHandler, ProcessedFile } from './file-handler';
import { TodoManager, Todo } from './todo-manager';
import { McpManager } from './mcp-manager';
import { permissionServer } from './permission-mcp-server';
import { config } from './config';
import { inspectInbound, applyReadOnlyForBotTurn, stampOutgoing, MAX_HOPS } from './agent-chat-guard';

// Known CalibrateRL agent bots: Slack user ID -> @name. Bot-to-bot detection
// (read-only turn + hop counter) keys on event.bot_id, which any bot-authored
// event carries; this list additionally (a) treats events authored by these
// users as bot turns even if bot_id is missing, and (b) drives the hop-limit
// mention strip so a chain can't be extended by @naming another agent.
const AGENT_BOTS: Record<string, string> = {
  U0B9C278VPW: 'gilbert',
  U0BA8P14L5N: 'kathryne',
  U0B9CFA6KFY: 'charizard',
  U0B9X82Q5FX: 'awesome-ash', // L40S training executor (formerly 'trainaws')
  U0B9Y47N1EH: 'sam', // L4 sampling executor
  U0B9L2MEKUP: 'sadie', // L4 sampling executor
};
const AGENT_BOT_MENTION_RE = new RegExp(
  `<@[A-Z0-9]+>|@(${Object.values(AGENT_BOTS).join('|')})\\b`,
  'gi'
);
// NON-global on purpose: .test() on a /g regex is stateful (lastIndex). Used to
// detect "this reply @mentions a known agent" => a human turn starts a chain.
const AGENT_USER_MENTION_RE = new RegExp(
  `<@(${Object.keys(AGENT_BOTS).join('|')})>`,
  'i'
);

// Turn timeout: abort any task running past this and say so in the status
// message, instead of holding the session open forever.
const MAX_TURN_SECONDS = parseInt(process.env.MAX_TURN_SECONDS || '1200', 10);
export const TIMEOUT_STATUS = ':stopwatch: timed out — re-ask or split';
export const INTERRUPTED_STATUS = ':black_square_for_stop: interrupted';

// Stale-event guard: Slack redelivers events (retry after a slow ack — our turns
// run minutes; replay of queued events after a socket reconnect). Without this,
// a bot that was deaf for hours answers every queued mention at once on restart,
// and a slow turn gets processed 2-3x. Events older than this are dropped.
const STALE_EVENT_MAX_AGE_SECONDS = parseInt(
  process.env.STALE_EVENT_MAX_AGE_SECONDS || '600',
  10
);

interface MessageEvent {
  user: string;
  channel: string;
  thread_ts?: string;
  ts: string;
  text?: string;
  files?: Array<{
    id: string;
    name: string;
    mimetype: string;
    filetype: string;
    url_private: string;
    url_private_download: string;
    size: number;
  }>;
}

export class SlackHandler {
  private app: App;
  private claudeHandler: ClaudeHandler;
  private activeControllers: Map<string, AbortController> = new Map();
  private logger = new Logger('SlackHandler');
  private workingDirManager: WorkingDirectoryManager;
  private fileHandler: FileHandler;
  private todoManager: TodoManager;
  private mcpManager: McpManager;
  private todoMessages: Map<string, string> = new Map(); // sessionKey -> messageTs
  private originalMessages: Map<string, { channel: string; ts: string }> = new Map(); // sessionKey -> original message info
  private statusMessages: Map<string, { channel: string; ts: string }> = new Map(); // sessionKey -> in-flight status message (for SIGTERM)
  private currentReactions: Map<string, string> = new Map(); // sessionKey -> current emoji
  private botUserId: string | null = null;

  constructor(app: App, claudeHandler: ClaudeHandler, mcpManager: McpManager) {
    this.app = app;
    this.claudeHandler = claudeHandler;
    this.mcpManager = mcpManager;
    this.workingDirManager = new WorkingDirectoryManager();
    this.fileHandler = new FileHandler();
    this.todoManager = new TodoManager();
  }

  async handleMessage(event: MessageEvent, say: any) {
    const { user, channel, thread_ts, ts, text, files } = event;
    
    // Process any attached files
    let processedFiles: ProcessedFile[] = [];
    if (files && files.length > 0) {
      this.logger.info('Processing uploaded files', { count: files.length });
      processedFiles = await this.fileHandler.downloadAndProcessFiles(files);
      
      if (processedFiles.length > 0) {
        await say({
          text: `📎 Processing ${processedFiles.length} file(s): ${processedFiles.map(f => f.name).join(', ')}`,
          thread_ts: thread_ts || ts,
        });
      }
    }

    // If no text and no files, nothing to process
    if (!text && processedFiles.length === 0) return;

    this.logger.debug('Received message from Slack', {
      user,
      channel,
      thread_ts,
      ts,
      text: text ? text.substring(0, 100) + (text.length > 100 ? '...' : '') : '[no text]',
      fileCount: processedFiles.length,
    });

    // Check if this is a working directory command (only if there's text)
    const setDirPath = text ? this.workingDirManager.parseSetCommand(text) : null;
    if (setDirPath) {
      const isDM = channel.startsWith('D');
      const result = this.workingDirManager.setWorkingDirectory(
        channel,
        setDirPath,
        thread_ts,
        isDM ? user : undefined
      );

      if (result.success) {
        const context = thread_ts ? 'this thread' : (isDM ? 'this conversation' : 'this channel');
        await say({
          text: `✅ Working directory set for ${context}: \`${result.resolvedPath}\``,
          thread_ts: thread_ts || ts,
        });
      } else {
        await say({
          text: `❌ ${result.error}`,
          thread_ts: thread_ts || ts,
        });
      }
      return;
    }

    // Check if this is a get directory command (only if there's text)
    if (text && this.workingDirManager.isGetCommand(text)) {
      const isDM = channel.startsWith('D');
      const directory = this.workingDirManager.getWorkingDirectory(
        channel,
        thread_ts,
        isDM ? user : undefined
      );
      const context = thread_ts ? 'this thread' : (isDM ? 'this conversation' : 'this channel');
      
      await say({
        text: this.workingDirManager.formatDirectoryMessage(directory, context),
        thread_ts: thread_ts || ts,
      });
      return;
    }

    // Check if this is an MCP info command (only if there's text)
    if (text && this.isMcpInfoCommand(text)) {
      await say({
        text: this.mcpManager.formatMcpInfo(),
        thread_ts: thread_ts || ts,
      });
      return;
    }

    // Check if this is an MCP reload command (only if there's text)
    if (text && this.isMcpReloadCommand(text)) {
      const reloaded = this.mcpManager.reloadConfiguration();
      if (reloaded) {
        await say({
          text: `✅ MCP configuration reloaded successfully.\n\n${this.mcpManager.formatMcpInfo()}`,
          thread_ts: thread_ts || ts,
        });
      } else {
        await say({
          text: `❌ Failed to reload MCP configuration. Check the mcp-servers.json file.`,
          thread_ts: thread_ts || ts,
        });
      }
      return;
    }

    // Check if we have a working directory set
    const isDM = channel.startsWith('D');
    const workingDirectory = this.workingDirManager.getWorkingDirectory(
      channel,
      thread_ts,
      isDM ? user : undefined
    );

    // Working directory is always required
    if (!workingDirectory) {
      let errorMessage = `⚠️ No working directory set. `;
      
      if (!isDM && !this.workingDirManager.hasChannelWorkingDirectory(channel)) {
        // No channel default set
        errorMessage += `Please set a default working directory for this channel first using:\n`;
        if (config.baseDirectory) {
          errorMessage += `\`cwd project-name\` or \`cwd /absolute/path\`\n\n`;
          errorMessage += `Base directory: \`${config.baseDirectory}\``;
        } else {
          errorMessage += `\`cwd /path/to/directory\``;
        }
      } else if (thread_ts) {
        // In thread but no thread-specific directory
        errorMessage += `You can set a thread-specific working directory using:\n`;
        if (config.baseDirectory) {
          errorMessage += `\`@claudebot cwd project-name\` or \`@claudebot cwd /absolute/path\``;
        } else {
          errorMessage += `\`@claudebot cwd /path/to/directory\``;
        }
      } else {
        errorMessage += `Please set one first using:\n\`cwd /path/to/directory\``;
      }
      
      await say({
        text: errorMessage,
        thread_ts: thread_ts || ts,
      });
      return;
    }

    const sessionKey = this.claudeHandler.getSessionKey(user, channel, thread_ts || ts);
    
    // Store the original message info for status reactions
    const originalMessageTs = thread_ts || ts;
    this.originalMessages.set(sessionKey, { channel, ts: originalMessageTs });
    
    // Cancel any existing request for this conversation
    const existingController = this.activeControllers.get(sessionKey);
    if (existingController) {
      this.logger.debug('Cancelling existing request for session', { sessionKey });
      existingController.abort();
    }

    const abortController = new AbortController();
    this.activeControllers.set(sessionKey, abortController);

    // Turn timeout: a runaway turn is aborted and reported, not left spinning.
    let turnTimedOut = false;
    const turnTimer = setTimeout(() => {
      turnTimedOut = true;
      this.logger.warn('Turn exceeded MAX_TURN_SECONDS, aborting', { sessionKey, MAX_TURN_SECONDS });
      abortController.abort();
    }, MAX_TURN_SECONDS * 1000);

    let session = this.claudeHandler.getSession(user, channel, thread_ts || ts);
    if (!session) {
      this.logger.debug('Creating new session', { sessionKey });
      session = this.claudeHandler.createSession(user, channel, thread_ts || ts);
    } else {
      this.logger.debug('Using existing session', { sessionKey, sessionId: session.sessionId });
    }

    let currentMessages: string[] = [];
    let statusMessageTs: string | undefined;

    try {
      // Prepare the prompt with file attachments
      const finalPrompt = processedFiles.length > 0 
        ? await this.fileHandler.formatFilePrompt(processedFiles, text || '')
        : text || '';

      this.logger.info('Sending query to Claude Code SDK', { 
        prompt: finalPrompt.substring(0, 200) + (finalPrompt.length > 200 ? '...' : ''), 
        sessionId: session.sessionId,
        workingDirectory,
        fileCount: processedFiles.length,
      });

      // Send initial status message
      const statusResult = await say({
        text: '🤔 *Thinking...*',
        thread_ts: thread_ts || ts,
      });
      statusMessageTs = statusResult.ts;
      if (statusMessageTs) {
        this.statusMessages.set(sessionKey, { channel, ts: statusMessageTs });
      }

      // Add thinking reaction to original message (but don't spam if already set)
      await this.updateMessageReaction(sessionKey, '🤔');
      
      // Create Slack context for permission prompts
      const slackContext = {
        channel,
        threadTs: thread_ts,
        user
      };
      
      const chain = (this as any)._pendingChain || { isFromBot: false, hop: 0, atLimit: false };
      (this as any)._pendingChain = null; // consume it
      for await (const message of this.claudeHandler.streamQuery(finalPrompt, session, abortController, workingDirectory, slackContext, chain)) {
        if (abortController.signal.aborted) break;

        this.logger.debug('Received message from Claude SDK', {
          type: message.type,
          subtype: (message as any).subtype,
          message: message,
        });

        if (message.type === 'assistant') {
          // Check if this is a tool use message
          const hasToolUse = message.message.content?.some((part: any) => part.type === 'tool_use');
          
          if (hasToolUse) {
            // Update status to show working
            if (statusMessageTs) {
              await this.app.client.chat.update({
                channel,
                ts: statusMessageTs,
                text: '⚙️ *Working...*',
              });
            }

            // Update reaction to show working
            await this.updateMessageReaction(sessionKey, '⚙️');

            // Check for TodoWrite tool and handle it specially
            const todoTool = message.message.content?.find((part: any) => 
              part.type === 'tool_use' && part.name === 'TodoWrite'
            );

            if (todoTool) {
              await this.handleTodoUpdate(todoTool.input, sessionKey, session?.sessionId, channel, thread_ts || ts, say);
            }

            // Tool-use details are logged, not posted — keeps the channel readable.
            // Set VERBOSE_TOOLS=true in .env to restore per-tool Slack messages.
            const toolContent = this.formatToolUse(message.message.content);
            if (toolContent && process.env.VERBOSE_TOOLS === 'true') {
              await say({
                text: toolContent,
                thread_ts: thread_ts || ts,
              });
            } else if (toolContent) {
              this.logger.debug('Tool use (suppressed from Slack)', { toolContent: toolContent.substring(0, 200) });
            }
          } else {
            // Handle regular text content
            const content = this.extractTextContent(message);
            if (content) {
              currentMessages.push(content);
              
              // Send each new piece of content as a separate message
              const formatted = this.formatMessage(content, false);
              await say({
                text: formatted,
                thread_ts: thread_ts || ts,
              });
            }
          }
        } else if (message.type === 'result') {
          this.logger.info('Received result from Claude SDK', {
            subtype: message.subtype,
            hasResult: message.subtype === 'success' && !!(message as any).result,
            totalCost: (message as any).total_cost_usd,
            duration: (message as any).duration_ms,
          });
          
          if (message.subtype === 'success' && (message as any).result) {
            const finalResult = (message as any).result;
            if (finalResult && !currentMessages.includes(finalResult)) {
              const stamped = stampOutgoing(finalResult, chain, {
                stripRe: AGENT_BOT_MENTION_RE,
                agentRe: AGENT_USER_MENTION_RE,
                rootTs: ts, // a human turn that @mentions an agent roots a chain here
              });
              const formatted = this.formatMessage(stamped, true);
              await say({
                text: formatted,
                thread_ts: thread_ts || ts,
              });
            }
          }
        }
      }

      // The stream loop exits without throwing when the controller aborts
      // (timeout, or a newer message for the same session) — don't report that
      // as success.
      if (abortController.signal.aborted) {
        await this.finishAbortedTurn(sessionKey, statusMessageTs, channel, turnTimedOut);
      } else {
        // Update status to completed
        if (statusMessageTs) {
          await this.app.client.chat.update({
            channel,
            ts: statusMessageTs,
            text: '✅ *Task completed*',
          });
        }

        // Update reaction to show completion
        await this.updateMessageReaction(sessionKey, '✅');

        this.logger.info('Completed processing message', {
          sessionKey,
          messageCount: currentMessages.length,
        });
      }

      // Clean up temporary files
      if (processedFiles.length > 0) {
        await this.fileHandler.cleanupTempFiles(processedFiles);
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        this.logger.error('Error handling message', error);
        
        // Update status to error
        if (statusMessageTs) {
          await this.app.client.chat.update({
            channel,
            ts: statusMessageTs,
            text: '❌ *Error occurred*',
          });
        }

        // Update reaction to show error
        await this.updateMessageReaction(sessionKey, '❌');
        
        await say({
          text: `Error: ${error.message || 'Something went wrong'}`,
          thread_ts: thread_ts || ts,
        });
      } else {
        this.logger.debug('Request was aborted', { sessionKey, turnTimedOut });
        await this.finishAbortedTurn(sessionKey, statusMessageTs, channel, turnTimedOut);
      }

      // Clean up temporary files in case of error too
      if (processedFiles.length > 0) {
        await this.fileHandler.cleanupTempFiles(processedFiles);
      }
    } finally {
      clearTimeout(turnTimer);
      this.activeControllers.delete(sessionKey);
      this.statusMessages.delete(sessionKey);

      // Clean up todo tracking if session ended
      if (session?.sessionId) {
        // Don't immediately clean up - keep todos visible for a while
        setTimeout(() => {
          this.todoManager.cleanupSession(session.sessionId!);
          this.todoMessages.delete(sessionKey);
          this.originalMessages.delete(sessionKey);
          this.currentReactions.delete(sessionKey);
        }, 5 * 60 * 1000); // 5 minutes
      }
    }
  }

  // Report an aborted turn: timeout gets its own status so the asker knows to
  // re-ask or split; everything else is a plain cancellation.
  private async finishAbortedTurn(
    sessionKey: string,
    statusMessageTs: string | undefined,
    channel: string,
    timedOut: boolean
  ): Promise<void> {
    const text = timedOut ? TIMEOUT_STATUS : '⏹️ *Cancelled*';
    if (statusMessageTs) {
      try {
        await this.app.client.chat.update({ channel, ts: statusMessageTs, text });
      } catch (error) {
        this.logger.warn('Failed to update status for aborted turn', error);
      }
    }
    await this.updateMessageReaction(sessionKey, timedOut ? 'stopwatch' : '⏹️');
  }

  // SIGTERM/SIGINT with tasks in flight (pm2 restart sends SIGINT by default,
  // plain `kill` sends SIGTERM): mark every in-flight status message as
  // interrupted so the death isn't silent, then exit. Bounded — Slack gets a
  // few seconds, not a veto.
  private async handleShutdownSignal(signal: string): Promise<void> {
    const inFlight = [...this.activeControllers.keys()];
    this.logger.warn(`${signal} received`, { tasksInFlight: inFlight.length });
    const edits = inFlight.map(async (sessionKey) => {
      const status = this.statusMessages.get(sessionKey);
      if (status) {
        await this.app.client.chat.update({
          channel: status.channel,
          ts: status.ts,
          text: INTERRUPTED_STATUS,
        });
      }
    });
    for (const controller of this.activeControllers.values()) controller.abort();
    await Promise.race([
      Promise.allSettled(edits),
      new Promise((resolve) => setTimeout(resolve, 4000)),
    ]);
    process.exit(0);
  }

  // PROVENANCE GATE: a chain may do work only if its tagged root message was
  // authored by a real human. The tag is just a ts — never trusted by itself;
  // we fetch the message and check the author. Fail-safe: any error, missing
  // message, or bot author -> false (read-only turn).
  private async isChainHumanRooted(channel: string, rootTs: string): Promise<boolean> {
    try {
      let msg: any;
      try {
        // Works for thread replies too: conversations.replies with the reply's
        // own ts returns that message.
        const res = await this.app.client.conversations.replies({
          channel, ts: rootTs, limit: 1, inclusive: true,
        });
        msg = (res.messages as any[] | undefined)?.find((m) => m.ts === rootTs);
      } catch {
        // fall through to history lookup
      }
      if (!msg) {
        const res = await this.app.client.conversations.history({
          channel, latest: rootTs, oldest: rootTs, inclusive: true, limit: 1,
        });
        msg = (res.messages as any[] | undefined)?.find((m) => m.ts === rootTs);
      }
      if (!msg) return false;
      const author = msg.user as string | undefined;
      if (!author || AGENT_BOTS[author]) return false;
      return await this.isHumanAuthor(author);
    } catch {
      return false;
    }
  }

  private extractTextContent(message: SDKMessage): string | null {
    if (message.type === 'assistant' && message.message.content) {
      const textParts = message.message.content
        .filter((part: any) => part.type === 'text')
        .map((part: any) => part.text);
      return textParts.join('');
    }
    return null;
  }

  private formatToolUse(content: any[]): string {
    const parts: string[] = [];
    
    for (const part of content) {
      if (part.type === 'text') {
        parts.push(part.text);
      } else if (part.type === 'tool_use') {
        const toolName = part.name;
        const input = part.input;
        
        switch (toolName) {
          case 'Edit':
          case 'MultiEdit':
            parts.push(this.formatEditTool(toolName, input));
            break;
          case 'Write':
            parts.push(this.formatWriteTool(input));
            break;
          case 'Read':
            parts.push(this.formatReadTool(input));
            break;
          case 'Bash':
            parts.push(this.formatBashTool(input));
            break;
          case 'TodoWrite':
            // Handle TodoWrite separately - don't include in regular tool output
            return this.handleTodoWrite(input);
          default:
            parts.push(this.formatGenericTool(toolName, input));
        }
      }
    }
    
    return parts.join('\n\n');
  }

  private formatEditTool(toolName: string, input: any): string {
    const filePath = input.file_path;
    const edits = toolName === 'MultiEdit' ? input.edits : [{ old_string: input.old_string, new_string: input.new_string }];
    
    let result = `📝 *Editing \`${filePath}\`*\n`;
    
    for (const edit of edits) {
      result += '\n```diff\n';
      result += `- ${this.truncateString(edit.old_string, 200)}\n`;
      result += `+ ${this.truncateString(edit.new_string, 200)}\n`;
      result += '```';
    }
    
    return result;
  }

  private formatWriteTool(input: any): string {
    const filePath = input.file_path;
    const preview = this.truncateString(input.content, 300);
    
    return `📄 *Creating \`${filePath}\`*\n\`\`\`\n${preview}\n\`\`\``;
  }

  private formatReadTool(input: any): string {
    return `👁️ *Reading \`${input.file_path}\`*`;
  }

  private formatBashTool(input: any): string {
    return `🖥️ *Running command:*\n\`\`\`bash\n${input.command}\n\`\`\``;
  }

  private formatGenericTool(toolName: string, input: any): string {
    return `🔧 *Using ${toolName}*`;
  }

  private truncateString(str: string, maxLength: number): string {
    if (!str) return '';
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
  }

  private handleTodoWrite(input: any): string {
    // TodoWrite tool doesn't produce visible output - handled separately
    return '';
  }

  private async handleTodoUpdate(
    input: any, 
    sessionKey: string, 
    sessionId: string | undefined, 
    channel: string, 
    threadTs: string, 
    say: any
  ): Promise<void> {
    if (!sessionId || !input.todos) {
      return;
    }

    const newTodos: Todo[] = input.todos;
    const oldTodos = this.todoManager.getTodos(sessionId);
    
    // Check if there's a significant change
    if (this.todoManager.hasSignificantChange(oldTodos, newTodos)) {
      // Update the todo manager
      this.todoManager.updateTodos(sessionId, newTodos);
      
      // Format the todo list
      const todoList = this.todoManager.formatTodoList(newTodos);
      
      // Check if we already have a todo message for this session
      const existingTodoMessageTs = this.todoMessages.get(sessionKey);
      
      if (existingTodoMessageTs) {
        // Update existing todo message
        try {
          await this.app.client.chat.update({
            channel,
            ts: existingTodoMessageTs,
            text: todoList,
          });
          this.logger.debug('Updated existing todo message', { sessionKey, messageTs: existingTodoMessageTs });
        } catch (error) {
          this.logger.warn('Failed to update todo message, creating new one', error);
          // If update fails, create a new message
          await this.createNewTodoMessage(todoList, channel, threadTs, sessionKey, say);
        }
      } else {
        // Create new todo message
        await this.createNewTodoMessage(todoList, channel, threadTs, sessionKey, say);
      }

      // Send status change notification if there are meaningful changes
      const statusChange = this.todoManager.getStatusChange(oldTodos, newTodos);
      if (statusChange) {
        await say({
          text: `🔄 *Task Update:*\n${statusChange}`,
          thread_ts: threadTs,
        });
      }

      // Update reaction based on overall progress
      await this.updateTaskProgressReaction(sessionKey, newTodos);
    }
  }

  private async createNewTodoMessage(
    todoList: string, 
    channel: string, 
    threadTs: string, 
    sessionKey: string, 
    say: any
  ): Promise<void> {
    const result = await say({
      text: todoList,
      thread_ts: threadTs,
    });
    
    if (result?.ts) {
      this.todoMessages.set(sessionKey, result.ts);
      this.logger.debug('Created new todo message', { sessionKey, messageTs: result.ts });
    }
  }

  private static EMOJI_NAMES: Record<string, string> = {
    '🤔': 'thinking_face',
    '⚙️': 'gear',
    '✅': 'white_check_mark',
    '❌': 'x',
    '⏹️': 'black_square_for_stop',
    '🔄': 'arrows_counterclockwise',
    '📋': 'clipboard',
  };

  private async updateMessageReaction(sessionKey: string, emoji: string): Promise<void> {
    // Slack's reactions API takes emoji NAMES (e.g. 'gear'), not unicode chars.
    emoji = SlackHandler.EMOJI_NAMES[emoji] || emoji;
    const originalMessage = this.originalMessages.get(sessionKey);
    if (!originalMessage) {
      return;
    }

    // Check if we're already showing this emoji
    const currentEmoji = this.currentReactions.get(sessionKey);
    if (currentEmoji === emoji) {
      this.logger.debug('Reaction already set, skipping', { sessionKey, emoji });
      return;
    }

    try {
      // Remove the current reaction if it exists
      if (currentEmoji) {
        try {
          await this.app.client.reactions.remove({
            channel: originalMessage.channel,
            timestamp: originalMessage.ts,
            name: currentEmoji,
          });
          this.logger.debug('Removed previous reaction', { sessionKey, emoji: currentEmoji });
        } catch (error) {
          this.logger.debug('Failed to remove previous reaction (might not exist)', { 
            sessionKey, 
            emoji: currentEmoji,
            error: (error as any).message 
          });
        }
      }

      // Add the new reaction
      await this.app.client.reactions.add({
        channel: originalMessage.channel,
        timestamp: originalMessage.ts,
        name: emoji,
      });

      // Track the current reaction
      this.currentReactions.set(sessionKey, emoji);

      this.logger.debug('Updated message reaction', { 
        sessionKey, 
        emoji, 
        previousEmoji: currentEmoji,
        channel: originalMessage.channel, 
        ts: originalMessage.ts 
      });
    } catch (error) {
      this.logger.warn('Failed to update message reaction', error);
    }
  }

  private async updateTaskProgressReaction(sessionKey: string, todos: Todo[]): Promise<void> {
    if (todos.length === 0) {
      return;
    }

    const completed = todos.filter(t => t.status === 'completed').length;
    const inProgress = todos.filter(t => t.status === 'in_progress').length;
    const total = todos.length;

    let emoji: string;
    if (completed === total) {
      emoji = '✅'; // All tasks completed
    } else if (inProgress > 0) {
      emoji = '🔄'; // Tasks in progress
    } else {
      emoji = '📋'; // Tasks pending
    }

    await this.updateMessageReaction(sessionKey, emoji);
  }

  private isMcpInfoCommand(text: string): boolean {
    return /^(mcp|servers?)(\s+(info|list|status))?(\?)?$/i.test(text.trim());
  }

  private isMcpReloadCommand(text: string): boolean {
    return /^(mcp|servers?)\s+(reload|refresh)$/i.test(text.trim());
  }

  private async getBotUserId(): Promise<string> {
    if (!this.botUserId) {
      try {
        const response = await this.app.client.auth.test();
        this.botUserId = response.user_id as string;
      } catch (error) {
        this.logger.error('Failed to get bot user ID', error);
        this.botUserId = '';
      }
    }
    return this.botUserId;
  }

  private async handleChannelJoin(channelId: string, say: any): Promise<void> {
    try {
      // Get channel info
      const channelInfo = await this.app.client.conversations.info({
        channel: channelId,
      });

      const channelName = (channelInfo.channel as any)?.name || 'this channel';
      
      let welcomeMessage = `👋 Hi! I'm Claude Code, your AI coding assistant.\n\n`;
      welcomeMessage += `To get started, I need to know the default working directory for #${channelName}.\n\n`;
      
      if (config.baseDirectory) {
        welcomeMessage += `You can use:\n`;
        welcomeMessage += `• \`cwd project-name\` (relative to base directory: \`${config.baseDirectory}\`)\n`;
        welcomeMessage += `• \`cwd /absolute/path/to/project\` (absolute path)\n\n`;
      } else {
        welcomeMessage += `Please set it using:\n`;
        welcomeMessage += `• \`cwd /path/to/project\` or \`set directory /path/to/project\`\n\n`;
      }
      
      welcomeMessage += `This will be the default working directory for this channel. `;
      welcomeMessage += `You can always override it for specific threads by mentioning me with a different \`cwd\` command.\n\n`;
      welcomeMessage += `Once set, you can ask me to help with code reviews, file analysis, debugging, and more!`;

      await say({
        text: welcomeMessage,
      });

      this.logger.info('Sent welcome message to channel', { channelId, channelName });
    } catch (error) {
      this.logger.error('Failed to handle channel join', error);
    }
  }

  private formatMessage(text: string, isFinal: boolean): string {
    // Convert markdown code blocks to Slack format
    let formatted = text
      .replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
        return '```' + code + '```';
      })
      .replace(/`([^`]+)`/g, '`$1`')
      .replace(/\*\*([^*]+)\*\*/g, '*$1*')
      .replace(/__([^_]+)__/g, '_$1_');

    return formatted;
  }

  private seenEvents: Map<string, number> = new Map(); // channel:ts -> first-seen ms

  // Drop events that are stale (older than STALE_EVENT_MAX_AGE_SECONDS — e.g.
  // replayed after a reconnect) or already processed (Slack retries delivery
  // when the ack is slow, which long Claude turns guarantee). Call this only
  // once a handler has decided it WOULD process the event, so an event skipped
  // by one handler doesn't consume the dedupe slot of another.
  private shouldProcessEvent(event: any, kind: string): boolean {
    const eventTs = parseFloat(event.event_ts || event.ts || '0');
    if (eventTs > 0) {
      const ageSec = Date.now() / 1000 - eventTs;
      if (ageSec > STALE_EVENT_MAX_AGE_SECONDS) {
        this.logger.info('Dropping stale event', { kind, ts: event.ts, ageSec: Math.round(ageSec) });
        return false;
      }
    }
    const key = `${event.channel}:${event.ts}`;
    if (this.seenEvents.has(key)) {
      this.logger.info('Dropping duplicate event delivery', { kind, key });
      return false;
    }
    this.seenEvents.set(key, Date.now());
    if (this.seenEvents.size > 500) {
      const cutoff = Date.now() - 2 * STALE_EVENT_MAX_AGE_SECONDS * 1000;
      for (const [k, t] of this.seenEvents) {
        if (t < cutoff) this.seenEvents.delete(k);
      }
    }
    return true;
  }

  // Addressing rule: a message is FOR this bot when it mentions this bot
  // anywhere in the text. First-mention-wins was tried and silently dropped
  // the later addressees in multi-task messages ("<@a> do X and <@b> do Y").
  // The cost is that purely referential mentions ("ask <@me> about X") also
  // trigger a reply — accepted residual risk: the read-only bot-turn guard
  // and the hop cap bound any resulting pileup.
  private isAddressedToMe(text: string | undefined, botUserId: string): boolean {
    if (!text) return false;
    return text.includes(`<@${botUserId}>`);
  }

  private humanUserCache: Map<string, boolean> = new Map();

  // True when userId is a real workspace human (not a bot user). Messages posted
  // via a user token (xoxp) carry BOTH a human `user` and a `bot_id`; the human
  // author is what makes them human-authorized, so they count as human turns
  // (full tools) instead of read-only bot-to-bot turns.
  private async isHumanAuthor(userId: string | undefined): Promise<boolean> {
    if (!userId || userId === 'USLACKBOT') return false;
    const cached = this.humanUserCache.get(userId);
    if (cached !== undefined) return cached;
    let human = false;
    try {
      const res = await this.app.client.users.info({ user: userId });
      human = !!res.user && !(res.user as any).is_bot;
    } catch {
      human = false; // can't verify -> treat as bot (fail safe)
    }
    this.humanUserCache.set(userId, human);
    return human;
  }

  setupEventHandlers() {
    // Mark in-flight work as interrupted before dying (pm2 restart / kill).
    process.once('SIGTERM', () => { void this.handleShutdownSignal('SIGTERM'); });
    process.once('SIGINT', () => { void this.handleShutdownSignal('SIGINT'); });

    // Handle direct messages
    this.app.message(async ({ message, say }) => {
      // Ignore anything from a bot (prevents bots triggering each other / loops),
      // UNLESS the author is a real human posting via a user token (xoxp).
      if (((message as any).bot_id || (message as any).subtype === 'bot_message')
          && !(await this.isHumanAuthor((message as any).user))) {
        return;
      }
      // Only auto-respond in DMs. In channels, require an @mention (handled below).
      const isDM = typeof (message as any).channel === 'string' && (message as any).channel.startsWith('D');
      if (message.subtype === undefined && 'user' in message && isDM) {
        if (!this.shouldProcessEvent(message, 'dm')) return;
        this.logger.info('Handling direct message (DM) event');
        await this.handleMessage(message as MessageEvent, say);
      }
    });

    // Handle app mentions
    this.app.event('app_mention', async ({ event, say }) => {
      if (!this.shouldProcessEvent(event, 'app_mention')) return;

      const botId = (event as any).bot_id as string | undefined;
      const rawText = (event as any).text as string | undefined;

      // Only act when this bot is mentioned somewhere in the message text.
      // (app_mention can also fire on e.g. mentions inside attachments.)
      const myUserId = await this.getBotUserId();
      if (myUserId && !this.isAddressedToMe(rawText, myUserId)) {
        this.logger.info('Mention not in message text, not addressed to me; ignoring', {
          ts: (event as any).ts,
        });
        return;
      }

      this.logger.info('Handling app mention event');

      // Bounded bot-to-bot: inspect inbound for bot-author + hop count. A known
      // agent author counts as a bot even if the event lacks a bot_id (or looks
      // human to users.info). A human posting through a user token (xoxp) has a
      // bot_id but a human `user` -> treat as a human turn (full tools, hop
      // counter reset).
      const authorId = (event as any).user as string | undefined;
      const isKnownAgent = !!(authorId && AGENT_BOTS[authorId]);
      const effectiveBotId = botId || (isKnownAgent ? authorId : undefined);
      const authorIsHuman = !isKnownAgent && await this.isHumanAuthor(authorId);
      const chain = inspectInbound(authorIsHuman ? undefined : effectiveBotId, rawText);

      // Hard stop: if the inbound already passed the limit, do not respond at all.
      if (chain.isFromBot && chain.hop >= MAX_HOPS) {
        this.logger.info('Bot-to-bot chain at/over limit; ignoring', { hop: chain.hop });
        return;
      }

      if (chain.isFromBot) {
        chain.initiatorId = authorId; // @-back: address the reply to the asker
        if (chain.rootTs) {
          // Provenance gate: verify (never trust) the tagged root before
          // granting work permissions on this bot-initiated turn.
          chain.humanRooted = await this.isChainHumanRooted((event as any).channel, chain.rootTs);
        }
        this.logger.info('Bot-to-bot turn', {
          hop: chain.hop, rootTs: chain.rootTs, humanRooted: chain.humanRooted,
        });
      }

      const text = rawText ? rawText.replace(/<@[^>]+>/g, '').trim() : '';
      (this as any)._pendingChain = chain; // stash for handleMessage
      await this.handleMessage({ ...event, text } as MessageEvent, say);
    });

    // Handle file uploads in threads
    this.app.event('message', async ({ event, say }) => {
      // Only handle file uploads that are not from bots and have files
      if (event.subtype === 'file_share' && 'user' in event && event.files) {
        if (!this.shouldProcessEvent(event, 'file_share')) return;
        this.logger.info('Handling file upload event');
        await this.handleMessage(event as MessageEvent, say);
      }
    });

    // Handle bot being added to channels
    this.app.event('member_joined_channel', async ({ event, say }) => {
      // Check if the bot was added to the channel
      if (event.user === await this.getBotUserId()) {
        this.logger.info('Bot added to channel', { channel: event.channel });
        await this.handleChannelJoin(event.channel, say);
      }
    });

    // Handle permission approval button clicks
    this.app.action('approve_tool', async ({ ack, body, respond }) => {
      await ack();
      const approvalId = (body as any).actions[0].value;
      this.logger.info('Tool approval granted', { approvalId });
      
      permissionServer.resolveApproval(approvalId, true);
      
      await respond({
        response_type: 'ephemeral',
        text: '✅ Tool execution approved'
      });
    });

    // Handle permission denial button clicks
    this.app.action('deny_tool', async ({ ack, body, respond }) => {
      await ack();
      const approvalId = (body as any).actions[0].value;
      this.logger.info('Tool approval denied', { approvalId });
      
      permissionServer.resolveApproval(approvalId, false);
      
      await respond({
        response_type: 'ephemeral',
        text: '❌ Tool execution denied'
      });
    });

    // Cleanup inactive sessions periodically
    setInterval(() => {
      this.logger.debug('Running session cleanup');
      this.claudeHandler.cleanupInactiveSessions();
    }, 5 * 60 * 1000); // Every 5 minutes
  }
}