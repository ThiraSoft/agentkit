package agentkit

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/ThiraSoft/agentkit/llm"
)

// ErrMaxSteps is returned when a turn reaches MaxSteps model calls
// without finishing.
var ErrMaxSteps = errors.New("agentkit: maximum number of steps reached")

// Hooks are called during a Send, within its goroutine. All are
// optional. A hook that blocks (OnFinish waiting for voice output,
// Approve waiting for user confirmation) does not see the cancellation
// triggered by another Send or by Reset: it is up to the caller to unblock
// whatever it is waiting on when canceling.
//
// Calling Send or Reset from within a hook of the active turn will deadlock;
// Messages and SetSystem remain permitted.
type Hooks struct {
	OnText       func(chunk string)
	OnToolCall   func(ToolCall)
	OnToolResult func(ToolResult)
	// OnStepEnd: the model finished a message requesting tool calls.
	OnStepEnd func()
	// OnFinish: the model finished its response, the turn is about to close.
	// It may block (e.g. voice finishing speech); if the turn is interrupted
	// during this time, it is treated as an interruption and OnInterrupt decides
	// what text is kept.
	OnFinish func()
	// Approve: returning false rejects the call; the model receives a rejection.
	Approve func(ToolCall) bool
	// Prepare receives a copy of the history before each model call
	// and returns what will be sent. The copy is shallow: mutating in place
	// nested slice contents (Media, ToolCalls) modifies history; replacing
	// a field or an entire slice has no effect.
	Prepare func(msgs []llm.Message) []llm.Message
	// OnInterrupt receives the text written by the model during an interrupted
	// turn and returns what should be kept in history.
	OnInterrupt func(written string) (kept string)
}

// Turn summarizes a Send.
type Turn struct {
	Text        string // assistant text of the turn, joined by double newlines
	Steps       int    // model calls
	Interrupted bool
}

// Conversation holds history, including the system prompt, and has only
// one active turn: a Send during another interrupts the first.
type Conversation struct {
	agent *Agent

	mu       sync.Mutex
	messages []llm.Message // messages[0] is always the system prompt
	running  *run
}

type run struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// NewConversation starts a conversation, optionally with existing history
// (without system prompt).
func (a *Agent) NewConversation(system string, history ...llm.Message) *Conversation {
	msgs := append([]llm.Message{{Role: "system", Content: system}}, history...)
	return &Conversation{agent: a, messages: msgs}
}

// Messages returns a copy of the history.
func (c *Conversation) Messages() []llm.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]llm.Message(nil), c.messages...)
}

// SetSystem replaces the system prompt without modifying the rest.
func (c *Conversation) SetSystem(system string) {
	c.mu.Lock()
	c.messages[0].Content = system
	c.mu.Unlock()
}

// Reset interrupts the active turn if any, then clears all messages
// except the system prompt.
//
// Calling Send or Reset from within a hook of the active turn will deadlock;
// Messages and SetSystem remain permitted.
func (c *Conversation) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stopRunning()
	c.messages = c.messages[:1:1]
}

// stopRunning cancels the active turn and waits until it finishes writing.
// Called with c.mu held, which is released during the wait.
func (c *Conversation) stopRunning() {
	for c.running != nil {
		prev := c.running
		c.mu.Unlock()
		prev.cancel()
		<-prev.done
		c.mu.Lock()
	}
}

// Send appends text to the history and runs the model and its tools
// until a response without tool calls is produced. Any active Send is
// interrupted first, and the new turn sees whatever remains of it.
//
// Calling Send or Reset from within a hook of the active turn will deadlock;
// Messages and SetSystem remain permitted.
func (c *Conversation) Send(ctx context.Context, text string, h Hooks) (Turn, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r := &run{cancel: cancel, done: make(chan struct{})}
	defer close(r.done)
	defer func() {
		c.mu.Lock()
		if c.running == r {
			c.running = nil
		}
		c.mu.Unlock()
	}()

	c.mu.Lock()
	c.stopRunning()
	c.running = r
	start := len(c.messages)
	c.messages = append(c.messages, llm.Message{Role: "user", Content: text})
	c.mu.Unlock()

	return c.loop(ctx, h, start)
}

func (c *Conversation) loop(ctx context.Context, h Hooks, start int) (Turn, error) {
	a := c.agent
	var turn Turn
	for turn.Steps < a.maxSteps {
		turn.Steps++
		req := c.Messages()
		if h.Prepare != nil {
			req = h.Prepare(req)
		}
		resp, err := a.provider.Stream(ctx, req, a.defs, func(chunk string) error {
			if h.OnText != nil {
				h.OnText(chunk)
			}
			return nil
		})
		if err != nil {
			// Calls from an incomplete response are unreliable: only text
			// is kept.
			if resp != nil && resp.Content != "" {
				c.add(llm.Message{Role: "assistant", Content: resp.Content})
			}
			return c.end(ctx, h, start, turn, ctx.Err() != nil, err)
		}
		if resp == nil {
			resp = &llm.Message{}
		}
		resp.Role = "assistant"
		if resp.Content != "" || len(resp.ToolCalls) > 0 {
			c.add(*resp)
		}
		if len(resp.ToolCalls) == 0 {
			if h.OnFinish != nil {
				h.OnFinish()
			}
			return c.end(ctx, h, start, turn, ctx.Err() != nil, nil)
		}
		if h.OnStepEnd != nil {
			h.OnStepEnd()
		}
		// Each call receives a result, even if interrupted: pairs remain
		// complete by construction.
		for _, r := range a.runTools(ctx, resp.ToolCalls, h) {
			c.add(toolMessage(r))
			if h.OnToolResult != nil {
				h.OnToolResult(r)
			}
		}
		if ctx.Err() != nil {
			return c.end(ctx, h, start, turn, true, ctx.Err())
		}
	}
	return c.end(ctx, h, start, turn, ctx.Err() != nil, ErrMaxSteps)
}

func (c *Conversation) add(m llm.Message) {
	c.mu.Lock()
	c.messages = append(c.messages, m)
	c.mu.Unlock()
}

// end closes the turn. If interrupted, its assistant text is replaced by
// whatever OnInterrupt keeps, or "…" if nothing is kept. If not interrupted
// and the last message in history is not an assistant message (early error,
// empty response, or turn ending on tool results), an assistant message "…"
// is added to maintain the user/assistant alternation in history, without
// modifying turn.Text.
func (c *Conversation) end(ctx context.Context, h Hooks, start int, turn Turn, cut bool, err error) (Turn, error) {
	if cut {
		turn.Interrupted = true
		err = ctx.Err()
		c.mu.Lock()
		written := assistantText(c.messages[start:])
		c.mu.Unlock()
		kept := written
		if h.OnInterrupt != nil {
			kept = h.OnInterrupt(written)
		}
		if strings.TrimSpace(kept) == "" {
			kept = "…"
		}
		c.mu.Lock()
		c.keep(start, kept)
		c.mu.Unlock()
	}
	c.mu.Lock()
	turn.Text = assistantText(c.messages[start:])
	if !cut && len(c.messages) > 0 && c.messages[len(c.messages)-1].Role != "assistant" {
		c.messages = append(c.messages, llm.Message{Role: "assistant", Content: "…"})
	}
	c.mu.Unlock()
	return turn, err
}

// keep replaces the turn's assistant text with kept: it goes into the last
// message if it is an assistant without tool calls, or into a new message
// at the end of the turn. Called with c.mu held.
func (c *Conversation) keep(start int, kept string) {
	for i := start; i < len(c.messages); i++ {
		if c.messages[i].Role == "assistant" {
			c.messages[i].Content = ""
		}
	}
	last := &c.messages[len(c.messages)-1]
	if last.Role == "assistant" && len(last.ToolCalls) == 0 {
		last.Content = kept
		return
	}
	c.messages = append(c.messages, llm.Message{Role: "assistant", Content: kept})
}

func assistantText(msgs []llm.Message) string {
	var parts []string
	for _, m := range msgs {
		if m.Role == "assistant" && m.Content != "" {
			parts = append(parts, m.Content)
		}
	}
	return strings.Join(parts, "\n\n")
}
