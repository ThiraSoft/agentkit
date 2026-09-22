package agentkit

import (
	"context"
	"encoding/json"
	"fmt"
	"unicode/utf8"

	"github.com/ThiraSoft/agentkit/llm"
)

// ToolCall is a tool call requested by the model.
type ToolCall struct {
	ID, Name  string
	Arguments json.RawMessage
}

// ToolResult is what the model receives in response. Err is empty if and
// only if the tool succeeded.
type ToolResult struct {
	ID, Name string
	Content  string
	Err      string
}

const interrupted = "interrupted"

// runTools executes calls from the same message in parallel and returns
// results in the order of the calls. OnToolCall and Approve are called
// first, one call after another, then OnToolResult as each result is known,
// in the order they finish; all of them in the Send goroutine.
func (a *Agent) runTools(ctx context.Context, calls []llm.ToolCall, h Hooks) []ToolResult {
	results := make([]ToolResult, len(calls))
	done := make(chan int, len(calls))
	report := func(i int, r ToolResult) {
		results[i] = r
		if h.OnToolResult != nil {
			h.OnToolResult(r)
		}
	}
	running := 0
	for i, tc := range calls {
		c := ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments}
		if ctx.Err() != nil {
			report(i, ToolResult{ID: c.ID, Name: c.Name, Content: interrupted, Err: ctx.Err().Error()})
			continue
		}
		if h.OnToolCall != nil {
			h.OnToolCall(c)
		}
		if h.Approve != nil && !h.Approve(c) {
			report(i, ToolResult{ID: c.ID, Name: c.Name, Content: "declined by user", Err: "declined by user"})
			continue
		}
		running++
		go func() {
			results[i] = a.runTool(ctx, c)
			done <- i
		}()
	}
	for ; running > 0; running-- {
		i := <-done
		report(i, results[i])
	}
	return results
}

func (a *Agent) runTool(ctx context.Context, c ToolCall) (r ToolResult) {
	r = ToolResult{ID: c.ID, Name: c.Name}
	// A panicking tool is an error the model is told about, not the end
	// of the process.
	defer func() {
		if p := recover(); p != nil {
			msg := fmt.Sprintf("panic: %v", p)
			r.Content, r.Err = "error: "+msg, msg
		}
	}()
	var out string
	var err error
	switch {
	case ctx.Err() != nil:
	case a.local[c.Name].Run != nil:
		out, err = a.local[c.Name].Run(ctx, c.Arguments)
	case a.remote[c.Name]:
		out, err = a.manager.CallTool(ctx, c.Name, c.Arguments)
	default:
		err = fmt.Errorf("unknown tool %s", c.Name)
	}
	switch {
	case ctx.Err() != nil:
		r.Content, r.Err = interrupted, ctx.Err().Error()
	case err != nil:
		r.Content, r.Err = "error: "+err.Error(), err.Error()
	default:
		r.Content = truncate(out, a.maxToolResult)
	}
	return r
}

// truncate cuts s to at most max bytes, on a UTF-8 boundary, and tells
// how many bytes it had.
func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + fmt.Sprintf("\n[truncated, %d bytes]", len(s))
}

func toolMessage(r ToolResult) llm.Message {
	return llm.Message{Role: "tool", Content: r.Content, ToolID: r.ID, Name: r.Name}
}
