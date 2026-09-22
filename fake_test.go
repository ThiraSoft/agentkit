package agentkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/ThiraSoft/agentkit/llm"
)

// fakeReply is a reply from the fake model: its text chunks, tool calls,
// and optionally a block or an error.
type fakeReply struct {
	chunks []string
	calls  []llm.ToolCall
	block  chan struct{} // non-nil: waits for closure or cancellation
	err    error
}

// fakeProvider replays its replies in order, one per model call, and
// records the requests it received.
type fakeProvider struct {
	mu       sync.Mutex
	replies  []fakeReply
	requests [][]llm.Message
	tools    [][]llm.Tool
}

func (f *fakeProvider) Stream(ctx context.Context, msgs []llm.Message, tools []llm.Tool, onChunk func(string) error) (*llm.Message, error) {
	f.mu.Lock()
	f.requests = append(f.requests, msgs)
	f.tools = append(f.tools, tools)
	i := len(f.requests) - 1
	f.mu.Unlock()
	if i >= len(f.replies) {
		return nil, fmt.Errorf("fake: no reply %d", i)
	}
	r := f.replies[i]
	msg := &llm.Message{Role: "assistant"}
	for _, c := range r.chunks {
		msg.Content += c
		if err := onChunk(c); err != nil {
			return msg, err
		}
	}
	if r.block != nil {
		select {
		case <-r.block:
		case <-ctx.Done():
			select {
			case <-r.block:
			default:
				return msg, ctx.Err()
			}
		}
	}
	msg.ToolCalls = r.calls
	if r.err != nil {
		return msg, r.err
	}
	return msg, nil
}

func (f *fakeProvider) Chat(context.Context, []llm.Message, []llm.Tool) (*llm.Message, error) {
	return nil, errors.New("fake: Chat is not used")
}
func (f *fakeProvider) Name() string      { return "fake" }
func (f *fakeProvider) ModelName() string { return "fake" }

func (f *fakeProvider) request(i int) []llm.Message {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[i]
}

func call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: json.RawMessage(args)}}
}

func echoTool(name string, run func(context.Context, json.RawMessage) (string, error)) Tool {
	return Tool{
		Name:        name,
		Description: "test tool " + name,
		Parameters:  llm.ToolParams{Type: "object", Properties: llm.ToolProperties{}},
		Run:         run,
	}
}
