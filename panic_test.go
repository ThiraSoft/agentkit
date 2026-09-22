package agentkit

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/ThiraSoft/agentkit/llm"
)

func TestToolPanicIsReportedToTheModel(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "boom", `{}`)}},
		{chunks: []string{"recovered"}},
	}}
	boom := echoTool("boom", func(context.Context, json.RawMessage) (string, error) { panic("kaboom") })
	a, err := New(context.Background(), Config{ProviderImpl: f, Tools: []Tool{boom}})
	if err != nil {
		t.Fatal(err)
	}
	var got ToolResult
	turn, err := a.NewConversation("s").Send(context.Background(), "go", Hooks{
		OnToolResult: func(r ToolResult) { got = r },
	})
	if err != nil || turn.Text != "recovered" {
		t.Fatalf("turn %+v, err %v", turn, err)
	}
	if got.Err != "panic: kaboom" || got.Content != "error: panic: kaboom" {
		t.Fatalf("result %+v", got)
	}
	// system, user, assistant with the call, tool result
	if m := f.request(1)[3]; m.Role != "tool" || m.Content != "error: panic: kaboom" {
		t.Fatalf("the model saw %+v", m)
	}
}
