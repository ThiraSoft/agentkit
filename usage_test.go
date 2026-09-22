package agentkit

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/ThiraSoft/agentkit/llm"
)

func TestTurnSumsUsage(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "noop", `{}`)}, usage: &llm.Usage{InputTokens: 100, OutputTokens: 10, CacheReadTokens: 80}},
		{chunks: []string{"done"}, usage: &llm.Usage{InputTokens: 120, OutputTokens: 5, CacheWriteTokens: 20}},
	}}
	noop := echoTool("noop", func(context.Context, json.RawMessage) (string, error) { return "", nil })
	a, err := New(context.Background(), Config{ProviderImpl: f, Tools: []Tool{noop}})
	if err != nil {
		t.Fatal(err)
	}
	turn, err := a.NewConversation("s").Send(context.Background(), "go", Hooks{})
	want := llm.Usage{InputTokens: 220, OutputTokens: 15, CacheReadTokens: 80, CacheWriteTokens: 20}
	if err != nil || turn.Usage != want {
		t.Fatalf("usage %+v, err %v", turn.Usage, err)
	}
}
