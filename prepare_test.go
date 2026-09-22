package agentkit

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/ThiraSoft/agentkit/llm"
)

func history() []llm.Message {
	return []llm.Message{
		{Role: "system", Content: "s"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", ToolCalls: []llm.ToolCall{call("1", "t", `{}`)}},
		{Role: "tool", Content: "r", ToolID: "1", Name: "t"},
		{Role: "assistant", Content: "a2"},
		{Role: "user", Content: "u3"},
	}
}

// shape gives each message as role:content, content cut to 8 bytes.
func shape(msgs []llm.Message) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		c := m.Content
		if len(c) > 8 {
			c = c[:8]
		}
		out[i] = m.Role + ":" + c
	}
	return out
}

func TestKeepTurns(t *testing.T) {
	h := history()
	got := shape(KeepTurns(2)(h))
	want := []string{"system:s", "user:u2", "assistant:", "tool:r", "assistant:a2", "user:u3"}
	if !slices.Equal(got, want) {
		t.Fatalf("KeepTurns(2) %v", got)
	}
	if got := shape(KeepTurns(0)(h)); !slices.Equal(got, []string{"system:s", "user:u3"}) {
		t.Fatalf("KeepTurns(0) %v", got)
	}
	if got := KeepTurns(5)(h); len(got) != len(h) {
		t.Fatalf("KeepTurns(5) dropped messages: %v", shape(got))
	}
	if len(h) != 8 || h[1].Content != "u1" {
		t.Fatalf("the history was touched: %v", shape(h))
	}
	noSystem := h[1:]
	if got := shape(KeepTurns(1)(noSystem)); !slices.Equal(got, []string{"user:u3"}) {
		t.Fatalf("KeepTurns(1) without system %v", got)
	}
}

func TestKeepTokens(t *testing.T) {
	h := history()
	h[1].Content = strings.Repeat("x", 4000) // about 1000 tokens
	got := shape(KeepTokens(500)(h))
	want := []string{"system:s", "user:u2", "assistant:", "tool:r", "assistant:a2", "user:u3"}
	if !slices.Equal(got, want) {
		t.Fatalf("KeepTokens(500) %v", got)
	}
	if got := KeepTokens(1_000_000)(h); len(got) != len(h) {
		t.Fatalf("KeepTokens over budget dropped messages: %v", shape(got))
	}
	if got := shape(KeepTokens(0)(h)); !slices.Equal(got, []string{"system:s", "user:u3"}) {
		t.Fatalf("KeepTokens(0) %v", got)
	}
}

func TestEstimateTokens(t *testing.T) {
	m := llm.Message{Role: "user", Content: strings.Repeat("x", 400), Media: []llm.Media{{Type: llm.MediaImage}}}
	if got := EstimateTokens(m); got != 100+1000+4 {
		t.Fatalf("EstimateTokens %d", got)
	}
}

func TestKeepTurnsInASend(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{chunks: []string{"a1"}}, {chunks: []string{"a2"}}}}
	a, err := New(context.Background(), Config{ProviderImpl: f})
	if err != nil {
		t.Fatal(err)
	}
	c := a.NewConversation("s")
	for _, text := range []string{"u1", "u2"} {
		if _, err := c.Send(context.Background(), text, Hooks{Prepare: KeepTurns(1)}); err != nil {
			t.Fatal(err)
		}
	}
	if got := shape(f.request(1)); !slices.Equal(got, []string{"system:s", "user:u2"}) {
		t.Fatalf("the model saw %v", got)
	}
	if n := len(c.Messages()); n != 5 {
		t.Fatalf("history has %d messages, want 5", n)
	}
}
