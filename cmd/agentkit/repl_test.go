package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/ThiraSoft/agentkit"
	"github.com/ThiraSoft/agentkit/llm"
)

// canned answers each model call with the next of its replies.
type canned struct {
	replies []string
	calls   int
}

func (c *canned) Stream(_ context.Context, _ []llm.Message, _ []llm.Tool, onChunk func(string) error) (*llm.Message, error) {
	r := c.replies[c.calls]
	c.calls++
	if err := onChunk(r); err != nil {
		return nil, err
	}
	return &llm.Message{Role: "assistant", Content: r, Usage: &llm.Usage{InputTokens: 10, OutputTokens: 2}}, nil
}

func (c *canned) Chat(context.Context, []llm.Message, []llm.Tool) (*llm.Message, error) {
	return nil, errors.New("canned: Chat is not used")
}
func (c *canned) Name() string      { return "canned" }
func (c *canned) ModelName() string { return "canned" }

func TestREPL(t *testing.T) {
	agent, err := agentkit.New(context.Background(), agentkit.Config{ProviderImpl: &canned{replies: []string{"hello there", "again"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Close()
	var out, errOut bytes.Buffer
	r := &repl{
		agent:   agent,
		conv:    agent.NewConversation("s"),
		out:     &out,
		errOut:  &errOut,
		turnCtx: func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
	}
	if err := r.run(strings.NewReader("hi\n/usage\n\n/reset\nyo\n/tools\n/quit\nnever sent\n")); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "hello there") || !strings.Contains(out.String(), "again") {
		t.Fatalf("out %q", out.String())
	}
	for _, want := range []string{"input 10, output 2, cache read 0, cache write 0", "(history cleared)", "(no tools)"} {
		if !strings.Contains(errOut.String(), want) {
			t.Errorf("errOut %q lacks %q", errOut.String(), want)
		}
	}
	// system, "yo", "again": /reset cleared the first exchange, /quit
	// stopped before the last line.
	if n := len(r.conv.Messages()); n != 3 {
		t.Fatalf("history has %d messages, want 3", n)
	}
}

func TestLoadConfigFlags(t *testing.T) {
	if _, err := loadConfig("", "", "", ""); err == nil {
		t.Error("no -config and no -provider accepted")
	}
	if _, err := loadConfig("agent.json", "openai", "", ""); err == nil {
		t.Error("-config with -provider accepted")
	}
	cfg, err := loadConfig("", "openai-compat", "m", "http://localhost:8080/v1")
	if err != nil || cfg.Provider != "openai-compat" || cfg.Model != "m" || cfg.BaseURL != "http://localhost:8080/v1" {
		t.Fatalf("cfg %+v, err %v", cfg, err)
	}
}
