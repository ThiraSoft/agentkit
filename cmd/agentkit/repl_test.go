package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
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

// A session is saved after each turn and read back whole, so a run of -p
// can take up the conversation the previous one left.
func TestSessionTakenUp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "task.json")
	if msgs, err := loadSession(path); err != nil || msgs != nil {
		t.Fatalf("an absent session: %v %v", msgs, err)
	}
	run := func(system, text string, reply string) *agentkit.Conversation {
		t.Helper()
		agent, err := agentkit.New(context.Background(), agentkit.Config{ProviderImpl: &canned{replies: []string{reply}}})
		if err != nil {
			t.Fatal(err)
		}
		defer agent.Close()
		history, err := loadSession(path)
		if err != nil {
			t.Fatal(err)
		}
		conv := agent.NewConversation(system)
		if len(history) > 0 {
			conv = agent.NewConversation(history[0].Content, history[1:]...)
		}
		var out, errOut bytes.Buffer
		r := &repl{agent: agent, conv: conv, out: &out, errOut: &errOut,
			turnCtx: func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
			save:    func() error { return saveSession(path, conv.Messages()) }}
		if !r.send(text) {
			t.Fatalf("turn failed: %s", errOut.String())
		}
		return conv
	}
	run("first system", "do it", "done")
	conv := run("second system", "and fix that", "fixed")
	msgs := conv.Messages()
	var got []string
	for _, m := range msgs {
		got = append(got, m.Role+":"+m.Content)
	}
	want := "system:first system|user:do it|assistant:done|user:and fix that|assistant:fixed"
	if strings.Join(got, "|") != want {
		t.Fatalf("history %s", strings.Join(got, "|"))
	}
	os.WriteFile(path, []byte(`[{"role":"user","content":"x"}]`), 0o644)
	if _, err := loadSession(path); err == nil {
		t.Fatal("a session with no system prompt was read")
	}
}

func TestSystemPrompt(t *testing.T) {
	if systemPrompt("flag", "config") != "flag" || systemPrompt("", "config") != "config" || systemPrompt("", "") != "You are a helpful assistant." {
		t.Fatal("wrong precedence")
	}
}

// With -p, stdout gets the answer alone, and what came before goes nowhere,
// or to stderr with -v.
func TestFinalAnswerAlone(t *testing.T) {
	tool := agentkit.Tool{Name: "look", Run: func(context.Context, json.RawMessage) (string, error) { return "seen", nil }}
	provider := &calling{}
	agent, err := agentkit.New(context.Background(), agentkit.Config{ProviderImpl: provider, Tools: []agentkit.Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Close()
	var out, errOut bytes.Buffer
	r := &repl{agent: agent, conv: agent.NewConversation("s"), out: &out, errOut: &errOut, final: true,
		turnCtx: func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) }}
	if !r.send("go") {
		t.Fatalf("turn failed: %s", errOut.String())
	}
	if out.String() != "Report: done.\n" {
		t.Fatalf("out %q", out.String())
	}
	if errOut.Len() != 0 {
		t.Fatalf("errOut %q", errOut.String())
	}
	out.Reset()
	r.verbose = true
	provider.calls = 0
	r.send("again")
	if out.String() != "Report: done.\n" || !strings.Contains(errOut.String(), "Let me look.") || !strings.Contains(errOut.String(), "[look") {
		t.Fatalf("out %q, errOut %q", out.String(), errOut.String())
	}
}

// calling asks for the look tool once, then answers.
type calling struct{ calls int }

func (c *calling) Stream(_ context.Context, _ []llm.Message, _ []llm.Tool, onChunk func(string) error) (*llm.Message, error) {
	c.calls++
	if c.calls == 1 {
		onChunk("Let me look.")
		return &llm.Message{Role: "assistant", Content: "Let me look.", ToolCalls: []llm.ToolCall{{ID: "1", Type: "function", Function: llm.FunctionCall{Name: "look", Arguments: json.RawMessage(`{}`)}}}}, nil
	}
	onChunk("Report: done.")
	return &llm.Message{Role: "assistant", Content: "Report: done."}, nil
}

func (c *calling) Chat(context.Context, []llm.Message, []llm.Tool) (*llm.Message, error) {
	return nil, errors.New("not used")
}
func (c *calling) Name() string      { return "calling" }
func (c *calling) ModelName() string { return "calling" }

// An answer with nothing in it is a failed turn under -p.
func TestFinalAnswerEmpty(t *testing.T) {
	agent, err := agentkit.New(context.Background(), agentkit.Config{ProviderImpl: &canned{replies: []string{""}}})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Close()
	var out, errOut bytes.Buffer
	r := &repl{agent: agent, conv: agent.NewConversation("s"), out: &out, errOut: &errOut, final: true,
		turnCtx: func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) }}
	if r.send("go") || out.Len() != 0 || !strings.Contains(errOut.String(), "without an answer") {
		t.Fatalf("out %q, errOut %q", out.String(), errOut.String())
	}
}
