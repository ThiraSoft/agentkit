package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Each provider gets the knob its model takes, both ways; without Thinking,
// nothing is sent.
func TestThinkingReachesEveryProvider(t *testing.T) {
	on, off := true, false
	kwargs := func(b map[string]any) any {
		kw, _ := b["chat_template_kwargs"].(map[string]any)
		return kw["enable_thinking"]
	}
	gemini := func(b map[string]any) any {
		gc, _ := b["generationConfig"].(map[string]any)
		return gc["thinkingConfig"]
	}
	field := func(name string) func(map[string]any) any {
		return func(b map[string]any) any { return b[name] }
	}
	effort := func(b map[string]any) any {
		oc, _ := b["output_config"].(map[string]any)
		return oc["effort"]
	}
	cases := []struct {
		provider ProviderType
		model    string
		thinking *bool
		read     func(map[string]any) any
		want     string
	}{
		{ProviderLlamaCpp, "m", &on, kwargs, "true"},
		{ProviderLlamaCpp, "m", &off, kwargs, "false"},
		{ProviderLlamaCpp, "m", nil, kwargs, "<nil>"},
		{ProviderOpenAICompat, "m", &off, kwargs, "false"},
		{ProviderOllama, "qwen3", &on, field("think"), "true"},
		{ProviderOllama, "qwen3", &off, field("think"), "false"},
		{ProviderOllama, "gpt-oss:20b", &off, field("think"), "low"},
		{ProviderOllama, "qwen3", nil, field("think"), "<nil>"},
		{ProviderOpenAI, "gpt-5.1", &off, field("reasoning_effort"), "none"},
		{ProviderOpenAI, "gpt-5-mini", &off, field("reasoning_effort"), "minimal"},
		{ProviderOpenAI, "o3", &on, field("reasoning_effort"), "high"},
		{ProviderOpenAI, "gpt-4o", &on, field("reasoning_effort"), "<nil>"},
		{ProviderOpenAI, "gpt-5-chat-latest", &off, field("reasoning_effort"), "<nil>"},
		{ProviderMistral, "mistral-large", &on, field("reasoning_effort"), "<nil>"},
		{ProviderGemini, "gemini-2.5-flash", &off, gemini, "map[thinkingBudget:0]"},
		{ProviderGemini, "gemini-2.5-flash", &on, gemini, "map[includeThoughts:true thinkingBudget:-1]"},
		{ProviderGemini, "gemini-2.5-pro", &off, gemini, "<nil>"},
		{ProviderGemini, "gemini-3-pro-preview", &off, gemini, "map[includeThoughts:false thinkingLevel:low]"},
		{ProviderGemini, "gemini-3-flash", &off, gemini, "map[includeThoughts:false thinkingLevel:minimal]"},
		{ProviderGemini, "gemini-3-flash", &on, gemini, "map[includeThoughts:true thinkingLevel:high]"},
		{ProviderAnthropic, "claude-opus-4-8", &on, field("thinking"), "map[display:summarized type:adaptive]"},
		{ProviderAnthropic, "claude-opus-4-8", &off, field("thinking"), "map[type:disabled]"},
		{ProviderAnthropic, "claude-sonnet-5", &off, field("thinking"), "map[type:disabled]"},
		{ProviderAnthropic, "claude-sonnet-5-5", &off, field("thinking"), "map[type:between_tools]"},
		{ProviderAnthropic, "claude-opus-5-5", &off, field("thinking"), "<nil>"},
		{ProviderAnthropic, "claude-opus-5-5", &off, effort, "low"},
		{ProviderAnthropic, "claude-fable-5-1", &on, field("thinking"), "map[display:summarized type:adaptive]"},
		{ProviderAnthropic, "claude-haiku-4-5", &on, field("thinking"), "map[budget_tokens:16000 type:enabled]"},
		{ProviderAnthropic, "claude-sonnet-4-20250514", &off, field("thinking"), "<nil>"},
		{ProviderAnthropic, "claude-3-5-sonnet-latest", &on, field("thinking"), "<nil>"},
		{ProviderAnthropic, "claude-opus-4-8", nil, field("thinking"), "<nil>"},
	}
	for _, c := range cases {
		b := send(t, Config{Provider: c.provider, Model: c.model, Thinking: c.thinking})
		if got := fmt.Sprint(c.read(b)); got != c.want {
			t.Errorf("%s %s thinking=%v: got %s, want %s", c.provider, c.model, deref(c.thinking), got, c.want)
		}
	}
}

func deref(b *bool) any {
	if b == nil {
		return nil
	}
	return *b
}

// Thinking keeps what ExtraBody puts in chat_template_kwargs.
func TestThinkingKeepsTheTemplateKwargs(t *testing.T) {
	off := false
	b := send(t, Config{Provider: ProviderLlamaCpp, Thinking: &off,
		ExtraBody: map[string]any{"chat_template_kwargs": map[string]any{"preserve_thinking": true}}})
	kw := b["chat_template_kwargs"].(map[string]any)
	if kw["enable_thinking"] != false || kw["preserve_thinking"] != true {
		t.Fatalf("chat_template_kwargs %v", kw)
	}
}

// A Claude model thinking with a budget takes no temperature.
func TestClaudeBudgetDropsTheTemperature(t *testing.T) {
	on, temp := true, 0.3
	b := send(t, Config{Provider: ProviderAnthropic, Model: "claude-haiku-4-5", Thinking: &on, Temperature: &temp})
	if _, has := b["temperature"]; has {
		t.Fatalf("temperature sent with a budget: %v", b)
	}
}

// Claude's thinking is read from the stream, kept as Reasoning and as its
// blocks, and the blocks go back while the turn is going on, not after.
func TestClaudeThinkingBlocksGoBackWithTheTurn(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		bodies = append(bodies, b)
		for _, e := range []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"see"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"ls"}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			`{"type":"content_block_stop","index":2}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", e)
		}
	}))
	t.Cleanup(srv.Close)
	on := true
	p, err := NewProvider(Config{Provider: ProviderAnthropic, Model: "claude-opus-4-8", BaseURL: srv.URL, APIKey: "k", Thinking: &on})
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: "user", Content: "list"}}
	answer, err := p.Stream(context.Background(), msgs, nil, func(string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if answer.Reasoning != "let me see" || len(answer.ThinkingBlocks) != 2 || len(answer.ToolCalls) != 1 {
		t.Fatalf("answer %+v", answer)
	}

	// Within the turn: the blocks go first, as they came.
	msgs = append(msgs, *answer, Message{Role: "tool", ToolID: "t1", Content: "a b"})
	if _, err := p.Stream(context.Background(), msgs, nil, func(string) error { return nil }); err != nil {
		t.Fatal(err)
	}
	content := bodies[1]["messages"].([]any)[1].(map[string]any)["content"].([]any)
	first, second := content[0].(map[string]any), content[1].(map[string]any)
	if first["type"] != "thinking" || first["thinking"] != "let me see" || first["signature"] != "sig" ||
		second["type"] != "redacted_thinking" || second["data"] != "opaque" || content[2].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("assistant content %v", content)
	}

	// After a new question, the past turn goes back without them.
	msgs = append(msgs, Message{Role: "assistant", Content: "a, b"}, Message{Role: "user", Content: "thanks"})
	if _, err := p.Stream(context.Background(), msgs, nil, func(string) error { return nil }); err != nil {
		t.Fatal(err)
	}
	content = bodies[2]["messages"].([]any)[1].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("past turn content %v", content)
	}
}

// Gemini's thought summaries are its Reasoning, not its answer.
func TestGeminiThoughtsAreReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"pondering","thought":true},{"text":"Answer."}]}}]}`+"\n\n")
	}))
	t.Cleanup(srv.Close)
	p, _ := NewProvider(Config{Provider: ProviderGemini, Model: "gemini-2.5-flash", BaseURL: srv.URL, APIKey: "k"})
	var streamed string
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, func(s string) error { streamed += s; return nil })
	if err != nil || msg.Content != "Answer." || msg.Reasoning != "pondering" || streamed != "Answer." {
		t.Fatalf("message %+v, streamed %q, %v", msg, streamed, err)
	}
}

// Ollama's thinking is its Reasoning, streamed or not.
func TestOllamaThinkingIsReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		if b["stream"] == true {
			fmt.Fprintln(w, `{"message":{"role":"assistant","thinking":"hm"}}`)
			fmt.Fprintln(w, `{"message":{"role":"assistant","content":"Yes."},"done":true}`)
			return
		}
		fmt.Fprint(w, `{"message":{"role":"assistant","content":"Yes.","thinking":"hm"}}`)
	}))
	t.Cleanup(srv.Close)
	p, _ := NewProvider(Config{Provider: ProviderOllama, Model: "qwen3", BaseURL: srv.URL})
	msgs := []Message{{Role: "user", Content: "hi"}}
	msg, err := p.Stream(context.Background(), msgs, nil, func(string) error { return nil })
	if err != nil || msg.Content != "Yes." || msg.Reasoning != "hm" {
		t.Fatalf("stream %+v, %v", msg, err)
	}
	msg, err = p.Chat(context.Background(), msgs, nil)
	if err != nil || msg.Content != "Yes." || msg.Reasoning != "hm" {
		t.Fatalf("chat %+v, %v", msg, err)
	}
}
