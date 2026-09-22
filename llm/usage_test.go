package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// replay starts a server that answers every request with body, and keeps
// the JSON of the last request in *got.
func replay(t *testing.T, body string, got *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got != nil {
			*got = nil
			_ = json.NewDecoder(r.Body).Decode(got)
		}
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func stream(t *testing.T, cfg Config) *Message {
	t.Helper()
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, func(string) error { return nil })
	if err != nil {
		t.Fatalf("%s: %v", cfg.Provider, err)
	}
	return msg
}

func TestOpenAICompatUsageAndToolCallOrder(t *testing.T) {
	var body map[string]any
	srv := replay(t, ""+
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":2,\"id\":\"c\",\"function\":{\"name\":\"third\",\"arguments\":\"{}\"}}]}}]}\n\n"+
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n"+
		"data: [DONE]\n\n", &body)

	msg := stream(t, Config{Provider: ProviderOpenAICompat, Model: "m", BaseURL: srv.URL})
	if msg.Usage == nil || *msg.Usage != (Usage{InputTokens: 50, OutputTokens: 7, CacheReadTokens: 32}) {
		t.Fatalf("usage %+v", msg.Usage)
	}
	if len(msg.ToolCalls) != 3 || msg.ToolCalls[0].ID != "a" || msg.ToolCalls[1].ID != "b" || msg.ToolCalls[2].ID != "c" {
		t.Fatalf("tool calls %+v", msg.ToolCalls)
	}
	if so, _ := body["stream_options"].(map[string]any); so["include_usage"] != true {
		t.Fatalf("stream_options %v", body["stream_options"])
	}

	stream(t, Config{Provider: ProviderMistral, Model: "m", BaseURL: srv.URL})
	if _, has := body["stream_options"]; has {
		t.Fatal("mistral asked for stream_options")
	}
}

func TestOpenAICompatChatUsage(t *testing.T) {
	srv := replay(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":9,"completion_tokens":2}}`, nil)
	p, err := NewProvider(Config{Provider: ProviderOpenAICompat, Model: "m", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	msg, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil || msg.Content != "ok" || msg.Usage == nil || *msg.Usage != (Usage{InputTokens: 9, OutputTokens: 2}) {
		t.Fatalf("chat %+v, usage %+v, %v", msg, msg.Usage, err)
	}
}

func TestAnthropicUsage(t *testing.T) {
	srv := replay(t, ""+
		"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":20,\"cache_read_input_tokens\":30,\"output_tokens\":1}}}\n\n"+
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n"+
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n", nil)
	msg := stream(t, Config{Provider: ProviderAnthropic, Model: "m", APIKey: "k", BaseURL: srv.URL})
	want := Usage{InputTokens: 60, OutputTokens: 9, CacheReadTokens: 30, CacheWriteTokens: 20}
	if msg.Content != "ok" || msg.Usage == nil || *msg.Usage != want {
		t.Fatalf("message %+v, usage %+v", msg, msg.Usage)
	}

	chat := replay(t, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`, nil)
	p, err := NewProvider(Config{Provider: ProviderAnthropic, Model: "m", APIKey: "k", BaseURL: chat.URL})
	if err != nil {
		t.Fatal(err)
	}
	m, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil || m.Content != "ok" || m.Usage == nil || *m.Usage != (Usage{InputTokens: 5, OutputTokens: 2}) {
		t.Fatalf("chat %+v, usage %+v, %v", m, m.Usage, err)
	}
}

func TestGeminiUsage(t *testing.T) {
	srv := replay(t, ""+
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"o\"}]}}],\"usageMetadata\":{\"promptTokenCount\":40,\"candidatesTokenCount\":1}}\n\n"+
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"k\"}]}}],\"usageMetadata\":{\"promptTokenCount\":40,\"candidatesTokenCount\":5,\"thoughtsTokenCount\":3,\"cachedContentTokenCount\":16}}\n\n", nil)
	msg := stream(t, Config{Provider: ProviderGemini, Model: "m", APIKey: "k", BaseURL: srv.URL})
	if msg.Content != "ok" || msg.Usage == nil || *msg.Usage != (Usage{InputTokens: 40, OutputTokens: 8, CacheReadTokens: 16}) {
		t.Fatalf("message %+v, usage %+v", msg, msg.Usage)
	}
}

func TestOllamaUsage(t *testing.T) {
	srv := replay(t, ""+
		"{\"message\":{\"content\":\"ok\"},\"done\":false}\n"+
		"{\"message\":{\"content\":\"\"},\"done\":true,\"prompt_eval_count\":12,\"eval_count\":4}\n", nil)
	msg := stream(t, Config{Provider: ProviderOllama, Model: "m", BaseURL: srv.URL})
	if msg.Content != "ok" || msg.Usage == nil || *msg.Usage != (Usage{InputTokens: 12, OutputTokens: 4}) {
		t.Fatalf("message %+v, usage %+v", msg, msg.Usage)
	}
}

func TestUsageAdd(t *testing.T) {
	got := Usage{1, 2, 3, 4}.Add(Usage{10, 20, 30, 40})
	if got != (Usage{11, 22, 33, 44}) {
		t.Fatalf("Add %+v", got)
	}
}
