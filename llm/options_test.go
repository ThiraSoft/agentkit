package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// captured starts a server that keeps the JSON body of the last request
// and answers every request with a stream that ends at once, which every
// provider reads as an empty answer.
func captured(t *testing.T) (*httptest.Server, func() map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		body = b
		mu.Unlock()
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return body
	}
}

// send streams one user message through a provider built from cfg, with
// BaseURL pointing at a captured server, and returns the body it received.
func send(t *testing.T, cfg Config) map[string]any {
	t.Helper()
	return sendWith(t, cfg, "")
}

// sendWith is send with a system prompt first, when system is not empty.
func sendWith(t *testing.T, cfg Config, system string) map[string]any {
	t.Helper()
	srv, body := captured(t)
	cfg.BaseURL = srv.URL
	cfg.APIKey = "k"
	if cfg.Model == "" {
		cfg.Model = "m"
	}
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var msgs []Message
	if system != "" {
		msgs = append(msgs, Message{Role: "system", Content: system})
	}
	msgs = append(msgs, Message{Role: "user", Content: "hi"})
	if _, err := p.Stream(context.Background(), msgs, nil, func(string) error { return nil }); err != nil {
		t.Fatalf("%s: %v", cfg.Provider, err)
	}
	return body()
}

func TestTemperatureAndMaxTokensReachEveryProvider(t *testing.T) {
	temp := 0.3
	compat := func(b map[string]any) bool {
		return b["temperature"] == 0.3 && b["max_tokens"] == 100.0 && b["max_completion_tokens"] == nil
	}
	cases := []struct {
		provider ProviderType
		ok       func(map[string]any) bool
	}{
		{ProviderOpenAI, func(b map[string]any) bool {
			return b["temperature"] == 0.3 && b["max_completion_tokens"] == 100.0 && b["max_tokens"] == nil
		}},
		{ProviderOpenAICompat, compat},
		{ProviderMistral, compat},
		{ProviderLlamaCpp, compat},
		{ProviderAnthropic, func(b map[string]any) bool {
			return b["temperature"] == 0.3 && b["max_tokens"] == 100.0
		}},
		{ProviderGemini, func(b map[string]any) bool {
			gc, _ := b["generationConfig"].(map[string]any)
			return gc["temperature"] == 0.3 && gc["maxOutputTokens"] == 100.0
		}},
		{ProviderOllama, func(b map[string]any) bool {
			o, _ := b["options"].(map[string]any)
			return o["temperature"] == 0.3 && o["num_predict"] == 100.0
		}},
	}
	for _, c := range cases {
		b := send(t, Config{Provider: c.provider, Temperature: &temp, MaxTokens: 100})
		if !c.ok(b) {
			t.Errorf("%s: body %v", c.provider, b)
		}
	}
}

func TestProvidersKeepTheirDefaultsWithoutOptions(t *testing.T) {
	for _, p := range []ProviderType{ProviderOpenAI, ProviderOpenAICompat, ProviderMistral, ProviderLlamaCpp} {
		b := send(t, Config{Provider: p})
		if _, has := b["temperature"]; has {
			t.Errorf("%s sent a temperature: %v", p, b)
		}
		if _, has := b["max_tokens"]; has {
			t.Errorf("%s sent max_tokens: %v", p, b)
		}
	}
	if b := send(t, Config{Provider: ProviderAnthropic}); b["max_tokens"] != 32000.0 || b["temperature"] != nil {
		t.Errorf("anthropic body %v", b)
	}
	if b := send(t, Config{Provider: ProviderGemini}); b["generationConfig"] != nil {
		t.Errorf("gemini body %v", b)
	}
	if b := send(t, Config{Provider: ProviderOllama}); b["options"] != nil {
		t.Errorf("ollama body %v", b)
	}
}

func TestOllamaNumPredictWinsOverMaxTokens(t *testing.T) {
	b := send(t, Config{Provider: ProviderOllama, OllamaNumPredict: 512, MaxTokens: 100})
	if o, _ := b["options"].(map[string]any); o["num_predict"] != 512.0 {
		t.Fatalf("options %v", b["options"])
	}
}
