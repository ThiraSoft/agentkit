package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewProviderPrefersConfigOverEnv(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "env")
	p, err := NewProvider(Config{Provider: ProviderGemini, Model: "m", APIKey: "cle", BaseURL: "http://gemini.test"})
	if err != nil {
		t.Fatal(err)
	}
	g, ok := UnwrapProvider(p).(*geminiProvider)
	if !ok || g.apiKey != "cle" || g.baseURL != "http://gemini.test" {
		t.Fatalf("provider %#v, want the key and URL of the config", UnwrapProvider(p))
	}

	p, err = NewProvider(Config{Provider: ProviderGemini, Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if g := UnwrapProvider(p).(*geminiProvider); g.apiKey != "env" {
		t.Fatalf("apiKey %q, want the environment's when the config has none", g.apiKey)
	}
}

func TestNewProviderOpenAICompatWithBaseURL(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"salut\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	// No custom-models.json: BaseURL is enough.
	p, err := NewProvider(Config{Provider: ProviderOpenAICompat, Model: "default", BaseURL: srv.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hé"}}, nil, func(string) error { return nil })
	if err != nil || msg.Content != "salut" {
		t.Fatalf("stream %+v, %v", msg, err)
	}
}

func TestNewProviderUnknown(t *testing.T) {
	if _, err := NewProvider(Config{Provider: "inconnu", Model: "m"}); err == nil {
		t.Fatal("an unknown provider was accepted")
	}
}

func TestNewProviderKnown(t *testing.T) {
	for _, p := range []ProviderType{ProviderOpenAI, ProviderGemini, ProviderOllama, ProviderAnthropic, ProviderMistral, ProviderLlamaCpp} {
		got, err := NewProvider(Config{Provider: p, Model: "m"})
		if err != nil {
			t.Errorf("NewProvider(%s): %v", p, err)
			continue
		}
		if got.Name() != string(p) || got.ModelName() != "m" {
			t.Errorf("NewProvider(%s): Name %q, ModelName %q", p, got.Name(), got.ModelName())
		}
	}
}

func TestNewProviderOpenAICompatNeedsBaseURL(t *testing.T) {
	_, err := NewProvider(Config{Provider: ProviderOpenAICompat, Model: "m"})
	if err == nil || !strings.Contains(err.Error(), "BaseURL") {
		t.Fatalf("err %v, want BaseURL required", err)
	}
}

func TestNewProviderKnowsNoDeploymentProvider(t *testing.T) {
	for _, name := range []string{"vllm", "jimmy", "omlx", "remote", "claude-code"} {
		_, err := NewProvider(Config{Provider: ProviderType(name), Model: "m"})
		if err == nil || !strings.Contains(err.Error(), "unknown provider") {
			t.Errorf("%s: err %v, want unknown provider", name, err)
		}
	}
}

func TestOllamaOptionsComeFromConfig(t *testing.T) {
	p, err := NewProvider(Config{Provider: ProviderOllama, Model: "m", OllamaNumCtx: 16384, OllamaNumPredict: 512})
	if err != nil {
		t.Fatal(err)
	}
	o := UnwrapProvider(p).(*ollamaProvider)
	if o.numCtx != 16384 || o.numPredict != 512 {
		t.Fatalf("numCtx %d, numPredict %d", o.numCtx, o.numPredict)
	}
}

// ExtraBody goes into the request, after the fields Config sets, which it
// can override.
func TestNewProviderExtraBody(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	temp := 0.2
	p, err := NewProvider(Config{Provider: ProviderOpenAICompat, BaseURL: srv.URL + "/v1", Temperature: &temp,
		ExtraBody: map[string]any{"temperature": 0.7, "chat_template_kwargs": map[string]any{"enable_thinking": true}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil); err != nil {
		t.Fatal(err)
	}
	kwargs, _ := body["chat_template_kwargs"].(map[string]any)
	if body["temperature"] != 0.7 || kwargs["enable_thinking"] != true {
		t.Fatalf("body %v", body)
	}
}
