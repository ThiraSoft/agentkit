package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewOpenAICompat(t *testing.T) {
	var path, auth, extra string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		auth = r.Header.Get("Authorization")
		extra = r.Header.Get("X-Extra")
		_ = json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"salut\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := NewOpenAICompat(OpenAICompatConfig{
		APIKey:       "k",
		BaseURL:      srv.URL + "/v1",
		Model:        "m",
		Name:         "home",
		ExtraBody:    map[string]any{"temperature": 0.2, "tool_choice": "auto"},
		ExtraHeaders: map[string]string{"X-Extra": "yes"},
	})
	if p.Name() != "home" || p.ModelName() != "m" {
		t.Fatalf("Name %q, ModelName %q", p.Name(), p.ModelName())
	}
	if UnwrapProvider(p) != p {
		t.Fatal("NewOpenAICompat returned a wrapped provider; retry is WithRetry's job")
	}
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, func(string) error { return nil })
	if err != nil || msg.Content != "salut" {
		t.Fatalf("stream %+v, %v", msg, err)
	}
	if path != "/v1/chat/completions" || auth != "Bearer k" || extra != "yes" {
		t.Fatalf("path %q, auth %q, X-Extra %q", path, auth, extra)
	}
	if body["model"] != "m" || body["temperature"] != 0.2 || body["stream"] != true {
		t.Fatalf("body %v", body)
	}
	if _, sent := body["tool_choice"]; sent {
		t.Fatal("tool_choice sent on a request without tools")
	}
}
