package llm

import (
	"context"
	"encoding/json"
	"errors"
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

func TestOpenAICompatReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm, \"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning\":\"voyons\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"salut\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	var thought string
	p := NewOpenAICompat(OpenAICompatConfig{
		BaseURL:     srv.URL + "/v1",
		OnReasoning: func(s string) { thought += s },
	})
	var said string
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, func(s string) error { said += s; return nil })
	if err != nil || msg.Content != "salut" || said != "salut" {
		t.Fatalf("stream %+v, said %q, %v", msg, said, err)
	}
	if thought != "hmm, voyons" {
		t.Fatalf("reasoning %q", thought)
	}
}

func TestOpenAICompatBodyPerRequest(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		bodies = append(bodies, b)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	think := false
	p := NewOpenAICompat(OpenAICompatConfig{
		BaseURL:   srv.URL + "/v1",
		ExtraBody: map[string]any{"temperature": 0.2},
		ExtraBodyFunc: func() map[string]any {
			return map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": think}}
		},
	})
	for _, on := range []bool{false, true} {
		think = on
		if _, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	for i, want := range []bool{false, true} {
		kw, _ := bodies[i]["chat_template_kwargs"].(map[string]any)
		if kw["enable_thinking"] != want || bodies[i]["temperature"] != 0.2 {
			t.Fatalf("request %d: %v", i, bodies[i])
		}
	}
}

// An error sent once the stream is open comes back as a StreamError, and none
// of it lands in the answer.
func TestOpenAICompatStreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Je regarde.\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"error\":{\"message\":\"a call it cannot read\",\"type\":\"server_error\"}}\n\n")
	}))
	defer srv.Close()

	p := NewOpenAICompat(OpenAICompatConfig{BaseURL: srv.URL + "/v1", Model: "m", Name: "golem"})
	var shown string
	msg, err := p.Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, func(s string) error { shown += s; return nil })
	var se *StreamError
	if !errors.As(err, &se) || se.Message != "a call it cannot read" || se.Type != "server_error" || se.Provider != "golem" {
		t.Fatalf("err %v", err)
	}
	if msg.Content != "Je regarde." || shown != "Je regarde." {
		t.Fatalf("content %q, shown %q", msg.Content, shown)
	}
}
