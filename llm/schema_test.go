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

func TestFunctionDefSchemaReplacesParameters(t *testing.T) {
	d := FunctionDef{Name: "f", Description: "d", Parameters: ToolParams{
		Type:       "object",
		Properties: ToolProperties{"a": {Type: "string"}},
	}}
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"string","description":""}}}}`
	if string(b) != want {
		t.Fatalf("without Schema:\n got %s\nwant %s", b, want)
	}

	d.Schema = json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer","minimum":1}}}`)
	b, err = json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	want = `{"name":"f","description":"d","parameters":{"type":"object","properties":{"n":{"type":"integer","minimum":1}}}}`
	if string(b) != want {
		t.Fatalf("with Schema:\n got %s\nwant %s", b, want)
	}
	if string(d.JSONSchema()) != string(d.Schema) {
		t.Fatalf("JSONSchema %s", d.JSONSchema())
	}
}

func TestAnthropicAndGeminiSendTheRawSchema(t *testing.T) {
	schema := json.RawMessage(`{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"n":{"type":"integer","minimum":1}}}`)
	tools := []Tool{{Type: "function", Function: FunctionDef{Name: "f", Description: "d", Schema: schema}}}
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body = nil
		_ = json.NewDecoder(r.Body).Decode(&body)
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n")
			return
		}
		fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n")
	}))
	defer srv.Close()
	msgs := []Message{{Role: "user", Content: "hi"}}
	noop := func(string) error { return nil }

	a, err := NewProvider(Config{Provider: ProviderAnthropic, Model: "m", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Stream(context.Background(), msgs, tools, noop); err != nil {
		t.Fatal(err)
	}
	in := body["tools"].([]any)[0].(map[string]any)["input_schema"].(map[string]any)
	if in["properties"].(map[string]any)["n"].(map[string]any)["minimum"] != 1.0 {
		t.Fatalf("anthropic input_schema %v", in)
	}

	g, err := NewProvider(Config{Provider: ProviderGemini, Model: "m", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Stream(context.Background(), msgs, tools, noop); err != nil {
		t.Fatal(err)
	}
	decl := body["tools"].([]any)[0].(map[string]any)["function_declarations"].([]any)[0].(map[string]any)
	if _, has := decl["parameters"]; has {
		t.Fatalf("gemini got parameters beside parametersJsonSchema: %v", decl)
	}
	ps := decl["parametersJsonSchema"].(map[string]any)
	if _, has := ps["$schema"]; has {
		t.Fatalf("gemini got $schema: %v", ps)
	}
	if ps["properties"].(map[string]any)["n"].(map[string]any)["minimum"] != 1.0 {
		t.Fatalf("gemini parametersJsonSchema %v", ps)
	}
}

func TestFunctionDefRoundTrip(t *testing.T) {
	rich := Tool{Type: "function", Function: FunctionDef{Name: "f", Description: "d",
		Schema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer","default":10},"q":{"type":["string","null"]}}}`)}}
	b, err := json.Marshal(rich)
	if err != nil {
		t.Fatal(err)
	}
	var back Tool
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("decoding %s: %v", b, err)
	}
	if back.Function.Name != "f" || back.Function.Description != "d" || string(back.Function.Schema) != string(rich.Function.Schema) {
		t.Fatalf("back %+v, Schema %s", back.Function, back.Function.Schema)
	}

	plain := Tool{Type: "function", Function: FunctionDef{Name: "g", Parameters: ToolParams{
		Type: "object", Properties: ToolProperties{"a": {Type: "string", Description: "x"}}, Required: []string{"a"}}}}
	b, err = json.Marshal(plain)
	if err != nil {
		t.Fatal(err)
	}
	back = Tool{}
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Function.Parameters.Properties["a"].Description != "x" || len(back.Function.Parameters.Required) != 1 {
		t.Fatalf("Parameters lost: %+v", back.Function.Parameters)
	}
}
