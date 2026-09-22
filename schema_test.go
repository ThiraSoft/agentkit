package agentkit

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

type weatherArgs struct {
	City string `json:"city" jsonschema:"the city"`
	Days int    `json:"days,omitempty"`
}

func TestNewToolSchemaAndArguments(t *testing.T) {
	var got weatherArgs
	tool, err := NewTool("weather", "Forecast.", func(_ context.Context, a weatherArgs) (string, error) {
		got = a
		return "sunny", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if tool.Name != "weather" || tool.Description != "Forecast." {
		t.Fatalf("tool %q %q", tool.Name, tool.Description)
	}
	var s struct {
		Type       string                    `json:"type"`
		Properties map[string]map[string]any `json:"properties"`
		Required   []string                  `json:"required"`
	}
	if err := json.Unmarshal(tool.Schema, &s); err != nil {
		t.Fatal(err)
	}
	if s.Type != "object" || s.Properties["city"]["description"] != "the city" || s.Properties["days"]["type"] != "integer" {
		t.Fatalf("schema %s", tool.Schema)
	}
	if !slices.Equal(s.Required, []string{"city"}) {
		t.Fatalf("required %v", s.Required)
	}
	out, err := tool.Run(context.Background(), json.RawMessage(`{"city":"Lyon","days":2}`))
	if err != nil || out != "sunny" || got != (weatherArgs{City: "Lyon", Days: 2}) {
		t.Fatalf("run %q, %v, args %+v", out, err, got)
	}
	if _, err := tool.Run(context.Background(), json.RawMessage(`{"city":3}`)); err == nil {
		t.Fatal("arguments that do not decode were accepted")
	}
}

func TestSchemaReachesTheProvider(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{{chunks: []string{"done"}}}}
	schema := json.RawMessage(`{"type":"object","properties":{"unit":{"type":"string","enum":["c","f"]}}}`)
	tool := echoTool("temp", func(context.Context, json.RawMessage) (string, error) { return "", nil })
	tool.Schema = schema
	a, err := New(context.Background(), Config{ProviderImpl: f, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.NewConversation("s").Send(context.Background(), "hi", Hooks{}); err != nil {
		t.Fatal(err)
	}
	def := f.tools[0][0]
	if string(def.Function.Schema) != string(schema) {
		t.Fatalf("Schema %s", def.Function.Schema)
	}
	b, err := json.Marshal(def)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"enum":["c","f"]`) {
		t.Fatalf("encoded %s", b)
	}
}

func TestNewRejectsASchemaThatIsNotAnObject(t *testing.T) {
	for _, s := range []string{`[1]`, `null`, `{`, `"x"`} {
		tool := echoTool("x", func(context.Context, json.RawMessage) (string, error) { return "", nil })
		tool.Schema = json.RawMessage(s)
		if a, err := New(context.Background(), Config{ProviderImpl: &fakeProvider{}, Tools: []Tool{tool}}); err == nil {
			a.Close()
			t.Errorf("schema %s accepted", s)
		}
	}
}
