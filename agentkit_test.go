package agentkit

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ThiraSoft/agentkit/mcp"
)

func ok(context.Context, json.RawMessage) (string, error) { return "ok", nil }

func TestNewExposesTheTools(t *testing.T) {
	f := &fakeProvider{}
	a, err := New(context.Background(), Config{ProviderImpl: f, Tools: []Tool{echoTool("a", ok), echoTool("b", ok)}})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if got := strings.Join(a.Tools(), ","); got != "a,b" {
		t.Fatalf("tools %q, want a,b", got)
	}
	if a.maxSteps != 20 || a.maxToolResult != 32768 {
		t.Fatalf("defaults %d, %d", a.maxSteps, a.maxToolResult)
	}
}

func TestNewRefuses(t *testing.T) {
	f := &fakeProvider{}
	cases := map[string]Config{
		"duplicate":        {ProviderImpl: f, Tools: []Tool{echoTool("a", ok), echoTool("a", ok)}},
		"missing name":     {ProviderImpl: f, Tools: []Tool{echoTool("", ok)}},
		"missing Run":      {ProviderImpl: f, Tools: []Tool{echoTool("a", nil)}},
		"unknown provider": {Provider: "unknown", Model: "m"},
		"failing MCP":      {ProviderImpl: f, MCP: []mcp.ServerConfig{{Name: "absent", Transport: "stdio", Command: "/does/not/exist"}}},
	}
	for name, cfg := range cases {
		if a, err := New(context.Background(), cfg); err == nil {
			a.Close()
			t.Errorf("%s: New succeeded", name)
		}
	}
}
