package mcp

import (
	"testing"

	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestConvertMCPTool_Basic(t *testing.T) {
	mcpTool := &gomcp.Tool{
		Name:        "test_tool",
		Description: "A test tool",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "search query",
				},
				"count": map[string]any{
					"type":        "integer",
					"description": "number of results",
				},
			},
			"required": []any{"query"},
		},
	}

	tool := convertMCPTool(mcpTool)

	if tool.Type != "function" {
		t.Fatalf("expected type 'function', got '%s'", tool.Type)
	}
	if tool.Function.Name != "test_tool" {
		t.Fatalf("expected name 'test_tool', got '%s'", tool.Function.Name)
	}
	if tool.Function.Description != "A test tool" {
		t.Fatalf("unexpected description: %s", tool.Function.Description)
	}

	params := tool.Function.Parameters
	if params.Type != "object" {
		t.Fatalf("expected params type 'object', got '%s'", params.Type)
	}

	queryProp, ok := params.Properties["query"]
	if !ok {
		t.Fatal("expected 'query' property")
	}
	if queryProp.Type != "string" {
		t.Fatalf("expected query type 'string', got '%s'", queryProp.Type)
	}
	if queryProp.Description != "search query" {
		t.Fatalf("unexpected query description: %s", queryProp.Description)
	}

	countProp, ok := params.Properties["count"]
	if !ok {
		t.Fatal("expected 'count' property")
	}
	if countProp.Type != "integer" {
		t.Fatalf("expected count type 'integer', got '%s'", countProp.Type)
	}

	if len(params.Required) != 1 || params.Required[0] != "query" {
		t.Fatalf("unexpected required: %v", params.Required)
	}
}

func TestConvertMCPTool_EmptySchema(t *testing.T) {
	mcpTool := &gomcp.Tool{
		Name:        "no_params",
		Description: "Tool with no params",
	}

	tool := convertMCPTool(mcpTool)

	if tool.Function.Name != "no_params" {
		t.Fatalf("unexpected name: %s", tool.Function.Name)
	}
	if len(tool.Function.Parameters.Properties) != 0 {
		t.Fatalf("expected empty properties, got %d", len(tool.Function.Parameters.Properties))
	}
}

func TestConvertMCPTool_WithExample(t *testing.T) {
	mcpTool := &gomcp.Tool{
		Name:        "tool_with_example",
		Description: "desc",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "file path",
					"example":     "/tmp/test.txt",
					"default":     ".",
				},
			},
		},
	}

	tool := convertMCPTool(mcpTool)
	pathProp := tool.Function.Parameters.Properties["path"]

	if pathProp.Example != "/tmp/test.txt" {
		t.Fatalf("expected example '/tmp/test.txt', got '%s'", pathProp.Example)
	}
	if pathProp.Default != "." {
		t.Fatalf("expected default '.', got '%s'", pathProp.Default)
	}
}
