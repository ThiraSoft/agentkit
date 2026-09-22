package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ThiraSoft/agentkit/llm"
)

// clientName and clientVersion identify this MCP client during the
// handshake with an MCP server.
const (
	clientName    = "agentkit"
	clientVersion = "v0.1.0"
)

// headerRoundTripper wraps an http.RoundTripper and injects custom headers
// into every outgoing request.
type headerRoundTripper struct {
	next    http.RoundTripper
	headers map[string]string
}

func (h *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	for k, v := range h.headers {
		req.Header.Set(k, v)
	}
	return h.next.RoundTrip(req)
}

// httpClientWithHeaders creates an *http.Client that injects custom headers
// into every request. Returns nil if no headers are provided.
func httpClientWithHeaders(headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return nil
	}
	return &http.Client{
		Transport: &headerRoundTripper{
			next:    http.DefaultTransport,
			headers: headers,
		},
	}
}

// Client is a connection to one MCP server. It lists the server's tools in
// the llm.Tool format and calls them. Build one with Dial or with one of
// the New*Client functions.
type Client struct {
	session *mcp.ClientSession
	mu      sync.RWMutex
	tools   []*mcp.Tool
}

// NewStdioClient creates a new MCP client that communicates via stdio
// with a subprocess running the specified command.
func NewStdioClient(ctx context.Context, command string, args ...string) (*Client, error) {
	client := mcp.NewClient(
		&mcp.Implementation{Name: clientName, Version: clientVersion},
		nil,
	)

	transport := &mcp.CommandTransport{
		Command: exec.Command(command, args...),
	}

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect via stdio: %w", err)
	}

	return &Client{session: session}, nil
}

// NewSSEClient creates a new MCP client that communicates via Server-Sent Events
// with an HTTP server at the specified URL.
func NewSSEClient(ctx context.Context, baseURL string, headers map[string]string) (*Client, error) {
	if !strings.HasSuffix(baseURL, "/sse") {
		baseURL = strings.TrimSuffix(baseURL, "/") + "/sse"
	}

	client := mcp.NewClient(
		&mcp.Implementation{Name: clientName, Version: clientVersion},
		nil,
	)

	transport := &mcp.SSEClientTransport{
		Endpoint:   baseURL,
		HTTPClient: httpClientWithHeaders(headers),
	}

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect via SSE to %s: %w", baseURL, err)
	}

	return &Client{session: session}, nil
}

// NewStreamableClient creates a new MCP client that communicates via StreamableHTTP
// with an HTTP server at the specified URL. This is the modern MCP transport.
func NewStreamableClient(ctx context.Context, baseURL string, headers map[string]string) (*Client, error) {
	client := mcp.NewClient(
		&mcp.Implementation{Name: clientName, Version: clientVersion},
		nil,
	)

	transport := &mcp.StreamableClientTransport{
		Endpoint:   baseURL,
		HTTPClient: httpClientWithHeaders(headers),
	}

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect via StreamableHTTP to %s: %w", baseURL, err)
	}

	return &Client{session: session}, nil
}

// Close terminates the MCP connection and cleans up resources.
func (m *Client) Close() error {
	if m.session != nil {
		return m.session.Close()
	}
	return nil
}

// FetchTools asks the server for its tools and keeps them for
// GetToolDefinitions.
func (m *Client) FetchTools(ctx context.Context) error {
	result, err := m.session.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}
	m.mu.Lock()
	m.tools = result.Tools
	m.mu.Unlock()
	return nil
}

// GetToolDefinitions converts MCP tools to the llm.Tool format.
// This allows the LLM to see and use MCP tools as if they were local tools.
func (m *Client) GetToolDefinitions() []llm.Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var defs []llm.Tool

	for _, mcpTool := range m.tools {
		def := convertMCPTool(mcpTool)
		defs = append(defs, def)
	}

	return defs
}

// CallTool executes a tool on the MCP server with the given arguments.
// Returns the tool result as a string.
func (m *Client) CallTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
	// Parse arguments from JSON
	var arguments map[string]any
	if len(args) > 0 {
		if err := json.Unmarshal(args, &arguments); err != nil {
			return "", fmt.Errorf("failed to parse tool arguments: %w", err)
		}
	}

	// Execute the tool call
	result, err := m.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      name,
		Arguments: arguments,
	})
	if err != nil {
		return "", fmt.Errorf("tool call failed: %w", err)
	}

	// Check for tool-level errors
	if result.IsError {
		for _, content := range result.Content {
			if tc, ok := content.(*mcp.TextContent); ok {
				return "", fmt.Errorf("tool error: %s", tc.Text)
			}
		}
		return "", fmt.Errorf("tool returned an error")
	}

	// Extract text content from the result
	var output string
	for _, content := range result.Content {
		if tc, ok := content.(*mcp.TextContent); ok {
			output += tc.Text
		}
	}

	return output, nil
}

// convertMCPTool converts an MCP tool definition to the llm.Tool format.
func convertMCPTool(mcpTool *mcp.Tool) llm.Tool {
	props := llm.ToolProperties{}
	var required []string

	// InputSchema is any - marshal then unmarshal to extract properties
	schemaBytes, err := json.Marshal(mcpTool.InputSchema)
	if err != nil {
		return llm.Tool{
			Type: "function",
			Function: llm.FunctionDef{
				Name:        mcpTool.Name,
				Description: mcpTool.Description,
				Parameters:  llm.ToolParams{Type: "object", Properties: props},
			},
		}
	}

	var schema struct {
		Properties map[string]map[string]any `json:"properties"`
		Required   []string                  `json:"required"`
	}
	if err := json.Unmarshal(schemaBytes, &schema); err == nil {
		for propName, propMap := range schema.Properties {
			arg := llm.ToolArg{}
			if t, ok := propMap["type"].(string); ok {
				arg.Type = t
			}
			if d, ok := propMap["description"].(string); ok {
				arg.Description = d
			}
			if e, ok := propMap["example"].(string); ok {
				arg.Example = e
			}
			if def, ok := propMap["default"].(string); ok {
				arg.Default = def
			}
			if items, ok := propMap["items"]; ok {
				arg.Items = items
			}
			props[propName] = arg
		}
		required = schema.Required
	}

	// The whole schema goes along, for what Parameters cannot say (enums,
	// nested objects); a tool without one has "null" here.
	var full json.RawMessage
	if len(schemaBytes) > 0 && schemaBytes[0] == '{' {
		full = schemaBytes
	}
	return llm.Tool{
		Type: "function",
		Function: llm.FunctionDef{
			Name:        mcpTool.Name,
			Description: mcpTool.Description,
			Parameters: llm.ToolParams{
				Type:       "object",
				Properties: props,
				Required:   required,
			},
			Schema: full,
		},
	}
}
