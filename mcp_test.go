package agentkit

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/ThiraSoft/agentkit/llm"
	"github.com/ThiraSoft/agentkit/mcp"
)

var (
	buildOnce sync.Once
	buildDir  string
	buildErr  error
)

// testServer builds the test MCP server once for all tests.
func testServer(t *testing.T) string {
	t.Helper()
	buildOnce.Do(func() {
		dir, err := os.MkdirTemp("", "agentkit-mcp")
		if err != nil {
			buildErr = err
			return
		}
		buildDir = dir
		out, err := exec.Command("go", "build", "-o", filepath.Join(dir, "agentkit-testmcp"), "github.com/ThiraSoft/agentkit/internal/testmcp").CombinedOutput()
		if err != nil {
			buildErr = fmt.Errorf("%v: %s", err, out)
		}
	})
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	return filepath.Join(buildDir, "agentkit-testmcp")
}

func TestMCPToolsAreCalled(t *testing.T) {
	f := &fakeProvider{replies: []fakeReply{
		{calls: []llm.ToolCall{call("1", "get_time", `{"timezone":"Europe/Paris"}`)}},
		{chunks: []string{"It is time."}},
	}}
	a, err := New(context.Background(), Config{
		ProviderImpl: f,
		Tools:        []Tool{echoTool("local", ok)},
		MCP:          []mcp.ServerConfig{{Name: "system", Transport: "stdio", Command: testServer(t)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if tools := a.Tools(); tools[0] != "local" || !slices.Contains(tools, "get_time") {
		t.Fatalf("tools %v", tools)
	}
	var result ToolResult
	_, err = a.NewConversation("s").Send(context.Background(), "time ?", Hooks{OnToolResult: func(r ToolResult) { result = r }})
	if err != nil {
		t.Fatal(err)
	}
	if result.Err != "" || !strings.Contains(result.Content, "Europe/Paris") {
		t.Fatalf("result %+v", result)
	}
}

func TestMCPDuplicateToolWithGoToolFails(t *testing.T) {
	f := &fakeProvider{}
	a, err := New(context.Background(), Config{
		ProviderImpl: f,
		Tools:        []Tool{echoTool("get_time", ok)},
		MCP:          []mcp.ServerConfig{{Name: "system", Transport: "stdio", Command: testServer(t)}},
	})
	if err == nil {
		a.Close()
		t.Fatal("New succeeded even though a Go tool and an MCP tool share the name 'get_time'")
	}
	if !strings.Contains(err.Error(), "get_time") || !strings.Contains(err.Error(), "twice") {
		t.Fatalf("unexpected error: %v", err)
	}
}
