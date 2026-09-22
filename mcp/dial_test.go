package mcp

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
)

var (
	testBinOnce sync.Once
	testBin     string
	testBinErr  error
)

// testServerBinary builds the test MCP server (internal/testmcp) once per
// test binary and returns its path.
func testServerBinary(t *testing.T) string {
	t.Helper()
	testBinOnce.Do(func() {
		dir, err := os.MkdirTemp("", "agentkit-mcp-test")
		if err != nil {
			testBinErr = err
			return
		}
		testBin = filepath.Join(dir, "testmcp")
		out, err := exec.Command("go", "build", "-o", testBin, "github.com/ThiraSoft/agentkit/internal/testmcp").CombinedOutput()
		if err != nil {
			testBinErr = fmt.Errorf("%v: %s", err, out)
		}
	})
	if testBinErr != nil {
		t.Fatal(testBinErr)
	}
	return testBin
}

func TestServerConfigExpanded(t *testing.T) {
	t.Setenv("AK_TEST_HOST", "example.test")
	cfg := ServerConfig{
		Name:    "x",
		Command: "${AK_TEST_HOST}/bin -v",
		URL:     "http://${AK_TEST_HOST}/mcp",
		Headers: map[string]string{"Authorization": "Bearer ${AK_TEST_HOST}"},
	}
	got := cfg.expanded()
	if got.Command != "example.test/bin -v" || got.URL != "http://example.test/mcp" || got.Headers["Authorization"] != "Bearer example.test" {
		t.Fatalf("expanded %+v", got)
	}
	if cfg.Headers["Authorization"] != "Bearer ${AK_TEST_HOST}" {
		t.Fatal("expanded modified the original headers")
	}
}

func TestDialUnknownTransport(t *testing.T) {
	_, err := Dial(context.Background(), ServerConfig{Name: "p", Transport: "pigeon"})
	if err == nil || err.Error() != "unknown transport: pigeon" {
		t.Fatalf("err %v", err)
	}
}

func TestDialMissingCommand(t *testing.T) {
	if _, err := Dial(context.Background(), ServerConfig{Name: "a", Transport: "stdio", Command: "/does/not/exist"}); err == nil {
		t.Fatal("Dial succeeded on a missing command")
	}
}

func TestDialFetchesTools(t *testing.T) {
	c, err := Dial(context.Background(), ServerConfig{Name: "t", Transport: "stdio", Command: testServerBinary(t)})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	defs := c.GetToolDefinitions()
	if len(defs) != 1 || defs[0].Function.Name != "get_time" {
		t.Fatalf("tools %+v", defs)
	}
}
