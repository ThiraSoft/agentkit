package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	mcpSdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestManagerConnectsListsAndCalls(t *testing.T) {
	ctx := context.Background()
	m := NewManager([]ServerConfig{{Name: "system", Transport: "stdio", Command: testServerBinary(t)}})
	defer m.Close()
	if st := m.Status(); len(st) != 1 || st[0].Status != "disconnected" {
		t.Fatalf("status before Connect %+v", st)
	}
	if err := m.Connect(ctx); err != nil {
		t.Fatal(err)
	}
	defs := m.GetToolDefinitions()
	if len(defs) != 1 || defs[0].Function.Name != "get_time" {
		t.Fatalf("tools %+v", defs)
	}
	out, err := m.CallTool(ctx, "get_time", json.RawMessage(`{"timezone":"UTC"}`))
	if err != nil || !strings.Contains(out, "UTC") {
		t.Fatalf("CallTool %q, %v", out, err)
	}
	st := m.Status()
	if st[0].Status != "connected" || strings.Join(st[0].Tools, ",") != "get_time" || st[0].Error != "" {
		t.Fatalf("status after Connect %+v", st)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if defs := m.GetToolDefinitions(); len(defs) != 0 {
		t.Fatalf("tools after Close %+v", defs)
	}
	if st := m.Status(); st[0].Status != "disconnected" {
		t.Fatalf("status after Close %+v", st)
	}
}

func TestManagerSkipsDisabledServers(t *testing.T) {
	off, on := false, true
	m := NewManager([]ServerConfig{
		{Name: "off", Transport: "stdio", Command: "/does/not/exist", Enabled: &off},
		{Name: "on", Transport: "stdio", Command: testServerBinary(t), Enabled: &on},
		{Name: "default", Transport: "stdio", Command: testServerBinary(t)},
	})
	defer m.Close()
	// "off" would fail to start if it were dialed: no error means it was skipped.
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect dialed a disabled server: %v", err)
	}
	st := m.Status()
	if st[0].Status != "disabled" || st[1].Status != "connected" || st[2].Status != "connected" {
		t.Fatalf("status %+v, want disabled, connected, connected", st)
	}
	if (ServerConfig{}).IsEnabled() != true || (ServerConfig{Enabled: &off}).IsEnabled() {
		t.Fatal("IsEnabled: nil must mean enabled, false disabled")
	}
	var c ServerConfig
	if err := json.Unmarshal([]byte(`{"name":"x","transport":"stdio","enabled":false}`), &c); err != nil || c.IsEnabled() {
		t.Fatalf("enabled:false from JSON gives %+v, %v", c, err)
	}
}

func TestManagerReportsFailures(t *testing.T) {
	m := NewManager([]ServerConfig{
		{Name: "absent", Transport: "stdio", Command: "/does/not/exist"},
		{Name: "pigeon", Transport: "pigeon"},
	})
	defer m.Close()
	err := m.Connect(context.Background())
	if err == nil {
		t.Fatal("Connect succeeded with two broken servers")
	}
	for _, name := range []string{"absent", "pigeon"} {
		if !strings.Contains(err.Error(), "MCP server "+name+": ") {
			t.Fatalf("error %q does not name %s", err, name)
		}
	}
	st := m.Status()
	if st[0].Status != "error" || st[0].Error == "" {
		t.Fatalf("absent %+v", st[0])
	}
	if st[1].Status != "error" || st[1].Error != "unknown transport: pigeon" {
		t.Fatalf("pigeon %+v", st[1])
	}
}

func TestManagerKeepsItsOwnConfigs(t *testing.T) {
	configs := []ServerConfig{{Name: "a", Transport: "pigeon"}}
	m := NewManager(configs)
	configs[0].Name = "changed"
	if got := m.Status()[0].Name; got != "a" {
		t.Fatalf("the Manager follows the caller's slice: %q", got)
	}
}

func TestManagerCallToolErrors(t *testing.T) {
	ctx := context.Background()
	m := NewManager(nil)
	if _, err := m.CallTool(ctx, "nope", nil); err == nil {
		t.Fatal("expected an error for an unknown tool")
	}
	m.mu.Lock()
	m.toolIndex["ghost"] = "gone"
	m.mu.Unlock()
	if _, err := m.CallTool(ctx, "ghost", nil); err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Fatalf("err %v, want a server not connected", err)
	}
}

// memorySession connects an in-memory client session to an in-memory
// server with no tools.
func memorySession(t *testing.T, transport func(mcpSdk.Transport) mcpSdk.Transport) *mcpSdk.ClientSession {
	t.Helper()
	ctx := context.Background()
	server := mcpSdk.NewServer(&mcpSdk.Implementation{Name: "srv", Version: "1.0"}, nil)
	client := mcpSdk.NewClient(&mcpSdk.Implementation{Name: "cli", Version: "1.0"}, nil)
	tServer, tClient := mcpSdk.NewInMemoryTransports()
	if _, err := server.Connect(ctx, tServer, nil); err != nil {
		t.Fatal(err)
	}
	session, err := client.Connect(ctx, transport(tClient), nil)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

func TestManagerFetchToolsReportsFailures(t *testing.T) {
	session := memorySession(t, func(tr mcpSdk.Transport) mcpSdk.Transport { return tr })
	m := NewManager([]ServerConfig{{Name: "srv", Transport: "stdio"}})
	m.mu.Lock()
	m.clients["srv"] = &Client{session: session}
	m.mu.Unlock()
	session.Close()
	if err := m.FetchTools(context.Background()); err == nil || !strings.Contains(err.Error(), "MCP server srv: ") {
		t.Fatalf("FetchTools error %v", err)
	}
	st := m.Status()
	if len(st) != 1 || st[0].Status != "connected" || st[0].Error == "" {
		t.Fatalf("status after failed FetchTools %+v, want Status connected with non-empty Error", st)
	}
}

type trackingTransport struct {
	underlying mcpSdk.Transport
	closed     atomic.Bool
}

func (t *trackingTransport) Connect(ctx context.Context) (mcpSdk.Connection, error) {
	conn, err := t.underlying.Connect(ctx)
	if err != nil {
		return nil, err
	}
	return &trackingConn{Connection: conn, closed: &t.closed}, nil
}

type trackingConn struct {
	mcpSdk.Connection
	closed *atomic.Bool
}

func (c *trackingConn) Close() error {
	c.closed.Store(true)
	return c.Connection.Close()
}

func TestConnectClosesExistingClient(t *testing.T) {
	var tracked *trackingTransport
	session := memorySession(t, func(tr mcpSdk.Transport) mcpSdk.Transport {
		tracked = &trackingTransport{underlying: tr}
		return tracked
	})
	old := &Client{session: session}
	m := NewManager([]ServerConfig{{Name: "srv", Transport: "stdio", Command: "/does/not/exist"}})
	m.mu.Lock()
	m.clients["srv"] = old
	m.mu.Unlock()

	// Connect tries "srv" again (and fails on the missing command): the
	// previous client must be closed and dropped first.
	_ = m.Connect(context.Background())

	if !tracked.closed.Load() {
		t.Fatal("expected existing client to be closed before reconnecting")
	}
	m.mu.RLock()
	c := m.clients["srv"]
	m.mu.RUnlock()
	if c == old {
		t.Fatal("expected existing client to be removed from m.clients")
	}
}

// TestGetToolDefinitionsIsDeterministic ensures tool order follows the
// manager's config order (and, within a server, the order the server
// returned its tools), not Go's randomized map iteration order.
func TestGetToolDefinitionsIsDeterministic(t *testing.T) {
	configs := []ServerConfig{
		{Name: "alpha", Transport: "stdio"},
		{Name: "bravo", Transport: "stdio"},
		{Name: "charlie", Transport: "stdio"},
		{Name: "delta", Transport: "stdio"},
		{Name: "echo", Transport: "stdio"},
	}
	m := NewManager(configs)

	makeClient := func(names ...string) *Client {
		tools := make([]*mcpSdk.Tool, len(names))
		for i, n := range names {
			tools[i] = &mcpSdk.Tool{Name: n}
		}
		return &Client{tools: tools}
	}

	m.mu.Lock()
	m.clients["alpha"] = makeClient("a1", "a2")
	m.clients["bravo"] = makeClient("b1")
	m.clients["charlie"] = makeClient("c1", "c2", "c3")
	m.clients["delta"] = makeClient("d1")
	m.clients["echo"] = makeClient("e1", "e2")
	m.mu.Unlock()

	want := "a1,a2,b1,c1,c2,c3,d1,e1,e2"
	for i := 0; i < 50; i++ {
		defs := m.GetToolDefinitions()
		got := make([]string, len(defs))
		for j, d := range defs {
			got[j] = d.Function.Name
		}
		if strings.Join(got, ",") != want {
			t.Fatalf("run %d: got %v, want %s", i, got, want)
		}
	}
}

func TestClientFetchToolsIsRaceFree(t *testing.T) {
	session := memorySession(t, func(tr mcpSdk.Transport) mcpSdk.Transport { return tr })
	defer session.Close()

	client := &Client{session: session}
	ctx := context.Background()

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			if err := client.FetchTools(ctx); err != nil {
				t.Errorf("FetchTools: %v", err)
				return
			}
		}
	}()

	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_ = client.GetToolDefinitions()
		}
	}()

	wg.Wait()
}

func TestManagerDuplicateServerName(t *testing.T) {
	bin := testServerBinary(t)
	m := NewManager([]ServerConfig{
		{Name: "srv", Transport: "stdio", Command: bin},
		{Name: "srv", Transport: "stdio", Command: bin},
	})
	defer m.Close()

	err := m.Connect(context.Background())
	if err == nil || !strings.Contains(err.Error(), "MCP server srv: duplicate name") {
		t.Fatalf("Connect err %v, want duplicate name error", err)
	}

	st := m.Status()
	if len(st) != 2 || st[0].Status != "connected" {
		t.Fatalf("status %+v, want first server connected", st)
	}
	defs := m.GetToolDefinitions()
	if len(defs) != 1 || defs[0].Function.Name != "get_time" {
		t.Fatalf("tools %+v, want exactly one tool from the first server", defs)
	}
}
