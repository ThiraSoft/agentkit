package mcp_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/ThiraSoft/agentkit/mcp"
)

func ExampleManager() {
	ctx := context.Background()
	m := mcp.NewManager([]mcp.ServerConfig{
		{Name: "files", Transport: "stdio", Command: "/usr/local/bin/files-mcp --root /tmp"},
		{Name: "search", Transport: "streamable", URL: "https://mcp.example.com/mcp", Headers: map[string]string{"Authorization": "Bearer ${SEARCH_TOKEN}"}},
	})
	defer m.Close()
	if err := m.Connect(ctx); err != nil {
		log.Print(err) // servers that answered stay connected
	}
	for _, s := range m.Status() {
		fmt.Println(s.Name, s.Status, s.Tools)
	}
	out, err := m.CallTool(ctx, "list_files", json.RawMessage(`{"path":"/tmp"}`))
	fmt.Println(out, err)
}
