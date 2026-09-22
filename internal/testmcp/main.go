// Command testmcp is a minimal stdio MCP server for agentkit tests,
// with a single tool, get_time.
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type input struct {
	Timezone string `json:"timezone,omitempty" jsonschema:"IANA timezone, e.g. Europe/Paris; UTC if empty"`
}

type output struct {
	Time     string `json:"time" jsonschema:"time in RFC3339 format"`
	Timezone string `json:"timezone" jsonschema:"timezone used"`
}

func getTime(_ context.Context, _ *mcp.CallToolRequest, in input) (*mcp.CallToolResult, output, error) {
	tz := in.Timezone
	if tz == "" {
		tz = "UTC"
	}
	loc, err := time.LoadLocation(tz)
	if err != nil {
		return nil, output{}, fmt.Errorf("invalid timezone %q: %w", tz, err)
	}
	return nil, output{Time: time.Now().In(loc).Format(time.RFC3339), Timezone: tz}, nil
}

func main() {
	s := mcp.NewServer(&mcp.Implementation{Name: "agentkit-testmcp", Version: "v0.0.0"}, nil)
	mcp.AddTool(s, &mcp.Tool{Name: "get_time", Description: "Returns the current time in a given timezone or UTC."}, getTime)
	if err := s.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}
