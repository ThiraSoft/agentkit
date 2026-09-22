// Package mcp connects to Model Context Protocol servers over stdio, SSE or
// streamable HTTP, and exposes their tools in the llm.Tool format so an
// agent can call them next to its own tools.
//
// Client is one connection, built with Dial. Manager connects a fixed list
// of servers, lists their tools in the order of that list, routes calls,
// reports each server's state and closes them. Neither reads nor writes
// any file.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/ThiraSoft/agentkit/llm"
)

// ToolExecutor is a source of tools that can be listed and called. Client
// and Manager implement it.
type ToolExecutor interface {
	FetchTools(ctx context.Context) error
	GetToolDefinitions() []llm.Tool
	CallTool(ctx context.Context, name string, args json.RawMessage) (string, error)
	Close() error
}

// ServerConfig describes one MCP server. Command, URL and header values
// may hold ${VAR} placeholders, replaced with environment variables when
// connecting.
type ServerConfig struct {
	Name      string `json:"name"`
	Transport string `json:"transport"` // "stdio", "sse" or "streamable"
	// Command is the executable and its arguments for the stdio transport,
	// split on spaces. There is no quoting: an argument containing a
	// space cannot be expressed here.
	Command string            `json:"command,omitempty"`
	URL     string            `json:"url,omitempty"` // for "sse" and "streamable"
	Headers map[string]string `json:"headers,omitempty"`
	// Enabled set to false keeps the server out of Manager.Connect. Nil
	// (the field left out) means enabled, so a config that never mentions
	// it connects.
	Enabled *bool `json:"enabled,omitempty"`
}

// IsEnabled reports whether the server should be connected: true unless
// Enabled is set to false.
func (c ServerConfig) IsEnabled() bool {
	return c.Enabled == nil || *c.Enabled
}

// ServerStatus is the state of one server of a Manager. It embeds the
// ServerConfig as given, Headers included (placeholders unexpanded), so a
// caller serializing Status should not put tokens in clear in Headers.
type ServerStatus struct {
	ServerConfig
	Status string `json:"status"` // "connected", "disconnected", "disabled" or "error"
	// Error is the last error; for a connected server, the error of the last FetchTools.
	Error string   `json:"error,omitempty"`
	Tools []string `json:"tools"`
}

// Manager connects the servers it was built with and routes each tool call
// to the server that declared the tool. It is safe for concurrent use.
type Manager struct {
	configs []ServerConfig // set by NewManager, never modified

	mu        sync.RWMutex
	clients   map[string]*Client // server name -> connected client
	errors    map[string]string  // server name -> last error
	toolIndex map[string]string  // tool name -> server name
}

// NewManager returns a Manager for configs, not connected yet. It keeps
// its own copy of the slice.
func NewManager(configs []ServerConfig) *Manager {
	return &Manager{
		configs:   append([]ServerConfig(nil), configs...),
		clients:   map[string]*Client{},
		errors:    map[string]string{},
		toolIndex: map[string]string{},
	}
}

// Connect connects every enabled server, after closing its client from a
// previous Connect if any, and returns the joined errors of those that
// failed. Servers that answered stay connected. A disabled server is not
// dialed; Status reports it as "disabled". If multiple configs share the
// same name, the first is dialed and subsequent ones return an error
// without being dialed.
func (m *Manager) Connect(ctx context.Context) error {
	var errs []error
	seen := make(map[string]bool)
	for _, cfg := range m.configs {
		if seen[cfg.Name] {
			errs = append(errs, fmt.Errorf("MCP server %s: duplicate name", cfg.Name))
			continue
		}
		seen[cfg.Name] = true

		if !cfg.IsEnabled() {
			continue
		}
		m.mu.Lock()
		if old, ok := m.clients[cfg.Name]; ok {
			old.Close()
			delete(m.clients, cfg.Name)
		}
		m.mu.Unlock()

		client, err := Dial(ctx, cfg)
		m.mu.Lock()
		if err != nil {
			m.errors[cfg.Name] = err.Error()
			errs = append(errs, fmt.Errorf("MCP server %s: %w", cfg.Name, err))
		} else {
			delete(m.errors, cfg.Name)
			m.clients[cfg.Name] = client
		}
		m.mu.Unlock()
	}
	m.rebuildToolIndex()
	return errors.Join(errs...)
}

// FetchTools asks every connected server for its tools again and returns
// the joined errors of those that failed. A failing server stays
// connected; its error shows in Status.
func (m *Manager) FetchTools(ctx context.Context) error {
	m.mu.RLock()
	clients := make(map[string]*Client, len(m.clients))
	for name, c := range m.clients {
		clients[name] = c
	}
	m.mu.RUnlock()

	var errs []error
	for name, c := range clients {
		if err := c.FetchTools(ctx); err != nil {
			m.mu.Lock()
			m.errors[name] = err.Error()
			m.mu.Unlock()
			errs = append(errs, fmt.Errorf("MCP server %s: %w", name, err))
		} else {
			m.mu.Lock()
			delete(m.errors, name)
			m.mu.Unlock()
		}
	}
	m.rebuildToolIndex()
	return errors.Join(errs...)
}

// GetToolDefinitions returns the tools of the connected servers, in the
// order of the configs and, within a server, in the order it returned them.
func (m *Manager) GetToolDefinitions() []llm.Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var defs []llm.Tool
	seen := make(map[string]bool)
	for _, cfg := range m.configs {
		if seen[cfg.Name] {
			continue
		}
		seen[cfg.Name] = true
		if c, ok := m.clients[cfg.Name]; ok {
			defs = append(defs, c.GetToolDefinitions()...)
		}
	}
	return defs
}

// CallTool calls the tool name on the server that declared it. When two
// servers declare the same name, the first one in the configs wins.
func (m *Manager) CallTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
	m.mu.RLock()
	server, ok := m.toolIndex[name]
	if !ok {
		m.mu.RUnlock()
		return "", fmt.Errorf("tool '%s' not found in any connected MCP server", name)
	}
	c, ok := m.clients[server]
	m.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("server '%s' not connected", server)
	}
	return c.CallTool(ctx, name, args)
}

// Status returns the state of each server, in the order of the configs.
// Error holds the last error; for a connected server, the error of the last FetchTools.
func (m *Manager) Status() []ServerStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]ServerStatus, 0, len(m.configs))
	for _, cfg := range m.configs {
		s := ServerStatus{ServerConfig: cfg, Status: "disconnected"}
		if !cfg.IsEnabled() {
			s.Status = "disabled"
		} else if c, ok := m.clients[cfg.Name]; ok {
			s.Status = "connected"
			s.Error = m.errors[cfg.Name]
			for _, t := range c.GetToolDefinitions() {
				s.Tools = append(s.Tools, t.Function.Name)
			}
		} else if msg, ok := m.errors[cfg.Name]; ok {
			s.Status = "error"
			s.Error = msg
		}
		out = append(out, s)
	}
	return out
}

// Close disconnects every server. Errors from closing sessions are
// ignored. Connect may be called again afterwards.
func (m *Manager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for name, c := range m.clients {
		c.Close()
		delete(m.clients, name)
	}
	m.toolIndex = map[string]string{}
	return nil
}

func (m *Manager) rebuildToolIndex() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.toolIndex = map[string]string{}
	for _, cfg := range m.configs {
		c, ok := m.clients[cfg.Name]
		if !ok {
			continue
		}
		for _, t := range c.GetToolDefinitions() {
			if _, taken := m.toolIndex[t.Function.Name]; !taken {
				m.toolIndex[t.Function.Name] = cfg.Name
			}
		}
	}
}
