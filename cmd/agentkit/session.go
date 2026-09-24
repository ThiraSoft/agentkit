package main

// A session file keeps a conversation between two runs, so a task can be
// taken up again where it stopped: the history as JSON, system prompt first.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/ThiraSoft/agentkit/llm"
)

// loadSession reads the history kept in path. A file that does not exist is
// a session not begun, and gives no history.
func loadSession(path string) ([]llm.Message, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var msgs []llm.Message
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return nil, fmt.Errorf("session %s: %w", path, err)
	}
	if len(msgs) == 0 || msgs[0].Role != "system" {
		return nil, fmt.Errorf("session %s: the history does not open with a system prompt", path)
	}
	return msgs, nil
}

// saveSession writes the history to path, whole or not at all.
func saveSession(path string, msgs []llm.Message) error {
	raw, err := json.MarshalIndent(msgs, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".session-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
