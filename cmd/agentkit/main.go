// Command agentkit talks with a model and its MCP tools in the terminal.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"sync"

	"github.com/ThiraSoft/agentkit"
)

// version is set at release time by goreleaser.
var version = "dev"

const usageText = `usage: agentkit [flags]

Talk with a model and its MCP tools, one line at a time. ` + help + `.
Ctrl-C cuts the answer under way, and leaves at the prompt.

`

func main() {
	config := flag.String("config", "", "a JSON agent file, as agentkit.LoadConfig reads it")
	provider := flag.String("provider", "", "without -config: openai, gemini, anthropic, mistral, ollama, llamacpp or openai-compat")
	model := flag.String("model", "", "without -config: the model")
	baseURL := flag.String("base-url", "", "without -config: the provider's base URL")
	system := flag.String("system", "You are a helpful assistant.", "the system prompt")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usageText)
		flag.PrintDefaults()
	}
	flag.Parse()
	if *showVersion {
		fmt.Println(version)
		return
	}
	cfg, err := loadConfig(*config, *provider, *model, *baseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "agentkit:", err)
		os.Exit(2)
	}
	ctx := context.Background()
	agent, err := agentkit.New(ctx, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Ctrl-C cuts the turn under way, or leaves when there is none.
	var mu sync.Mutex
	var cancelTurn context.CancelFunc
	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	go func() {
		for range interrupts {
			mu.Lock()
			cancel := cancelTurn
			mu.Unlock()
			if cancel == nil {
				fmt.Println()
				agent.Close()
				os.Exit(0)
			}
			cancel()
		}
	}()

	r := &repl{
		agent:  agent,
		conv:   agent.NewConversation(*system),
		out:    os.Stdout,
		errOut: os.Stderr,
		turnCtx: func() (context.Context, context.CancelFunc) {
			turnCtx, cancel := context.WithCancel(ctx)
			mu.Lock()
			cancelTurn = cancel
			mu.Unlock()
			return turnCtx, func() {
				mu.Lock()
				cancelTurn = nil
				mu.Unlock()
				cancel()
			}
		},
	}
	err = r.run(os.Stdin)
	agent.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// loadConfig reads the -config file, or builds a Config from the flags.
func loadConfig(path, provider, model, baseURL string) (agentkit.Config, error) {
	if path != "" {
		if provider != "" || model != "" || baseURL != "" {
			return agentkit.Config{}, errors.New("-config excludes -provider, -model and -base-url")
		}
		return agentkit.LoadConfig(path)
	}
	if provider == "" {
		return agentkit.Config{}, errors.New("give -config, or -provider and -model")
	}
	return agentkit.Config{Provider: provider, Model: model, BaseURL: baseURL}, nil
}
