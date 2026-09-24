// Command agentkit talks with a model and its MCP tools in the terminal.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"

	"github.com/ThiraSoft/agentkit"
	"github.com/ThiraSoft/agentkit/llm"
)

// version is set at release time by goreleaser.
var version = "dev"

const usageText = `usage: agentkit [flags]

Talk with a model and its tools, one line at a time. ` + help + `.
Ctrl-C cuts the answer under way, and leaves at the prompt.

With -p, send one message, print the answer and leave: stdout gets the last
message alone, stderr the errors, and -v shows there what came before. The
exit code is 1 when the turn failed or was cut. With -session, the conversation is read
from the file if it exists and written back after each turn.

`

func main() {
	config := flag.String("config", "", "a JSON agent file, as agentkit.LoadConfig reads it")
	provider := flag.String("provider", "", "without -config: openai, gemini, anthropic, mistral, ollama, llamacpp or openai-compat")
	model := flag.String("model", "", "without -config: the model")
	baseURL := flag.String("base-url", "", "without -config: the provider's base URL")
	system := flag.String("system", "", "the system prompt, else the config's, else a helpful assistant's")
	prompt := flag.String("p", "", "send this message, print the answer and leave; - reads it from stdin")
	session := flag.String("session", "", "a JSON file the conversation is read from and written to")
	verbose := flag.Bool("v", false, "with -p, show on stderr what the model writes on the way and its tool calls")
	workdir := flag.String("workdir", "", "the directory the built-in tools work in, instead of the config's")
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
	if *workdir != "" {
		cfg.Workdir = *workdir
	}
	var history []llm.Message
	if *session != "" {
		if history, err = loadSession(*session); err != nil {
			fmt.Fprintln(os.Stderr, "agentkit:", err)
			os.Exit(2)
		}
	}
	message := *prompt
	if message == "-" {
		raw, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintln(os.Stderr, "agentkit:", err)
			os.Exit(2)
		}
		message = string(raw)
	}
	if *prompt != "" && strings.TrimSpace(message) == "" {
		fmt.Fprintln(os.Stderr, "agentkit: -p gives an empty message")
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

	// A session taken up keeps the system prompt it began with.
	var conv *agentkit.Conversation
	if len(history) > 0 {
		conv = agent.NewConversation(history[0].Content, history[1:]...)
	} else {
		conv = agent.NewConversation(systemPrompt(*system, agent.System()))
	}
	r := &repl{
		agent:  agent,
		conv:   conv,
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
	if *session != "" {
		r.save = func() error { return saveSession(*session, conv.Messages()) }
	}
	if *prompt != "" {
		r.final, r.verbose = true, *verbose
		ok := r.send(message)
		agent.Close()
		if !ok {
			os.Exit(1)
		}
		return
	}
	err = r.run(os.Stdin)
	agent.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// systemPrompt is the flag's prompt, else the config's, else a default.
func systemPrompt(flag, config string) string {
	switch {
	case flag != "":
		return flag
	case config != "":
		return config
	}
	return "You are a helpful assistant."
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
