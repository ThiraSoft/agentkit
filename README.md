# agentkit

[![Go Reference](https://pkg.go.dev/badge/github.com/ThiraSoft/agentkit.svg)](https://pkg.go.dev/github.com/ThiraSoft/agentkit)
[![test](https://github.com/ThiraSoft/agentkit/actions/workflows/test.yml/badge.svg)](https://github.com/ThiraSoft/agentkit/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An agent loop in Go, as a library: a model, Go tools and MCP servers. The
`Agent` holds the provider and the tools; each `Conversation` holds its own
history, and a `Send` runs the model and its tools until the final answer.

## What it does, and what it leaves to you

agentkit does:

- the loop: model call, tool calls (in parallel), results back to the
  model, until an answer without tool calls or `MaxSteps`;
- streaming, interruption (a new `Send` cuts the running one) and hooks on
  every step;
- providers for the main APIs and any OpenAI-compatible server, with
  retries on transient errors;
- MCP servers over stdio, SSE or streamable HTTP, whose tools sit next to
  your Go tools.

agentkit does not do long-term memory, persistence of conversations or
prompt templates. It holds no global state and
writes no file: keep `Conversation.Messages()` wherever you like and pass
it back to `NewConversation`.

## Install

```sh
go get github.com/ThiraSoft/agentkit
```

Requires Go 1.25 or later.

## Example

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/ThiraSoft/agentkit"
	"github.com/ThiraSoft/agentkit/llm"
)

func main() {
	ctx := context.Background()

	agent, err := agentkit.New(ctx, agentkit.Config{
		Provider: "gemini",
		Model:    "gemini-2.5-flash-lite", // key: GEMINI_API_KEY
		Tools: []agentkit.Tool{{
			Name:        "clock",
			Description: "Tells the time.",
			Parameters:  llm.ToolParams{Type: "object", Properties: llm.ToolProperties{}},
			Run: func(ctx context.Context, args json.RawMessage) (string, error) {
				return time.Now().Format("15:04"), nil
			},
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	conv := agent.NewConversation("Answer in one sentence.")
	turn, err := conv.Send(ctx, "What time is it?", agentkit.Hooks{
		OnText: func(chunk string) { fmt.Print(chunk) },
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\n(%d step(s))\n", turn.Steps)
}
```

## Providers

Set `Config.Provider` and `Config.Model`. `APIKey` and `BaseURL` in the
config take precedence over the environment variable and the default URL.

| Provider | Key from | Default URL |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `gemini` | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| `mistral` | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| `ollama` | none | `http://localhost:11434` |
| `llamacpp` | none | `LLAMACPP_URL` (without `/v1`), or `http://localhost:8080`; if using `BaseURL`, include `/v1` (e.g. `http://localhost:8080/v1`) |
| `openai-compat` | `Config.APIKey` | none: `BaseURL` is required, e.g. `http://localhost:8000/v1` |

Your own provider: implement `llm.Provider` and pass it as
`Config.ProviderImpl`. For a server that speaks the OpenAI chat/completions
format but needs its own settings (headers, extra body fields, timeout),
start from `llm.NewOpenAICompat` and wrap it with `llm.WithRetry`.

`Config.Temperature` and `Config.MaxTokens` go to every provider in its own
terms (`max_completion_tokens` for OpenAI, `maxOutputTokens` for Gemini,
`num_predict` for Ollama). Left unset, the provider's default holds;
Anthropic, which requires a cap, gets 32000.

`Config.PromptCache` marks the prompt for caching on Anthropic, which has to
be told: the tools and the system prompt, and the conversation as it grows.
OpenAI and Gemini cache on their own. Either way, `Turn.Usage` says how many
prompt tokens came from the cache.

## Hooks

`Hooks` are all optional and run in the goroutine of `Send`:

- `OnText(chunk)`: streamed text;
- `OnToolCall(call)` and `OnToolResult(result)`: each tool call and its
  result, the results as the tools finish (the history keeps them in the
  order of the calls);
- `Approve(call) bool`: return false to refuse a call; the model is told;
- `OnStepEnd()`: the model finished a message that asks for tools;
- `OnFinish()`: the answer is complete; it may block (for instance while a
  voice finishes speaking);
- `Prepare(msgs) msgs`: rewrite what is sent to the model at each step,
  without touching the history;
- `OnInterrupt(written) kept`: choose what stays in the history when a turn
  is cut.

## Context

agentkit sends the whole history at every step. `KeepTurns(n)` and
`KeepTokens(budget)` are ready-made `Prepare` hooks that send only the last
turns, or as many as fit in a token budget (a rough four bytes per token),
always with the system prompt and the turn under way. They cut at a user
message, so a tool call never loses its result, and the history itself
stays whole.

## Tools

A `Tool` describes its arguments with `Parameters`, or with `Schema`, a raw
JSON Schema that can say more (enums, nested objects, bounds). `NewTool`
writes the schema from a Go type and decodes the arguments into it:

```go
type forecastArgs struct {
	City string `json:"city" jsonschema:"the city to forecast"`
	Days int    `json:"days,omitempty" jsonschema:"how many days, 1 by default"`
}

forecast, err := agentkit.NewTool("forecast", "Weather forecast for a city.",
	func(ctx context.Context, args forecastArgs) (string, error) {
		return lookup(ctx, args.City, args.Days)
	})
```

A panic in a tool is recovered: the model is told the tool failed.

## Usage

`Turn.Usage` sums the tokens of the model calls of a `Send`: `InputTokens`
(the whole prompt, cache included), `OutputTokens`, `CacheReadTokens` and
`CacheWriteTokens`, as far as the provider reports them. Each message a
provider returns carries its own in `llm.Message.Usage`.

## Structured output

`Config.ResponseSchema` makes the model answer with JSON that follows a
schema, which `SchemaFor` writes from a Go type:

```go
type verdict struct {
	Spam   bool   `json:"spam"`
	Reason string `json:"reason"`
}

schema, err := agentkit.SchemaFor[verdict]()
agent, err := agentkit.New(ctx, agentkit.Config{Provider: "openai", Model: "gpt-5-mini", ResponseSchema: schema})
turn, err := agent.NewConversation("Classify the message.").Send(ctx, text, agentkit.Hooks{})
var v verdict
err = json.Unmarshal([]byte(turn.Text), &v)
```

Not every model takes a response schema together with tools.

## Configuration file

`LoadConfig` reads a `Config` from JSON, everything but the Go tools:

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-5",
  "apiKey": "${ANTHROPIC_API_KEY}",
  "maxTokens": 8192,
  "promptCache": true,
  "mcp": [{"name": "files", "transport": "stdio", "command": "files-mcp --root /tmp"}]
}
```

The other fields are `baseURL`, `maxSteps`, `maxToolResult`, `temperature`
and `responseSchema`. `${VAR}` in `apiKey`, `baseURL` and the MCP servers
is replaced by the environment variable; an unknown field is an error.

## MCP

```go
agent, err := agentkit.New(ctx, agentkit.Config{
	Provider: "openai-compat",
	BaseURL:  "http://localhost:8080/v1",
	Model:    "local",
	MCP: []mcp.ServerConfig{
		{Name: "files", Transport: "stdio", Command: "/usr/local/bin/files-mcp --root /tmp"},
		{Name: "search", Transport: "streamable", URL: "https://mcp.example.com/mcp",
			Headers: map[string]string{"Authorization": "Bearer ${SEARCH_TOKEN}"}},
	},
})
```

`New` fails if a server does not answer, or if two tools share a name. The
`mcp` package can also be used alone: `mcp.Dial` for one server,
`mcp.NewManager` for a list.

## Command line

`cmd/agentkit` is a small terminal client, handy to try a model or an MCP
server:

```sh
go install github.com/ThiraSoft/agentkit/cmd/agentkit@latest
agentkit -config agent.json
agentkit -provider gemini -model gemini-2.5-flash
```

The answer streams on stdout, the tool calls on stderr. `/reset`, `/usage`,
`/tools` and `/quit` do what they say; Ctrl-C cuts the answer under way.
Each release on GitHub carries the binaries.

## Stability

agentkit is v0: the API may change before v1. Changes are listed in the
release notes.

## Tests

```sh
go test -race ./...
```

Integration tests talk to real models and are behind a build tag:

```sh
GEMINI_API_KEY=... go test -tags integration . -run TestGemini
AGENTKIT_OPENAI_URL=http://localhost:8080/v1 AGENTKIT_OPENAI_MODEL=local \
  go test -tags integration . -run TestOpenAICompat
```

## License

MIT, see [LICENSE](LICENSE).
