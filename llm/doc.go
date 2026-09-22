// Package llm talks to language models behind one interface, Provider
// (Chat and Stream), with the message and tool types they share.
//
// NewProvider builds the generic providers: "openai", "gemini",
// "anthropic", "mistral", "ollama", "llamacpp" and "openai-compat" (any
// OpenAI-compatible server, BaseURL required), each wrapped with WithRetry.
// A provider written outside this package implements Provider; one that
// speaks the OpenAI chat/completions format can start from NewOpenAICompat.
package llm
