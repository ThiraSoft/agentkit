package agentkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// SchemaFor returns the JSON Schema of T, a struct: its exported fields
// under their json names, required unless tagged omitempty or omitzero,
// described by the text of a `jsonschema:"..."` tag.
func SchemaFor[T any]() (json.RawMessage, error) {
	s, err := schemaFor[T]()
	if err != nil {
		return nil, fmt.Errorf("agentkit: %w", err)
	}
	return s, nil
}

func schemaFor[T any]() (json.RawMessage, error) {
	s, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, err
	}
	return json.Marshal(s)
}

// NewTool builds a Tool whose arguments are a T: its Schema comes from
// SchemaFor, and run receives the arguments decoded. Arguments that do not
// decode into a T are an error the model is told about.
func NewTool[T any](name, description string, run func(ctx context.Context, args T) (string, error)) (Tool, error) {
	schema, err := schemaFor[T]()
	if err != nil {
		return Tool{}, fmt.Errorf("agentkit: tool %q: %w", name, err)
	}
	return Tool{
		Name:        name,
		Description: description,
		Schema:      schema,
		Run: func(ctx context.Context, raw json.RawMessage) (string, error) {
			var args T
			if len(raw) > 0 {
				if err := json.Unmarshal(raw, &args); err != nil {
					return "", fmt.Errorf("arguments: %w", err)
				}
			}
			return run(ctx, args)
		},
	}, nil
}

// objectSchema checks that raw is a JSON object, the only kind of schema
// a tool or an answer can have.
func objectSchema(raw json.RawMessage) error {
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return fmt.Errorf("schema is not a JSON object: %w", err)
	}
	if obj == nil {
		return errors.New("schema is not a JSON object: null")
	}
	return nil
}
