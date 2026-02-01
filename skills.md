# Engineering Rules for Context-Aware Chatbot

This document defines how we build AI systems in this project.

---

## Language & Framework

- Use **Python 3.11+** as the primary language
- Use **FastAPI** for any API endpoints
- Use **Pydantic** for data validation and settings management
- Use **async/await** for I/O operations where applicable

---

## Architecture Principles

- **Modular design**: Each module has a single responsibility
- **No giant files**: Keep files under 200 lines; split if larger
- **Dependency injection**: Pass dependencies explicitly, avoid globals
- **Configuration centralized**: All settings live in `config.py`

---

## LLM Integration Rules

- **All LLM calls must go through `llm/client.py`** — never call APIs directly elsewhere
- **Track token usage** for every LLM call (input + output tokens)
- **Log every LLM interaction** with timestamp, tokens used, and latency
- **Never send full chat history** without checking token limits first
- **Set max_tokens explicitly** on every call to prevent runaway costs

---

## Context Window Management

- **Always estimate tokens** before building prompts
- **Summarize old memory** when context exceeds 70% of max window
- **Keep a token budget**: system prompt + context + user input + response buffer
- **Prefer recent messages** over older ones when trimming

---

## Memory Management

- **Short-term memory**: Last N messages, token-aware
- **Long-term memory**: Summaries stored in vector DB
- **Summarization trigger**: When short-term exceeds threshold
- **Retrieval**: Fetch relevant long-term context for each query

---

## Guardrails & Safety

- **Filter all user input** before processing
- **Detect prompt injection** patterns and block them
- **Validate all LLM outputs** before returning to user
- **Never let user text override system instructions**
- **Enforce response length limits**

---

## Error Handling

- **Wrap external calls** in try/except with proper logging
- **Fail gracefully**: Return meaningful error messages
- **Retry with backoff** for transient API failures
- **Never expose internal errors** to end users

---

## Logging & Observability

- **Log all important operations**: LLM calls, memory operations, errors
- **Use structured logging** (JSON format for production)
- **Include context**: request ID, user ID, operation type
- **Log token usage** for cost tracking

---

## Code Quality

- **Write clean, readable code** — optimize for humans first
- **Add comments** for non-obvious logic only
- **Use type hints** for function signatures
- **Follow PEP 8** style guidelines
- **Name functions/variables descriptively**

---

## Testing

- **Write tests** for critical paths: token counting, summarization, guardrails
- **Mock LLM calls** in unit tests to avoid API costs
- **Test edge cases**: empty input, max tokens, injection attempts
- **Integration tests** for the full chat flow

---

## Security

- **Store API keys** in environment variables, never in code
- **Sanitize all inputs** before logging (no PII in logs)
- **Rate limit** API endpoints to prevent abuse
- **Validate file paths** if any file operations are used
