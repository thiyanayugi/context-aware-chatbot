# Contributing to Context-Aware Chatbot

Thank you for considering contributing to this project! 🎉

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone git@github.com:YOUR-USERNAME/context-aware-chatbot.git
   cd context-aware-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## Code Style

This project follows these conventions:

- **Formatting**: Black (line length 88)
- **Linting**: Ruff
- **Type hints**: Use mypy for static type checking
- **Docstrings**: Google style

Run formatters before committing:
```bash
black app/ tests/
ruff check app/ tests/ --fix
mypy app/
```

## Engineering Guidelines

Follow the rules in [skills.md](skills.md):

- All LLM calls must go through `llm/client.py`
- Track tokens for every LLM call
- Add input filtering and output validation
- Use structured logging
- Write async code where possible

## Testing

Run the test suite before submitting:
```bash
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html
```

Write tests for new features:
```python
# tests/test_new_feature.py
import pytest
from app.your_module import YourClass

@pytest.mark.asyncio
async def test_your_feature():
    instance = YourClass()
    result = await instance.do_something()
    assert result.is_valid
```

## Commit Messages

Use conventional commits:

- `feat: add new feature`
- `fix: resolve bug in token counting`
- `docs: update README with examples`
- `refactor: simplify memory management`
- `test: add tests for guardrails`
- `chore: update dependencies`

Example:
```
feat: add streaming response support

- Implement streaming for OpenAI and Anthropic
- Add async generator for token-by-token output
- Update tests to cover streaming scenarios
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update docs

3. **Test thoroughly**
   ```bash
   pytest tests/ -v
   black app/ tests/
   ruff check app/ tests/
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   git push origin feat/your-feature-name
   ```

5. **Open a Pull Request**
   - Describe what you changed and why
   - Link to any related issues
   - Request review

## Areas for Contribution

Looking for ideas? Here are some areas that need work:

### High Priority
- [ ] FastAPI REST API implementation
- [ ] Streaming response support
- [ ] Multi-session conversation management
- [ ] Database persistence (PostgreSQL/SQLite)

### Medium Priority
- [ ] Admin dashboard for monitoring
- [ ] Rate limiting middleware
- [ ] User authentication
- [ ] Conversation export/import

### Low Priority
- [ ] Additional LLM providers (Google, Cohere)
- [ ] Custom embedding models
- [ ] Conversation templates
- [ ] Plugin system

### Documentation
- [ ] Video tutorials
- [ ] Architecture deep-dive blog post
- [ ] Code examples for common use cases
- [ ] API reference documentation

## Questions?

- **Found a bug?** [Open an issue](https://github.com/thiyanayugi/context-aware-chatbot/issues/new)
- **Have a question?** [Start a discussion](https://github.com/thiyanayugi/context-aware-chatbot/discussions)
- **Want to chat?** Reach out via [LinkedIn](https://www.linkedin.com/in/thiyanayugi-mariraj-a2b1b820b)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
