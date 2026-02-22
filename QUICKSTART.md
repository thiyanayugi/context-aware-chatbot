# Quick Start Guide

Get the chatbot running in 5 minutes! ⚡

## Prerequisites

- Python 3.10 or higher
- OpenAI or Anthropic API key

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/thiyanayugi/context-aware-chatbot.git
cd context-aware-chatbot
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
# Option A: OpenAI
OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Option B: Anthropic
ANTHROPIC_API_KEY=your-key-here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-haiku-20240307
```

### Step 5: Run!

```bash
python -m app.main
```

## First Conversation

```
You: Hello!
Assistant: Hi! How can I help you today?

You: What's the weather like?
Assistant: I don't have access to real-time weather data, but I can help you find...

You: exit
```

## What's Happening?

When you chat, the bot:

1. ✅ **Filters** your input for safety
2. 🔍 **Searches** past conversations for relevant context
3. 🧠 **Manages** conversation memory to fit token limits
4. 🤖 **Generates** a response using your chosen LLM
5. ✅ **Validates** the output for safety
6. 💾 **Stores** the conversation for future reference

## Common Configurations

### Use Claude 3.5 Sonnet (Better Quality)

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
MAX_CONTEXT_TOKENS=16000
```

### Use GPT-4 (OpenAI)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
MAX_CONTEXT_TOKENS=8000
```

### Budget Mode (Fastest/Cheapest)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
MAX_CONTEXT_TOKENS=4000
SUMMARIZATION_THRESHOLD=0.6
```

## Next Steps

- 📖 Read the [full README](README.md) for architecture details
- 🛠️ Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- 📝 See [skills.md](skills.md) for engineering guidelines
- 🐛 Report issues on [GitHub](https://github.com/thiyanayugi/context-aware-chatbot/issues)

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'app'`  
**Solution**: Run from project root: `python -m app.main`

**Problem**: `No API key found`  
**Solution**: Check your `.env` file has the correct key

**Problem**: Slow responses  
**Solution**: Use a faster model (gpt-4o-mini or claude-haiku)

More help in the [Troubleshooting section](README.md#troubleshooting).

---

**Questions?** [Open an issue](https://github.com/thiyanayugi/context-aware-chatbot/issues) 💬
