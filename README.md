# ai-chatbot

Week 4 milestone CLI chatbot for prompt engineering practice.

## Features
- Multi-turn memory via preserved `messages` history.
- Streaming output using `client.responses.stream(...)`.
- Tool calling with a built-in calculator function.
- Structured system prompt for role/task/constraints.
- Graceful error handling for API and tool errors.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
# optional
export OPENAI_MODEL="gpt-4.1-mini"
```

## Run
```bash
python chatbot.py
```

Type `exit` or `quit` to stop.
