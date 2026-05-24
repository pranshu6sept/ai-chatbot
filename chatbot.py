#!/usr/bin/env python3
"""Week 4 milestone: CLI chatbot with memory, streaming, and a calculator tool."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List

from openai import OpenAI

SYSTEM_PROMPT = """
You are a senior Python CLI assistant.
Task:
- Help the user conversationally.
- Use tools when calculations are requested.
Constraints:
- Be accurate and concise.
- If uncertain, ask a clarifying question.
- When using calculator results, explain in plain language.
Output style:
- Normal chat text unless user asks for a structured format.
""".strip()


class CalculatorTool:
    name = "calculator"
    description = "Safely evaluate a basic arithmetic expression using +, -, *, /, parentheses, and decimals."

    @staticmethod
    def run(expression: str) -> str:
        cleaned = expression.strip()
        if not cleaned:
            raise ValueError("Empty expression")
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", cleaned):
            raise ValueError("Expression contains unsupported characters")

        try:
            # Safe eval by removing builtins and restricting characters above.
            result = eval(cleaned, {"__builtins__": {}}, {})
        except ZeroDivisionError as exc:
            raise ValueError("Division by zero") from exc
        except Exception as exc:
            raise ValueError(f"Invalid expression: {expression}") from exc
        return str(result)


def build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "calculator",
            "description": CalculatorTool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, e.g. (120+90)/(2+1.5+0.5)",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        }
    ]


def run_chat() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("CLI Chatbot ready. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.responses.create(
                model=model,
                input=messages,
                tools=build_tools(),
                temperature=0.6,
            )

            while response.output and response.output[0].type == "function_call":
                call = response.output[0]
                args = json.loads(call.arguments)
                if call.name != "calculator":
                    tool_output = "Unsupported tool"
                else:
                    try:
                        tool_output = CalculatorTool.run(args.get("expression", ""))
                    except ValueError as exc:
                        tool_output = f"Calculator error: {exc}"

                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": tool_output,
                    }
                )

                response = client.responses.create(
                    model=model,
                    input=messages,
                    tools=build_tools(),
                    temperature=0.6,
                )

            print("Assistant: ", end="", flush=True)
            stream = client.responses.stream(
                model=model,
                input=messages,
                tools=build_tools(),
                temperature=0.6,
            )
            final_text = ""
            with stream as events:
                for event in events:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                    elif event.type == "response.output_text.done":
                        final_text = event.text
            print()

            messages.append({"role": "assistant", "content": final_text})

        except Exception as exc:
            print(f"Assistant error: {exc}")


if __name__ == "__main__":
    run_chat()
