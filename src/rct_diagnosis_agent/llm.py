from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment]


class StructuredLLM:
    def invoke_structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        raise NotImplementedError


class OpenAIStructuredLLM(StructuredLLM):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required to use OpenAIStructuredLLM.")
        self._model = ChatOpenAI(model=model, temperature=temperature)

    def invoke_structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        structured_model = self._model.with_structured_output(schema)
        return structured_model.invoke(prompt)


def build_prompt(system_instruction: str, **sections: Any) -> str:
    parts = [system_instruction.strip()]
    for name, value in sections.items():
        parts.append(f"{name.upper()}:\n{value}")
    return "\n\n".join(parts)
