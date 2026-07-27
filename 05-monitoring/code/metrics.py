import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import GeminiRAG

from google.genai import types
from typing import Optional


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)

# def calculate_cost(model: str, usage) -> float:
#     if usage is None:
#         return 0.0

#     prompt_tokens = usage.prompt_token_count or 0
#     output_tokens = usage.candidates_token_count or 0
#     thinking_tokens = usage.thoughts_token_count or 0

#     if "gemini-2.5-flash" in model.lower():
#         return (
#             prompt_tokens * 0.15
#             + (output_tokens + thinking_tokens) * 0.60
#         ) / 1_000_000

#     return 0.0


def calculate_cost(model, usage):
    cost = 0.0

    if "gemini-2.5-flash" in model:
        input_cost = usage.prompt_token_count * 0.15 / 1_000_000
        output_cost = usage.candidates_token_count * 0.60 / 1_000_000
        cost = input_cost + output_cost

    return cost

class RAGWithMetrics(GeminiRAG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: Optional[LLMCallRecord] = None

    def llm(self, prompt: str) -> str:
        start_time = time.perf_counter()

        response = self._call_llm(prompt)

        response_time = time.perf_counter() - start_time
        answer = response.text or ""

        self._log_response(
            prompt=prompt,
            response=response,
            answer=answer,
            response_time=response_time,
        )

        return answer

    def _call_llm(self, prompt: str):
        return self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
            ),
        )

    def _log_response(
        self,
        prompt: str,
        response,
        answer: str,
        response_time: float,
    ) -> None:
        usage = response.usage_metadata

        if usage is None:
            prompt_tokens = 0
            completion_tokens = 0
            thinking_tokens = 0
            total_tokens = 0
            cost = 0.0
        else:
            prompt_tokens = usage.prompt_token_count or 0
            completion_tokens = usage.candidates_token_count or 0
            thinking_tokens = usage.thoughts_token_count or 0
            total_tokens = usage.total_token_count or 0
            cost = calculate_cost(self.model, usage)

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            cost=cost,
        )

        print(call_record)
        self.last_call = call_record

