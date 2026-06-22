import json
import uuid
from types import SimpleNamespace

from pydantic import BaseModel
from google.genai import types

from toyaikit.chat.runners import (
    BaseToolUsingRunner,
    RunnerCallback,
    LoopResult,
)
from toyaikit.pricing import TokenUsage


class GeminiRunner(BaseToolUsingRunner):
    """Runner for Gemini generate_content API."""

    def _initialize_messages(self, previous_messages: list = None) -> list:
        if previous_messages is None or len(previous_messages) == 0:
            return [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=f"System instruction: {self.developer_prompt}"
                        )
                    ],
                )
            ]

        return list(previous_messages)

    def _make_user_message(self, text: str):
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )

    def _make_model_message(self, response):
        return response.candidates[0].content

    def _make_function_call_for_toyaikit(self, gemini_call):
        """
        Convert Gemini function_call into the OpenAI-like shape
        expected by ToyAIKit's original Tools.function_call().
        """
        return SimpleNamespace(
            name=gemini_call.name,
            arguments=json.dumps(dict(gemini_call.args or {})),
            call_id=str(uuid.uuid4()),
        )

    def _extract_tool_output(self, result):
        if isinstance(result, dict):
            return result.get("output", result)

        return getattr(result, "output", str(result))

    def _make_function_response_message(self, name: str, result):
        output = self._extract_tool_output(result)

        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=name,
                    response={"result": output},
                )
            ],
        )
    def _display_search_results(self, result):
        output = self._extract_tool_output(result)

        try:
            data = json.loads(output)
        except Exception:
            print(output)
            return

        print("\n=== RETRIEVED RESULTS ===")

        for i, item in enumerate(data, start=1):
            print(f"\n[{i}]")
            print("id:", item.get("id"))
            print("course:", item.get("course"))
            print("section:", item.get("section"))
            print("question:", item.get("question"))

            answer = item.get("answer", "")
            print("answer:", answer[:800], "..." if len(answer) > 800 else "")

        print("\n=========================\n")

    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
        output_format: BaseModel = None,
    ) -> LoopResult:
        if previous_messages is None or len(previous_messages) == 0:
            chat_messages = self._initialize_messages()
            prev_messages_len = 0
        else:
            chat_messages = list(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append(self._make_user_message(prompt))

        total_input_tokens = 0
        total_output_tokens = 0
        last_message = None

        while True:
            response = self.llm_client.send_request(
                chat_messages=chat_messages,
                tools=self.tools,
                output_format=output_format,
            )

            if callback:
                callback.on_response(response)

            usage = getattr(response, "usage_metadata", None)
            if usage:
                total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0

            model_message = self._make_model_message(response)
            chat_messages.append(model_message)

            has_function_calls = False

            for part in model_message.parts:
                function_call = getattr(part, "function_call", None)

                if function_call:
                    tool_call = self._make_function_call_for_toyaikit(function_call)

                    # This calls ToyAIKit's original Tools.function_call()
                    print("\n=== TOOL CALL ===")
                    print("name:", tool_call.name)
                    print("arguments:", tool_call.arguments)

                    result = self.tools.function_call(tool_call)

                    self._display_search_results(result)

                    chat_messages.append(
                        self._make_function_response_message(
                            name=function_call.name,
                            result=result,
                        )
                    )

                    if callback:
                        callback.on_function_call(
                            tool_call,
                            self._extract_tool_output(result),
                        )

                    has_function_calls = True

                elif getattr(part, "text", None):
                    last_message_text = part.text

                    if callback:
                        callback.on_message(last_message_text)

                    if output_format:
                        last_message = output_format.model_validate_json(
                            last_message_text
                        )
                    else:
                        last_message = last_message_text

            if not has_function_calls:
                break

        token_usage = TokenUsage(
            model=self.llm_client.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        cost_info = self.pricing_config.calculate_cost(
            self.llm_client.model,
            total_input_tokens,
            total_output_tokens,
        )

        return LoopResult(
            new_messages=chat_messages[prev_messages_len:],
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost_info,
            last_message=last_message,
        )