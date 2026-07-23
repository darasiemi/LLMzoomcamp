from typing import Any, List

from google import genai
from google.genai import types
from pydantic import BaseModel

from toyaikit.llm import LLMClient
from toyaikit.tools import Tools


class GeminiClient(LLMClient):
    """LLM client using Gemini's generate_content API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        client: genai.Client | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.client = client or genai.Client()
        self.extra_kwargs = extra_kwargs or {}

    def _convert_tools(
        self,
        tools: Tools | None,
    ) -> list[types.Tool] | None:
        """
        Convert ToyAIKit/OpenAI-style tool definitions into
        Gemini FunctionDeclaration objects.
        """
        if tools is None:
            return None

        tool_definitions = tools.get_tools()

        if not tool_definitions:
            return None

        function_declarations = []

        for tool in tool_definitions:
            if isinstance(tool, dict):
                name = tool.get("name")
                description = tool.get("description", "")
                parameters = tool.get("parameters")
            else:
                name = getattr(tool, "name", None)
                description = getattr(tool, "description", "")
                parameters = getattr(tool, "parameters", None)

            if not name:
                raise ValueError(
                    f"Tool definition has no name: {tool!r}"
                )

            if parameters is None:
                parameters = {
                    "type": "object",
                    "properties": {},
                }

            function_declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description or "",
                    parameters_json_schema=parameters,
                )
            )

        return [
            types.Tool(
                function_declarations=function_declarations
            )
        ]

    def send_request(
        self,
        chat_messages: List,
        tools: Tools | None = None,
        output_format: type[BaseModel] | None = None,
    ):
        gemini_tools = self._convert_tools(tools)

        config_kwargs = dict(self.extra_kwargs)

        if gemini_tools is not None:
            config_kwargs["tools"] = gemini_tools

        if output_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = output_format

        config = types.GenerateContentConfig(**config_kwargs)

        return self.client.models.generate_content(
            model=self.model,
            contents=chat_messages,
            config=config,
        )