from typing import List
from pydantic import BaseModel
from google import genai
from google.genai import types

from toyaikit.llm import LLMClient
from toyaikit.tools import Tools


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        client: genai.Client = None,
        extra_kwargs: dict = None,
    ):
        self.model = model
        self.client = client or genai.Client()
        self.extra_kwargs = extra_kwargs or {}

    def send_request(
        self,
        chat_messages: List,
        tools: Tools = None,
        output_format: BaseModel = None,
    ):
        tools_list = tools.get_tools() if tools is not None else None

        config = types.GenerateContentConfig(
            tools=tools_list,
            **self.extra_kwargs,
        )

        if output_format is not None:
            config.response_mime_type = "application/json"
            config.response_schema = output_format

        return self.client.models.generate_content(
            model=self.model,
            contents=chat_messages,
            config=config,
        )