import json
import uuid
from types import SimpleNamespace

from google.genai import types
from toyaikit.tools import Tools


class GeminiTools(Tools):
    """Gemini-compatible adapter around ToyAIKit's Tools."""

    def get_tools(self):
        function_declarations = []

        for schema in self.tools.values():
            function_declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    parameters=self._convert_parameters(schema["parameters"]),
                )
            )

        if not function_declarations:
            return None

        return [types.Tool(function_declarations=function_declarations)]

    def function_call(self, tool_call_response):
        """
        Accepts either:
        - Gemini function_call objects: name + args
        - ToyAIKit/OpenAI-like objects: name + arguments + call_id
        """

        if hasattr(tool_call_response, "args"):
            proxy = SimpleNamespace(
                name=tool_call_response.name,
                arguments=json.dumps(dict(tool_call_response.args or {})),
                call_id=str(uuid.uuid4()),
            )
            return super().function_call(proxy)

        if not hasattr(tool_call_response, "call_id"):
            tool_call_response.call_id = str(uuid.uuid4())

        return super().function_call(tool_call_response)

    def _convert_parameters(self, parameters: dict):
        properties = {}

        for name, prop in parameters.get("properties", {}).items():
            properties[name] = {
                "type": self._convert_type(prop.get("type", "string")),
                "description": prop.get("description", ""),
            }

        return {
            "type": "object",
            "properties": properties,
            "required": parameters.get("required", []),
        }

    def _convert_type(self, json_type: str):
        mapping = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        return mapping.get(json_type, "string")