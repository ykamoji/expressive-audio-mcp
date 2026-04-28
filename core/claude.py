from anthropic import Anthropic
from anthropic.types import Message
import uuid


class Claude:
    def __init__(self, model: str):
        self.client = Anthropic()
        self.model = model

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }

        if thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        message = self.client.messages.create(**params)
        return message


class GeminiBridge:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.tool_id_to_signature = {}
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            self.client = None

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        from google.genai import types
        from anthropic.types import TextBlock, ToolUseBlock, Usage

        if self.client is None:
            raise ImportError("google-genai SDK is not installed. Please install it to use GeminiBridge.")

        # Pre-scan messages to map tool_use_id to tool name
        tool_id_to_name = {}
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_id_to_name[block.get("id")] = block.get("name")
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_id_to_name[getattr(block, "id", None)] = getattr(block, "name", None)

        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]

            # print(content)
            parts = []
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        b_type = block.get("type")
                        if b_type == "text":
                            parts.append(types.Part.from_text(text=block["text"]))
                        elif b_type == "tool_use":
                            tool_use_id = block.get("id")
                            sig = self.tool_id_to_signature.get(tool_use_id)
                            if sig:
                                parts.append(types.Part(
                                    function_call=types.FunctionCall(name=block["name"], args=block["input"]),
                                    thought_signature=sig
                                ))
                            else:
                                parts.append(types.Part.from_function_call(
                                    name=block["name"],
                                    args=block["input"]
                                ))
                        elif b_type == "tool_result":
                            tool_use_id = block.get("tool_use_id")
                            tool_name = block.get("name") or tool_id_to_name.get(tool_use_id, "unknown_tool")
                            parts.append(types.Part.from_function_response(
                                name=tool_name,
                                response={"result": block.get("content", "")}
                            ))
                        elif "text" in block:
                            parts.append(types.Part.from_text(text=block["text"]))
                    else:
                        if hasattr(block, "type"):
                            if block.type == "text":
                                parts.append(types.Part.from_text(text=block.text))
                            elif block.type == "tool_use":
                                tool_use_id = getattr(block, "id", None)
                                sig = self.tool_id_to_signature.get(tool_use_id) if tool_use_id else None
                                if sig:
                                    parts.append(types.Part(
                                        function_call=types.FunctionCall(name=block.name, args=block.input),
                                        thought_signature=sig
                                    ))
                                else:
                                    parts.append(types.Part.from_function_call(
                                        name=block.name,
                                        args=block.input
                                    ))
                            elif block.type == "tool_result":
                                tool_use_id = getattr(block, "tool_use_id", None)
                                tool_name = getattr(block, "name", None) or tool_id_to_name.get(tool_use_id, "unknown_tool")
                                parts.append(types.Part.from_function_response(
                                    name=tool_name,
                                    response={"result": block.content}
                                ))
            if not parts and content:
                parts.append(types.Part.from_text(text=str(content)))

            gemini_messages.append(types.Content(role=role, parts=parts))

        gemini_tools = []
        if tools:
            for tool in tools:
                gemini_tools.append(types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            parameters=tool.get("input_schema")
                        )
                    ]
                ))

        thinking_config = None
        if thinking and thinking_budget > 0:
            thinking_config = types.ThinkingConfig(
                    budget_tokens=thinking_budget,
                )

        config = types.GenerateContentConfig(
            temperature=temperature,
            stop_sequences=stop_sequences if stop_sequences else None,
            max_output_tokens=8000,
            thinking_config=thinking_config
        )
        if system:
            config.system_instruction = system
        if tools:
            config.tools = gemini_tools

        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_messages,
            config=config,
        )

        content_blocks = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content_blocks.append(TextBlock(type="text", text=part.text))
                elif part.function_call:
                    # In Anthropic, we need an id for tool_use. Gemini doesn't provide ids for function calls in the same way, so we generate one.
                    # print(part.function_call.name)
                    tool_use_id = f"toolu_{uuid.uuid4().hex[:16]}"
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        self.tool_id_to_signature[tool_use_id] = part.thought_signature
                    content_blocks.append(ToolUseBlock(
                        type="tool_use",
                        id=tool_use_id,
                        name=part.function_call.name,
                        input=part.function_call.args if part.function_call.args is not None else {}
                    ))

        stop_reason = None
        if response.candidates:
            # Convert Gemini finish reason to Anthropic stop reason
            # Gemini: "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"
            finish_reason = getattr(response.candidates[0], "finish_reason", "STOP")
            if any(isinstance(b, ToolUseBlock) for b in content_blocks):
                stop_reason = "tool_use"
            elif finish_reason == "MAX_TOKENS":
                stop_reason = "max_tokens"
            elif finish_reason == "STOP":
                stop_reason = "end_turn"
            else:
                stop_reason = "end_turn"

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        return Message(
            id=f"msg_{uuid.uuid4().hex[:16]}",
            content=content_blocks,
            model=self.model,
            role="assistant",
            type="message",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
            stop_sequence=None
        )
