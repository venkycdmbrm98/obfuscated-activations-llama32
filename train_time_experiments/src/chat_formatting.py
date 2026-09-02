"""Chat-template helpers for the serialized Llama 3 jailbreak dataset."""

from __future__ import annotations

from dataclasses import dataclass


BEGIN_OF_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
END_OF_TURN = "<|eot_id|>"
HEADER_BREAK = "\n\n"
LLAMA32_CHAT_TEMPLATE_DATE = "26 Jul 2024"


@dataclass(frozen=True)
class ParsedChat:
    messages: list[dict[str, str]]
    has_generation_prompt: bool


def parse_llama3_chat(serialized_chat: str) -> ParsedChat:
    """Parse Llama 3 chat markup without treating message text as a template."""
    if not isinstance(serialized_chat, str) or not serialized_chat:
        raise ValueError("Expected a non-empty serialized chat string")

    cursor = 0
    if serialized_chat.startswith(BEGIN_OF_TEXT):
        cursor = len(BEGIN_OF_TEXT)

    messages: list[dict[str, str]] = []
    has_generation_prompt = False
    while cursor < len(serialized_chat):
        if not serialized_chat.startswith(START_HEADER, cursor):
            snippet = serialized_chat[cursor : cursor + 40]
            raise ValueError(
                f"Expected {START_HEADER!r} at offset {cursor}, got {snippet!r}"
            )

        role_start = cursor + len(START_HEADER)
        role_end = serialized_chat.find(END_HEADER, role_start)
        if role_end < 0:
            raise ValueError(f"Missing {END_HEADER!r} after offset {role_start}")
        role = serialized_chat[role_start:role_end]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role {role!r}")

        content_start = role_end + len(END_HEADER)
        if not serialized_chat.startswith(HEADER_BREAK, content_start):
            raise ValueError(
                f"Expected {HEADER_BREAK!r} after the {role!r} header"
            )
        content_start += len(HEADER_BREAK)

        turn_end = serialized_chat.find(END_OF_TURN, content_start)
        if turn_end < 0:
            trailing_content = serialized_chat[content_start:]
            if role != "assistant" or trailing_content.strip():
                raise ValueError(
                    "Only an empty final assistant turn may omit the end-of-turn token"
                )
            has_generation_prompt = True
            cursor = len(serialized_chat)
            continue

        messages.append(
            {"role": role, "content": serialized_chat[content_start:turn_end]}
        )
        cursor = turn_end + len(END_OF_TURN)

    if not messages:
        raise ValueError("Serialized chat did not contain any complete messages")
    return ParsedChat(messages=messages, has_generation_prompt=has_generation_prompt)


def render_chat(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The selected tokenizer has no chat_template; OAT requires an instruct model"
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        date_string=LLAMA32_CHAT_TEMPLATE_DATE,
    )


def _without_template_system_metadata(messages):
    """Remove metadata that Llama 3.2's template injects around system text."""
    normalized = [dict(message) for message in messages]
    if not normalized or normalized[0]["role"] != "system":
        return normalized

    content = normalized[0]["content"]
    prefix = "Cutting Knowledge Date: December 2023\nToday Date: "
    if not content.startswith(prefix):
        return normalized
    metadata_end = content.find("\n\n", len(prefix))
    if metadata_end < 0:
        raise ValueError("Malformed Llama 3.2 system metadata block")
    remaining_system_text = content[metadata_end + 2 :]
    if remaining_system_text:
        normalized[0]["content"] = remaining_system_text
    else:
        normalized.pop(0)
    return normalized


def format_dataset_chat(
    tokenizer,
    prompt: str,
    completion: str | None = None,
) -> str:
    """Re-render a legacy serialized dataset row with the model's chat template."""
    parsed = parse_llama3_chat(prompt)
    if not parsed.has_generation_prompt:
        raise ValueError("Dataset prompt must end with an empty assistant generation turn")

    messages = _without_template_system_metadata(parsed.messages)
    if completion is None:
        return render_chat(tokenizer, messages, add_generation_prompt=True)

    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("Dataset completion must be a non-empty string")
    messages.append({"role": "assistant", "content": completion})
    return render_chat(tokenizer, messages, add_generation_prompt=False)


def append_to_last_user_message(tokenizer, serialized_chat: str, suffix: str) -> str:
    """Append attack text without re-rendering model-injected system metadata."""
    parsed = parse_llama3_chat(serialized_chat)
    if not any(message["role"] == "user" for message in parsed.messages):
        raise ValueError("Serialized chat has no user message")

    user_header = f"{START_HEADER}user{END_HEADER}{HEADER_BREAK}"
    user_start = serialized_chat.rfind(user_header)
    if user_start < 0:
        raise ValueError("Could not locate the final serialized user header")
    user_end = serialized_chat.find(END_OF_TURN, user_start + len(user_header))
    if user_end < 0:
        raise ValueError("Final serialized user message has no end-of-turn token")
    return serialized_chat[:user_end] + suffix + serialized_chat[user_end:]


def chat_template_token_overhead(tokenizer) -> int:
    """Measure tokens added by re-rendering a legacy single-turn Llama 3 chat."""
    legacy = (
        f"{BEGIN_OF_TEXT}{START_HEADER}user{END_HEADER}{HEADER_BREAK}x"
        f"{END_OF_TURN}{START_HEADER}assistant{END_HEADER}{HEADER_BREAK}y"
    )
    rendered = render_chat(
        tokenizer,
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        add_generation_prompt=False,
    )
    legacy_length = len(tokenizer.encode(legacy, add_special_tokens=False))
    rendered_length = len(tokenizer.encode(rendered, add_special_tokens=False))
    overhead = rendered_length - legacy_length
    if overhead < 0:
        raise ValueError("Chat template unexpectedly removed tokens from the legacy format")
    return overhead
