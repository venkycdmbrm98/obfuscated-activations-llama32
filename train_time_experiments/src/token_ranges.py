"""Token-range predicates shared by the lightweight PGD runners."""


def _special_token_id(tokenizer, token_name, fallback=None):
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id is None or token_id == tokenizer.unk_token_id:
        if fallback is None:
            raise ValueError(f"Tokenizer has no token ID for {token_name}")
        return fallback
    return token_id


def _make_sequence_end_matcher(sequence):
    sequence = tuple(int(token_id) for token_id in sequence)
    if not sequence:
        raise ValueError("Cannot create a matcher for an empty token sequence")

    def matches_sequence_end(seq_idx, token, tokens):
        start = seq_idx - len(sequence) + 1
        if start < 0 or token != sequence[-1]:
            return False
        return tuple(int(value) for value in tokens[start : seq_idx + 1]) == sequence

    return matches_sequence_end


def _make_token_after_sequence(sequence):
    sequence = tuple(int(token_id) for token_id in sequence)

    def is_token_after_sequence(seq_idx, token, tokens):
        start = seq_idx - len(sequence)
        if start < 0:
            return False
        return tuple(int(value) for value in tokens[start:seq_idx]) == sequence

    return is_token_after_sequence


def get_token_ranges(masking_type, tokenizer):
    """Return Llama-3 chat masking predicates for generation or instruction."""

    start_header_token = _special_token_id(tokenizer, "<|start_header_id|>")
    end_header_token = _special_token_id(tokenizer, "<|end_header_id|>")
    eot_token = _special_token_id(tokenizer, "<|eot_id|>", tokenizer.eos_token_id)
    header_break = tokenizer.encode("\n\n", add_special_tokens=False)
    if not header_break:
        raise ValueError("Tokenizer produced no tokens for the role-header separator")

    def role_header(role):
        role_tokens = tokenizer.encode(role, add_special_tokens=False)
        if not role_tokens:
            raise ValueError(f"Tokenizer produced no tokens for role {role!r}")
        return [start_header_token, *role_tokens, end_header_token]

    assistant_header = role_header("assistant")
    assistant_header_break = [*assistant_header, *header_break]
    user_header_break = [*role_header("user"), *header_break]
    is_after_assistant_header = _make_sequence_end_matcher(assistant_header)
    is_after_assistant_header_break = _make_sequence_end_matcher(
        assistant_header_break
    )
    is_after_user_header_break = _make_sequence_end_matcher(user_header_break)
    is_first_assistant_content_token = _make_token_after_sequence(
        assistant_header_break
    )

    common = {
        "only_return_on_tokens_between": [
            is_after_assistant_header_break,
            eot_token,
        ],
        "only_choose_prompt_tokens_between": [
            is_after_user_header_break,
            eot_token,
        ],
    }
    if masking_type == "generation":
        return {
            **common,
            "only_probe_tokens_between": [
                is_after_assistant_header_break,
                eot_token,
            ],
        }
    if masking_type == "instruction":
        return {
            **common,
            "only_probe_tokens_between": [
                is_after_assistant_header,
                is_first_assistant_content_token,
            ],
        }
    raise ValueError(f"Unknown masking_type: {masking_type}")
