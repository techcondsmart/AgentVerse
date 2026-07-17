# Modified from AutoGPT https://github.com/Significant-Gravitas/AutoGPT/blob/release-v0.4.7/autogpt/llm/utils/token_counter.py

import tiktoken
from typing import List, Union, Dict
from agentverse.logging import logger
from agentverse.message import Message
from agentverse.llms import LOCAL_LLMS, LOCAL_LLMS_MAPPING


def _fallback_encode_len(text: str) -> int:
    """Token count for models tiktoken doesn't know (Gemini, vLLM aliases, ...).

    Tries the generic cl100k_base encoding; if that is unavailable too (e.g. no
    network to fetch the BPE file), approximates ~4 chars per token. Counts are
    only used to budget prompt trimming, so an approximation is safe.
    """
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def count_string_tokens(prompt: str = "", model: str = "gpt-3.5-turbo") -> int:
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        return len(tiktoken.encoding_for_model(model).encode(prompt))
    elif model.lower() in LOCAL_LLMS or model in LOCAL_LLMS:
        from transformers import AutoTokenizer
        encoding = AutoTokenizer.from_pretrained(LOCAL_LLMS_MAPPING[model.lower()]['hf_model_name'])
        return len(encoding.encode(prompt))
    # OpenAI-compatible third-party models (Gemini, OpenRouter, ...): the old
    # code silently returned None here, crashing prompt-budget arithmetic.
    return _fallback_encode_len(prompt)


def count_message_tokens(
    messages: Union[Dict, List[Dict]], model: str = "gpt-3.5-turbo"
) -> int:
    if isinstance(messages, dict):
        messages = [messages]

    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
        encoding_model = "gpt-3.5-turbo"
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
        encoding_model = "gpt-4"
    elif model.lower() in LOCAL_LLMS or model in LOCAL_LLMS:
        from transformers import AutoTokenizer

        encoding = AutoTokenizer.from_pretrained(LOCAL_LLMS_MAPPING[model.lower()]['hf_model_name'])
    else:
        # OpenAI-compatible third-party models (Gemini, OpenRouter, vLLM
        # aliases, ...): use the gpt-4 message framing with a generic encoding
        # instead of raising — counts only budget prompt trimming.
        tokens_per_message = 3
        tokens_per_name = 1

        class _ApproxEncoding:
            def encode(self, text: str):
                return [0] * _fallback_encode_len(text)

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = _ApproxEncoding()
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            logger.warn("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # TODO: count number of function_call's token more accurately
            if key == "function_call":
                num_tokens += len(encoding.encode(value["name"]))
                num_tokens += len(encoding.encode(value["arguments"]))
            else:
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
