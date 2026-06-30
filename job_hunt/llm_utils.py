import json
import os
import subprocess
import time

from openai import (
    APIStatusError,
    APITimeoutError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from job_hunt.log import get_logger

logger = get_logger()

# Per-request timeout (seconds) for HTTP-based LLM providers.
_LLM_REQUEST_TIMEOUT = 60.0  # tightened from 120s — a stalled free model wastes the batch

# ── Per-process model cooldown ─────────────────────────────────────────────
# When a model hits quota (429), mark it unavailable for _QUOTA_COOLDOWN_SECS.
# This prevents the same exhausted model from being retried on every subsequent
# company in the batch, which previously wasted ~60 s per company.
_model_cooldown: dict[str, float] = {}
_QUOTA_COOLDOWN_SECS = 600  # 10 min — typically covers the rest of a batch run


def _is_model_available(model: str) -> bool:
    return time.time() >= _model_cooldown.get(model, 0.0)


def _mark_model_exhausted(model: str) -> None:
    _model_cooldown[model] = time.time() + _QUOTA_COOLDOWN_SECS
    logger.warning(f"Model {model!r} quota exhausted — on cooldown for {_QUOTA_COOLDOWN_SECS // 60} min")


def _make_openrouter_client(config: dict) -> OpenAI:
    return OpenAI(
        api_key=config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=_LLM_REQUEST_TIMEOUT,
    )


def _chat_with_anthropic(config: dict, messages: list[dict], temperature: float, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install 'autopilot-jobs[claude]'")
    api_key = config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
    model = config.get("anthropic_model", "claude-haiku-4-5-20251001")
    logger.debug(f"LLM call → Anthropic / {model}")
    t0 = time.time()
    client = anthropic.Anthropic(api_key=api_key, timeout=_LLM_REQUEST_TIMEOUT)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_msgs = [m for m in messages if m["role"] != "system"]
    kwargs: dict = {
        "model": model,
        "messages": user_msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    r = client.messages.create(**kwargs)
    elapsed = time.time() - t0
    text = r.content[0].text
    logger.debug(f"LLM response: {len(text)} chars in {elapsed:.1f}s (input={r.usage.input_tokens} out={r.usage.output_tokens} tokens)")
    return text


def _chat_with_claude_cli(config: dict, messages: list[dict], temperature: float, max_tokens: int) -> str:
    model = config.get("claude_cli_model", "")
    logger.debug(f"LLM call → Claude CLI{' / ' + model if model else ''}")
    t0 = time.time()

    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_msgs = [m for m in messages if m["role"] != "system"]
    prompt_text = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in user_msgs)

    # --strict-mcp-config suppresses all MCP servers in the subprocess; reduces ~27k context tokens
    cmd = [
        "claude", "--print", "--output-format", "json", "--tools", "",
        "--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config",
        "--disable-slash-commands",
    ]
    if system:
        cmd += ["--system-prompt", system]
    if model:
        cmd += ["--model", model]

    try:
        result = subprocess.run(
            cmd,
            input=prompt_text,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "claude binary not found in PATH.\n"
            "Install Claude Code from https://claude.ai/code and run 'claude auth login'."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude CLI timed out after 300s.")

    if result.returncode != 0:
        raise RuntimeError(f"claude CLI exited {result.returncode}: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            text = data.get("result")
            if text is None:
                raise KeyError("no 'result' field in output")
        elif isinstance(data, list):
            result_event = next((e for e in data if isinstance(e, dict) and e.get("type") == "result"), None)
            if result_event is None:
                raise KeyError("no 'result' event found in output")
            text = result_event["result"]
        else:
            raise TypeError(f"unexpected output type: {type(data)}")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        raise RuntimeError(f"claude CLI unexpected output ({e}): {result.stdout[:200]}")

    elapsed = time.time() - t0
    logger.debug(f"LLM response: {len(text)} chars in {elapsed:.1f}s via claude CLI")
    return text


def chat_with_llm(
    config: dict,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    provider = config.get("llm_provider")
    if provider == "anthropic":
        return _chat_with_anthropic(config, messages, temperature, max_tokens)
    if provider == "claude_cli":
        return _chat_with_claude_cli(config, messages, temperature, max_tokens)
    return chat_with_fallback(_make_openrouter_client(config), config, messages, temperature, max_tokens)


def chat_with_fallback(
    llm: OpenAI,
    config: dict,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    primary = config.get("openrouter_model", "nvidia/nemotron-3-super-120b-a12b:free")
    fallbacks = config.get("openrouter_fallback_models", [])
    all_models = [primary] + [m for m in fallbacks if m != primary]

    available = [m for m in all_models if _is_model_available(m)]
    skipped = [m for m in all_models if not _is_model_available(m)]
    if skipped:
        logger.info(f"Skipping {len(skipped)} cooled-down model(s): {skipped}")
    if not available:
        raise RuntimeError("All LLM models are on quota cooldown — wait for cooldown to expire or run later.")

    for model_idx, model in enumerate(available):
        label = f"[{model_idx + 1}/{len(available)}] {model}"
        try:
            logger.debug(f"LLM call → {label}")
            t0 = time.time()
            resp = llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - t0
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            if usage:
                logger.debug(
                    f"LLM OK: {len(text)} chars in {elapsed:.1f}s "
                    f"(in={usage.prompt_tokens} out={usage.completion_tokens}) via {model}"
                )
            else:
                logger.debug(f"LLM OK: {len(text)} chars in {elapsed:.1f}s via {model}")
            return text
        except RateLimitError:
            _mark_model_exhausted(model)  # no retry — immediately switch to next model
        except NotFoundError:
            logger.warning(f"Model not found: {model} (404) — skipping")
        except APITimeoutError:
            logger.warning(f"Timeout on {model} — skipping")
        except APIStatusError as e:
            logger.error(f"API error on {model} (HTTP {e.status_code}) — skipping")
        except Exception as e:
            logger.error(f"LLM error ({model}): {e}")

    raise RuntimeError("All LLM models failed. Check your OpenRouter API key and quota.")
