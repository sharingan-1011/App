"""
llm.py — chat-model client factory.

Phase 3 integration: this file used to build a raw AsyncOpenAI client.
It now builds a langchain_openai.ChatOpenAI model instead, so agent.py's
LangGraph StateGraph can call model.bind_tools(...).ainvoke(...) directly.
Still points at the same OpenAI-compatible endpoint (llama-server via the
Cloudflare tunnel) — no proxy/gateway, no hosted-provider swap.

Retry/backoff (P1 hardening)
──────────────────────────────────────────────────────────────────────
ainvoke_with_retry() wraps a model.ainvoke(...) call with retry-on-
connection-failure, for the Cloudflare-tunnel-to-Kaggle path this app
depends on actually blipping sometimes. Deliberately narrow about WHAT
it retries: the model is built with streaming=True, so a chat-model
call that's already partway through streaming tokens to the UI (via
backend/agent.py's astream_events → on_chat_model_stream → SSE delta
events) has no way to "un-send" those tokens if it then fails and gets
retried — the retried call's output would visibly overlap/duplicate
with whatever already streamed. There's no token-level dedup anywhere
in this pipeline, and building one would mean buffering every delta
until a call fully succeeds, which changes the live-streaming UX this
app is built around. So this only retries exception types that occur
BEFORE any token can have streamed — connection failures (refused,
timed out, DNS, TLS handshake) — not generic/5xx-after-headers
failures, which are left to fail through as before. A connection drop
mid-stream is still possible and still NOT retried here, for the same
reason: by the time bytes were flowing, some UI-visible text may
already be out.
"""
import httpx
import openai
from langchain_openai import ChatOpenAI
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.config import BASE_URL, API_KEY, MODEL_NAME
from backend.logging_config import get_logger
import logging

logger = get_logger("llm")


def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0.6,
        max_tokens=4096,
        streaming=True,
    )


# Only errors that mean the request never actually reached (or got a
# response from) llama-server — connection refused, DNS/TLS handshake
# failure, connect-phase timeout. openai.APIConnectionError wraps
# httpx connect-phase failures; openai.APITimeoutError is included
# because a timeout with zero bytes received behaves the same as a
# refused connection here, and llama-server cold-starting under the
# Cloudflare tunnel is a known, real cause of exactly that. Deliberately
# NOT included: anything that implies a response (or partial response)
# was already in flight — see this module's docstring for why a
# mid-stream failure is never retried.
_RETRYABLE_EXCEPTIONS = (
    ConnectionRefusedError,
    ConnectionResetError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def ainvoke_with_retry(model: ChatOpenAI, messages, **kwargs):
    """
    Drop-in replacement for `await model.ainvoke(messages, **kwargs)`
    with connection-failure retry: up to 3 attempts, exponential
    backoff (1s, then 2s before the final attempt). `**kwargs` passes
    through unchanged — backend/tools/subagent.py calls this with a
    `config={"tags": ..., "metadata": ...}` kwarg for its per-specialist
    tracing tags, which this must not drop. Only retries the exception
    types in _RETRYABLE_EXCEPTIONS — anything else (including a
    mid-stream failure after tokens have already reached the UI, or a
    4xx/5xx after a real response started) propagates on the first
    failure, unretried. See this module's docstring for why.
    """
    return await model.ainvoke(messages, **kwargs)