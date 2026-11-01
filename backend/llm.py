"""
llm.py — chat-model client factory.

Phase 3 integration: this file used to build a raw AsyncOpenAI client.
It now builds a langchain_openai.ChatOpenAI model instead, so agent.py's
LangGraph StateGraph can call model.bind_tools(...).ainvoke(...) directly.
Still points at the same OpenAI-compatible endpoint (llama-server via the
Cloudflare tunnel) — no proxy/gateway, no hosted-provider swap.
"""
from langchain_openai import ChatOpenAI
from backend.config import BASE_URL, API_KEY, MODEL_NAME


def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0.6,
        max_tokens=4096,
        streaming=True,
    )
