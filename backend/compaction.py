"""
backend/compaction.py — folds old chat turns into a running, incremental
summary once a session's history exceeds a token budget, so a long
session degrades gracefully instead of growing linearly toward
llama-server's -c context limit and then hard-failing.

How "incremental" works here
──────────────────────────────────────────────────────────────────────
backend/db.py's session_summary table stores ONE digest per session
plus covered_through_ts — the timestamp of the last stored message
already folded into that digest. Each turn:

  1. Split this session's stored messages into "already covered"
     (ts <= covered_through_ts) and "raw" (everything after).
  2. If the raw portion's estimated size is still under budget, do
     nothing — return the existing summary (if any) + all of raw.
  3. If it's over budget AND there's more raw than
     CONTEXT_KEEP_RAW_TURNS: split raw into an older chunk and a kept
     tail of exactly CONTEXT_KEEP_RAW_TURNS messages. Ask the model to
     produce ONE updated digest from {existing digest + older chunk} —
     not a re-summarization of the entire session from scratch — and
     advance covered_through_ts to the older chunk's last message.

So the summarization call's input size is bounded by one compaction
cycle's worth of newly-old messages, not the session's total length,
however many times compaction has already run.

What this does NOT touch: backend/db.py's `messages` table itself.
Compaction only changes what gets assembled into the prompt sent to
the model for a given turn — the full, uncompacted history is still
exactly what's stored and exactly what the UI replays. If a human
wants to see the whole thing, it's still there.

Token budget: see backend/config.py's CONTEXT_TOKEN_BUDGET comment —
this uses a chars-per-token heuristic, not a real tokenizer, since
FORGE's model has no tiktoken-compatible encoder and tiktoken's own
encoders wouldn't be accurate for it either.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.config import CONTEXT_CHARS_PER_TOKEN, CONTEXT_KEEP_RAW_TURNS, CONTEXT_TOKEN_BUDGET
from backend.db import get_history, get_session_summary, set_session_summary
from backend.logging_config import get_logger

logger = get_logger("compaction")

_SUMMARIZE_PROMPT = """\
{prior}Below is the next chunk of this same conversation, in order (role: content pairs).

Produce ONE updated, self-contained summary covering everything relevant so far — \
the existing summary (if any) plus this new chunk. Capture: facts stated, decisions \
made, open questions or tasks still pending, and any tool/command results that matter \
for later turns. Be concise — this replaces the raw messages in the model's context, \
so it should be dense, not padded. Do not add commentary about the act of summarizing.

{transcript}"""


def _chars_to_tokens(n_chars: int) -> int:
    return max(1, n_chars // CONTEXT_CHARS_PER_TOKEN)


async def _summarize(get_model, session_id: str, previous_summary: str | None, chunk: list[dict]) -> str | None:
    """
    Returns the new digest, or None if the summarization call itself
    failed — callers must NOT advance covered_through_ts in that case,
    or the un-summarized chunk would be silently lost from the prompt
    forever (dropped from raw, never actually folded into the digest).
    """
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in chunk)
    prior = f"Existing summary of the conversation so far:\n{previous_summary}\n\n" if previous_summary else ""
    prompt = _SUMMARIZE_PROMPT.format(prior=prior, transcript=transcript)

    try:
        model = get_model()
        result = await model.ainvoke([HumanMessage(content=prompt)])
        text = result.content if isinstance(result.content, str) else str(result.content)
        text = text.strip()
        return text or None
    except Exception:
        logger.exception("session %s: compaction summarization call failed — skipping this cycle", session_id)
        return None


async def build_compacted_history(session_id: str, get_model) -> list[dict]:
    """
    Returns the message list (OpenAI-style role/content dicts) to
    actually send to the model for this turn: an optional leading
    system-role digest of older turns, followed by whatever's still
    raw. Re-fetches this session's stored messages directly (with
    timestamps) rather than trusting a history list a caller already
    has, since knowing what's already covered by the summary requires
    the timestamps backend/db.py's get_history_for_llm strips out.
    """
    rows = await get_history(session_id)  # ascending ts; includes 'ts'
    summary_row = await get_session_summary(session_id)

    if summary_row:
        summary_text: str | None = summary_row["summary_text"]
        covered_through = summary_row["covered_through_ts"]
        raw = [r for r in rows if r["ts"] > covered_through]
    else:
        summary_text = None
        raw = rows

    raw_chars = sum(len(r["content"]) for r in raw)
    budget_chars = CONTEXT_TOKEN_BUDGET * CONTEXT_CHARS_PER_TOKEN

    if raw_chars > budget_chars and len(raw) > CONTEXT_KEEP_RAW_TURNS:
        older_chunk = raw[:-CONTEXT_KEEP_RAW_TURNS]
        kept_tail = raw[-CONTEXT_KEEP_RAW_TURNS:]

        new_summary = await _summarize(get_model, session_id, summary_text, older_chunk)
        if new_summary is not None:
            new_covered_through = older_chunk[-1]["ts"]
            await set_session_summary(session_id, new_summary, new_covered_through)
            logger.info(
                "session %s: compacted %d message(s) into the running summary (raw was ~%d est. tokens, budget %d)",
                session_id, len(older_chunk), _chars_to_tokens(raw_chars), CONTEXT_TOKEN_BUDGET,
            )
            summary_text = new_summary
            raw = kept_tail
        # else: summarization failed — fall through and send `raw`
        # (still everything, over budget for this one turn) rather than
        # silently dropping the older_chunk with nothing to show for it.

    out: list[dict] = []
    if summary_text:
        out.append({
            "role": "system",
            "content": f"Summary of earlier conversation (older messages condensed to save context):\n{summary_text}",
        })
    out.extend({"role": r["role"], "content": r["content"]} for r in raw)
    return out