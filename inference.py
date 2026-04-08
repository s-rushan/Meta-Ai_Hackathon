"""
inference.py — Email Triage OpenEnv
====================================
MANDATORY FORMAT — emits [START], [STEP], [END] to stdout exactly as specified.

Environment variables:
    API_BASE_URL    The API endpoint for the LLM  (default: HF router)
    MODEL_NAME      The model identifier           (default: Qwen2.5-72B-Instruct)
    HF_TOKEN        Your Hugging Face / API key
    EMAIL_TASK      Task level: easy | medium | hard (default: easy)
"""

import os
import json
import asyncio
import textwrap
from typing import List, Optional

from openai import OpenAI

from environment import EmailTriageEnv, Action, EMAILS

# ── Mandatory env vars ────────────────────────────────────────────────────── #
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME     = os.getenv("EMAIL_TASK", "easy")
BENCHMARK     = "email-triage-env"
MAX_STEPS     = 1   # Each email episode is exactly 1 step
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Stdout logging (mandatory format) ────────────────────────────────────── #

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompts ───────────────────────────────────────────────────────────────── #

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant for a B2B SaaS company.
    Your job is to analyze each incoming email and classify it with precision.

    You must respond with ONLY a valid JSON object — no markdown, no preamble, no explanation.

    JSON schema (respond with exactly this structure):
    {
      "priority": "<urgent|high|medium|low>",
      "category": "<billing|technical_support|account|sales|spam|general>",
      "route": "<billing_team|tech_support_team|account_team|sales_team|trash|general_queue>",
      "summary": "<one sentence, max 20 words>"
    }

    Priority rules:
    - urgent: Production down, legal deadline <24h, active revenue loss, demo in <4h
    - high:   Financial dispute, important client, time-sensitive but not crisis
    - medium: Feature request, general inquiry, non-urgent support
    - low:    Newsletters, promotions, unsubscribe, spam

    Category → correct route mapping:
    - billing           → billing_team
    - technical_support → tech_support_team
    - account           → account_team
    - sales             → sales_team
    - spam              → trash
    - general           → general_queue

    For multi-issue emails: identify the PRIMARY blocker and use that category.
    A board demo in 2 hours + login failure = technical_support (urgent).
    A contract expiring in 16 days + legal questions = account (urgent).
""").strip()


def build_user_prompt(obs) -> str:
    return textwrap.dedent(f"""
        Triage this email carefully.

        TASK LEVEL: {obs.task_id}
        TASK NOTE: {obs.task_description}

        ---
        Email ID:  {obs.email_id}
        From:      {obs.sender}
        Subject:   {obs.subject}
        Timestamp: {obs.timestamp}

        {obs.body}
        ---

        Available priorities: {obs.available_priorities}
        Available categories: {obs.available_categories}
        Available routes:     {obs.available_routes}

        Respond with JSON only.
    """).strip()


# ── Agent call ────────────────────────────────────────────────────────────── #

def get_action(client: OpenAI, obs) -> tuple[Action, str]:
    """Call LLM and parse response into an Action. Returns (action, raw_action_str)."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model ignores instructions
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        action = Action(**data)
        action_str = f"priority={action.priority},category={action.category},route={action.route}"
        return action, action_str
    except Exception as exc:
        # Fallback safe action
        fallback = Action(
            priority="medium",
            category="general",
            route="general_queue",
            summary="Unable to parse email triage response."
        )
        return fallback, f"fallback error={exc}"


# ── Episode runner ────────────────────────────────────────────────────────── #

async def run_episode(client: OpenAI, task_id: str, email_data: dict) -> float:
    """Run one episode for a single email. Returns reward."""
    env = EmailTriageEnv(task_id=task_id, seed=42)
    env._current_email = email_data

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        obs = env.reset()
        # Override to our specific email
        env._current_email = email_data
        obs = env._make_observation()

        for step in range(1, MAX_STEPS + 1):
            action, action_str = get_action(client, obs)

            try:
                obs, reward_obj, done, info = env.step(action)
                reward = reward_obj.value
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main: run all 3 tasks ─────────────────────────────────────────────────── #

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}

    for task_id in ["easy", "medium", "hard"]:
        pool = EMAILS[task_id]
        task_scores = []

        for email_data in pool:
            score = await run_episode(client, task_id, email_data)
            task_scores.append(score)

        all_scores[task_id] = task_scores

    # Final summary to stderr (doesn't interfere with mandatory stdout format)
    import sys
    import statistics
    print("\n" + "="*50, file=sys.stderr)
    print("EVALUATION SUMMARY", file=sys.stderr)
    print("="*50, file=sys.stderr)
    grand = []
    for tid, scores in all_scores.items():
        avg = statistics.mean(scores)
        grand.extend(scores)
        print(f"  {tid.upper():8s} → avg={avg:.4f}  {[round(s,4) for s in scores]}", file=sys.stderr)
    print(f"\n  OVERALL  → avg={statistics.mean(grand):.4f}", file=sys.stderr)
    print("="*50, file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
