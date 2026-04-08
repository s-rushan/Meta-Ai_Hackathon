"""
Baseline Inference Script — Email Triage OpenEnv
Runs an LLM agent against all 3 task difficulty levels and reports scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

    # Or with a different model:
    python baseline.py --model gpt-4o-mini
"""

import os
import json
import argparse
import statistics
from openai import OpenAI

from environment import EmailTriageEnv, Action, EMAILS

SYSTEM_PROMPT = """You are an expert email triage assistant for a B2B SaaS company.

Your job is to analyze incoming emails and classify them accurately.

You must respond with ONLY a valid JSON object — no markdown, no explanation, no preamble.

JSON schema:
{
  "priority": "<urgent|high|medium|low>",
  "category": "<billing|technical_support|account|sales|spam|general>",
  "route": "<billing_team|tech_support_team|account_team|sales_team|trash|general_queue>",
  "summary": "<one sentence, max 20 words>"
}

Priority rules:
- urgent: System down, legal deadline, active revenue loss, or demo in <4 hours
- high: Financial issue, important client, time-sensitive but not crisis
- medium: Feature request, general inquiry, non-urgent support
- low: Newsletters, promotions, spam, generic FYIs

Category → Route mapping:
- billing → billing_team
- technical_support → tech_support_team
- account → account_team
- sales → sales_team
- spam → trash
- general → general_queue

Think carefully about multi-issue emails — route to the team that resolves the PRIMARY blocker."""

USER_TEMPLATE = """Triage this email:

TASK LEVEL: {task_id}
TASK NOTE: {task_description}

---
Email ID: {email_id}
From: {sender}
Subject: {subject}
Date: {timestamp}

{body}
---

Respond with JSON only."""


def run_agent(client: OpenAI, obs, model: str) -> Action:
    prompt = USER_TEMPLATE.format(
        task_id=obs.task_id,
        task_description=obs.task_description,
        email_id=obs.email_id,
        sender=obs.sender,
        subject=obs.subject,
        timestamp=obs.timestamp,
        body=obs.body,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return Action(**data)


def evaluate_task(client: OpenAI, task_id: str, model: str, verbose: bool = True) -> list[float]:
    scores = []
    pool = EMAILS[task_id]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id.upper()}  ({len(pool)} emails)")
        print(f"{'='*60}")

    for email_data in pool:
        # Create env seeded on email index so each email is shown exactly once
        env = EmailTriageEnv(task_id=task_id, seed=hash(email_data["email_id"]))
        obs = env.reset()

        # Force the specific email for reproducibility
        env._current_email = email_data
        obs = env._make_observation()

        try:
            action = run_agent(client, obs, model)
            _, reward, _, info = env.step(action)

            scores.append(reward.value)

            if verbose:
                print(f"\n  📧 [{email_data['email_id']}] {email_data['subject'][:55]}...")
                print(f"     Agent  → priority={action.priority}, category={action.category}, route={action.route}")
                correct = info["correct_label"]
                print(f"     Correct→ priority={correct['priority']}, category={correct['category']}, route={correct['route']}")
                print(f"     Score  → {reward.value:.4f}  |  {reward.feedback}")
                print(f"     Breakdown: {reward.breakdown}")

        except Exception as e:
            print(f"  ❌ Error on {email_data['email_id']}: {e}")
            scores.append(0.0)

    return scores


def main():
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv Baseline")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-email output")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    verbose = not args.quiet

    print(f"\n🚀 Email Triage OpenEnv — Baseline Evaluation")
    print(f"   Model : {args.model}")
    print(f"   Tasks : easy, medium, hard")

    all_scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores = evaluate_task(client, task_id, args.model, verbose=verbose)
        all_scores[task_id] = scores

    # ── Summary Report ── #
    print(f"\n{'='*60}")
    print(f"  BASELINE SCORE REPORT")
    print(f"{'='*60}")
    grand_scores = []
    for task_id, scores in all_scores.items():
        avg = statistics.mean(scores) if scores else 0.0
        grand_scores.extend(scores)
        print(f"  {task_id.upper():8s} → avg={avg:.4f}  scores={[round(s,4) for s in scores]}")

    overall = statistics.mean(grand_scores) if grand_scores else 0.0
    print(f"\n  OVERALL  → avg={overall:.4f}  ({len(grand_scores)} emails evaluated)")
    print(f"{'='*60}\n")

    return overall


if __name__ == "__main__":
    main()
