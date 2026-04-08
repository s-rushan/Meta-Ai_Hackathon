"""
Email Triage OpenEnv Environment
Real-world task: An AI agent must triage incoming emails by
correctly assigning priority, category, and routing destination.
"""

from __future__ import annotations
import random
from typing import Any
from pydantic import BaseModel, Field


# ─────────────────────────── Pydantic Models ────────────────────────────── #

class Observation(BaseModel):
    email_id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    task_id: str
    task_description: str
    available_priorities: list[str] = ["urgent", "high", "medium", "low"]
    available_categories: list[str] = [
        "billing", "technical_support", "account", "sales", "spam", "general"
    ]
    available_routes: list[str] = [
        "billing_team", "tech_support_team", "account_team",
        "sales_team", "trash", "general_queue"
    ]


class Action(BaseModel):
    priority: str = Field(..., description="One of: urgent, high, medium, low")
    category: str = Field(..., description="One of: billing, technical_support, account, sales, spam, general")
    route: str = Field(..., description="One of: billing_team, tech_support_team, account_team, sales_team, trash, general_queue")
    summary: str = Field(..., description="One-sentence summary of the email (max 20 words)")


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float]
    feedback: str


# ──────────────────────────── Email Dataset ──────────────────────────────── #

EMAILS = {
    "easy": [
        {
            "email_id": "E001",
            "subject": "URGENT: Server is completely down - production affected",
            "sender": "cto@clientcorp.com",
            "body": (
                "Hi Support,\n\nOur production server has been down for 30 minutes. "
                "ALL our customers are affected. We are losing $10,000/minute. "
                "This is a critical emergency. Please respond IMMEDIATELY.\n\nCTO, ClientCorp"
            ),
            "timestamp": "2024-01-15T09:02:00Z",
            "label": {"priority": "urgent", "category": "technical_support", "route": "tech_support_team"},
        },
        {
            "email_id": "E002",
            "subject": "Get rich quick - make $5000 from home!!!",
            "sender": "noreply@spam123.biz",
            "body": (
                "Congratulations! You have been selected to earn $5000 per week from home. "
                "Click the link below NOW! Limited time offer. Act fast!!!\n"
                "http://totally-legit-money.xyz/click-here"
            ),
            "timestamp": "2024-01-15T09:05:00Z",
            "label": {"priority": "low", "category": "spam", "route": "trash"},
        },
        {
            "email_id": "E003",
            "subject": "Invoice #4821 payment confirmation request",
            "sender": "accounts@vendor.com",
            "body": (
                "Dear Finance Team,\n\nWe would like to confirm payment status for Invoice #4821 "
                "dated December 30th for $2,450. The payment was due on January 10th. "
                "Please advise on the payment status.\n\nBest,\nAccounts Receivable, Vendor Inc."
            ),
            "timestamp": "2024-01-15T09:10:00Z",
            "label": {"priority": "high", "category": "billing", "route": "billing_team"},
        },
    ],
    "medium": [
        {
            "email_id": "M001",
            "subject": "Question about my subscription renewal",
            "sender": "john.smith@gmail.com",
            "body": (
                "Hello,\n\nI noticed my credit card was charged $99 yesterday for an annual "
                "subscription renewal. I thought I had cancelled my account last month. "
                "Could you check my account (john.smith@gmail.com) and process a refund if "
                "the cancellation went through? I'd also like to know why I wasn't notified "
                "before the charge.\n\nThanks,\nJohn"
            ),
            "timestamp": "2024-01-15T10:15:00Z",
            "label": {"priority": "high", "category": "billing", "route": "billing_team"},
        },
        {
            "email_id": "M002",
            "subject": "Feature request: dark mode for mobile app",
            "sender": "sarah.jones@company.org",
            "body": (
                "Hi Product Team,\n\nI've been using your mobile app for 6 months and love it! "
                "One feature I really miss is dark mode — it would be great for late-night use "
                "and battery saving. I know other users have asked for this too based on the "
                "community forum. Is this on your roadmap?\n\nBest,\nSarah"
            ),
            "timestamp": "2024-01-15T10:30:00Z",
            "label": {"priority": "medium", "category": "general", "route": "general_queue"},
        },
        {
            "email_id": "M003",
            "subject": "Partnership opportunity - B2B integration",
            "sender": "partnerships@techstartup.io",
            "body": (
                "Hello,\n\nI'm the VP of Partnerships at TechStartup. We have 50,000 enterprise "
                "customers who could benefit from your platform. We'd like to explore a formal "
                "integration and reseller partnership. Our average deal size is $50K ARR. "
                "Are you open to a call this week?\n\nRegards,\nAlex Chen, VP Partnerships"
            ),
            "timestamp": "2024-01-15T10:45:00Z",
            "label": {"priority": "high", "category": "sales", "route": "sales_team"},
        },
    ],
    "hard": [
        {
            "email_id": "H001",
            "subject": "Re: Re: Re: Follow up on our conversation",
            "sender": "mike@unknown-domain.net",
            "body": (
                "Hey,\n\nAs per our chat last week I'm still waiting. My account #78234 shows "
                "the wrong plan but billing keeps charging me the old rate. I spoke to Lisa on "
                "Tuesday who said she'd escalate but nothing happened. Also my team members "
                "can't log in since the migration. Can someone PLEASE sort this out - I have "
                "a board meeting in 2 hours where I need to demo the product.\n\nMike"
            ),
            "timestamp": "2024-01-15T11:00:00Z",
            "label": {"priority": "urgent", "category": "technical_support", "route": "tech_support_team"},
            # Rationale: despite mentioning billing, the primary blocker is login/access (tech)
            # and the board meeting makes it urgent. A sophisticated agent must weigh multi-issue emails.
        },
        {
            "email_id": "H002",
            "subject": "Regarding our enterprise contract terms",
            "sender": "legal@bigclient.com",
            "body": (
                "Dear Team,\n\nOur legal department is reviewing the enterprise agreement "
                "renewal (Contract #ENT-2024-089). We have concerns about clause 7.3 regarding "
                "data residency and GDPR compliance. We will need written clarification before "
                "signing. Note that our current contract expires on January 31st — just 16 days "
                "away. Please route to the appropriate team.\n\nRegards,\nLegal Counsel, BigClient"
            ),
            "timestamp": "2024-01-15T11:15:00Z",
            "label": {"priority": "urgent", "category": "account", "route": "account_team"},
        },
        {
            "email_id": "H003",
            "subject": "Newsletter: Industry trends in Q4",
            "sender": "newsletter@industrymag.com",
            "body": (
                "Hello Subscriber,\n\nIn this month's issue: AI adoption rates surge 40% in "
                "enterprise — what it means for your business. Plus: Our annual survey results "
                "on remote work productivity. Click to read the full report. To unsubscribe "
                "reply STOP.\n\nIndustry Magazine"
            ),
            "timestamp": "2024-01-15T11:30:00Z",
            "label": {"priority": "low", "category": "spam", "route": "trash"},
            # Hard because it's a legitimate newsletter (not malicious spam) but still trash/low
        },
    ],
}

TASK_DESCRIPTIONS = {
    "easy": (
        "Triage this email by assigning the correct priority level, category, and routing "
        "destination. This email has clear, unambiguous signals."
    ),
    "medium": (
        "Triage this email carefully. The signals may require reading between the lines — "
        "consider sender context, urgency, and business impact."
    ),
    "hard": (
        "Triage this complex email. It may contain multiple issues, ambiguous signals, or "
        "require nuanced judgment about priority vs. category. Reason carefully."
    ),
}


# ──────────────────────────── Grader Logic ───────────────────────────────── #

PRIORITY_ORDER = {"urgent": 3, "high": 2, "medium": 1, "low": 0}

def grade_action(action: Action, label: dict) -> Reward:
    scores = {}

    # Priority (0.0–0.4): partial credit for being 1 level off
    p_pred = PRIORITY_ORDER.get(action.priority, -1)
    p_true = PRIORITY_ORDER.get(label["priority"], -1)
    diff = abs(p_pred - p_true)
    if diff == 0:
        scores["priority"] = 0.40
    elif diff == 1:
        scores["priority"] = 0.20
    else:
        scores["priority"] = 0.0

    # Category (0.0–0.35): exact match only
    scores["category"] = 0.35 if action.category == label["category"] else 0.0

    # Route (0.0–0.20): exact match only (derived from category)
    scores["route"] = 0.20 if action.route == label["route"] else 0.0

    # Summary quality (0.0–0.05): length-based heuristic
    word_count = len(action.summary.split())
    if 5 <= word_count <= 20:
        scores["summary"] = 0.05
    elif word_count < 5:
        scores["summary"] = 0.02
    else:
        scores["summary"] = 0.01

    total = sum(scores.values())

    if total >= 0.9:
        feedback = "Excellent triage! All fields correct."
    elif total >= 0.6:
        feedback = f"Good attempt. Errors in: {[k for k,v in scores.items() if v < 0.15 and k != 'summary']}."
    elif total >= 0.3:
        feedback = "Partial credit. Review priority and category signals more carefully."
    else:
        feedback = "Incorrect triage. Re-read the email for urgency cues and sender context."

    return Reward(value=round(total, 4), breakdown=scores, feedback=feedback)


# ─────────────────────────── Main Environment ────────────────────────────── #

class EmailTriageEnv:
    """
    OpenEnv-compatible Email Triage Environment.

    An AI agent reads business emails and must assign:
    - Priority (urgent / high / medium / low)
    - Category (billing / technical_support / account / sales / spam / general)
    - Route (billing_team / tech_support_team / account_team / sales_team / trash / general_queue)
    - Summary (1-sentence, max 20 words)

    Three task difficulty levels: easy → medium → hard.
    """

    VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
    VALID_CATEGORIES = {"billing", "technical_support", "account", "sales", "spam", "general"}
    VALID_ROUTES = {"billing_team", "tech_support_team", "account_team", "sales_team", "trash", "general_queue"}

    def __init__(self, task_id: str = "easy", seed: int | None = None):
        assert task_id in ("easy", "medium", "hard"), f"task_id must be easy/medium/hard, got {task_id}"
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._current_email: dict | None = None
        self._done = False
        self._step_count = 0

    def reset(self) -> Observation:
        """Reset environment and return first observation."""
        self._done = False
        self._step_count = 0
        pool = EMAILS[self.task_id]
        self._current_email = self._rng.choice(pool)
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Process one agent action.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._validate_action(action)
        label = self._current_email["label"]
        reward = grade_action(action, label)

        self._done = True
        self._step_count += 1

        info = {
            "step": self._step_count,
            "email_id": self._current_email["email_id"],
            "correct_label": label,
            "agent_action": action.model_dump(),
        }

        return self._make_observation(), reward, self._done, info

    def state(self) -> dict[str, Any]:
        """Return current environment state."""
        return {
            "task_id": self.task_id,
            "done": self._done,
            "step_count": self._step_count,
            "current_email_id": self._current_email["email_id"] if self._current_email else None,
        }

    # ── Helpers ─────────────────────────────────────────────────────────── #

    def _make_observation(self) -> Observation:
        e = self._current_email
        return Observation(
            email_id=e["email_id"],
            subject=e["subject"],
            sender=e["sender"],
            body=e["body"],
            timestamp=e["timestamp"],
            task_id=self.task_id,
            task_description=TASK_DESCRIPTIONS[self.task_id],
        )

    def _validate_action(self, action: Action) -> None:
        errors = []
        if action.priority not in self.VALID_PRIORITIES:
            errors.append(f"Invalid priority '{action.priority}'")
        if action.category not in self.VALID_CATEGORIES:
            errors.append(f"Invalid category '{action.category}'")
        if action.route not in self.VALID_ROUTES:
            errors.append(f"Invalid route '{action.route}'")
        if errors:
            raise ValueError(f"Action validation failed: {'; '.join(errors)}")
