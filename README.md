# 📧 Email Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/huggingface/openenv-course)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Meta PyTorch OpenEnv Hackathon × Scaler SST Submission**

---

## 🎯 Environment Description

**Email Triage** is a real-world AI training environment where an agent must process incoming business emails and make accurate triage decisions across three dimensions:

| Decision | Options |
|----------|---------|
| **Priority** | `urgent` · `high` · `medium` · `low` |
| **Category** | `billing` · `technical_support` · `account` · `sales` · `spam` · `general` |
| **Route** | `billing_team` · `tech_support_team` · `account_team` · `sales_team` · `trash` · `general_queue` |

Email triage is performed by humans every single day in customer support, operations, and executive assistance roles. It requires **reading comprehension**, **contextual reasoning**, **urgency detection**, and **business domain knowledge** — making it an ideal real-world challenge for training and evaluating AI agents.

---

## 🧠 Motivation

Email overload is a [$650B/year productivity problem](https://hbr.org). Automated triage reduces response time, prevents SLA breaches, and routes issues to the right team — but it requires nuanced understanding that goes beyond keyword matching.

This environment teaches agents to:
- Distinguish urgency signals (explicit vs. implicit)
- Handle multi-issue emails and pick the primary blocker
- Avoid routing errors that waste team resources
- Identify spam, including "legitimate-looking" newsletters

---

## 📐 Action & Observation Spaces

### Observation

```python
class Observation(BaseModel):
    email_id: str           # Unique ID
    subject: str            # Email subject line
    sender: str             # Sender email address
    body: str               # Full email body
    timestamp: str          # ISO 8601 datetime
    task_id: str            # "easy" | "medium" | "hard"
    task_description: str   # Human-readable task guidance
    available_priorities: list[str]   # Valid priority options
    available_categories: list[str]   # Valid category options
    available_routes: list[str]       # Valid routing options
```

### Action

```python
class Action(BaseModel):
    priority: str   # "urgent" | "high" | "medium" | "low"
    category: str   # "billing" | "technical_support" | "account" | "sales" | "spam" | "general"
    route: str      # "billing_team" | "tech_support_team" | "account_team" | "sales_team" | "trash" | "general_queue"
    summary: str    # One sentence, max 20 words
```

### Reward

```python
class Reward(BaseModel):
    value: float              # 0.0 – 1.0 total score
    breakdown: dict[str, float]  # Per-field scores
    feedback: str             # Human-readable explanation
```

---

## 🎮 Tasks

### Task 1 — Easy (Baseline: ~0.95)
Emails with clear, unambiguous triage signals. A production server down email, an obvious spam message, and a straightforward billing inquiry.

### Task 2 — Medium (Baseline: ~0.82)
Emails requiring inference. A customer disputing a renewal charge (billing vs. account?), a feature request, and a partnership outreach with high business value.

### Task 3 — Hard (Baseline: ~0.71)
Complex, multi-issue emails. A user with both billing AND access issues but an urgent demo in 2 hours. A legal contract email with a looming deadline. A legitimate newsletter that still belongs in trash.

---

## 🔌 OpenEnv API

```python
from environment import EmailTriageEnv, Action

# Initialize
env = EmailTriageEnv(task_id="easy", seed=42)

# Reset — get first observation
obs = env.reset()
print(obs.subject, obs.body)

# Act
action = Action(
    priority="urgent",
    category="technical_support",
    route="tech_support_team",
    summary="Production server is down causing customer outage."
)

# Step
obs, reward, done, info = env.step(action)
print(f"Score: {reward.value:.4f}")
print(f"Feedback: {reward.feedback}")
print(f"Breakdown: {reward.breakdown}")

# State
print(env.state())
```

---

## 🏆 Reward Function

The reward function provides **partial credit** across 4 dimensions:

| Dimension | Max Weight | Partial Credit |
|-----------|-----------|----------------|
| Priority | 0.40 | 0.20 for ±1 level off (e.g., `high` instead of `urgent`) |
| Category | 0.35 | Exact match only |
| Route | 0.20 | Exact match only |
| Summary | 0.05 | Full credit for 5–20 word summaries |

**Key design decisions:**
- Priority gets the highest weight because urgency misjudgment has the highest real-world cost (e.g., missing a SLA breach)
- Partial credit for priority encourages learning the ordinal scale, not just binary correct/wrong
- Route is derived from category but scored separately — an agent that picks the right category but wrong route is penalized less than one that picks the wrong category
- Reward is continuous over the full trajectory (not just end-of-episode), allowing RL algorithms to learn from every step

---

## 📊 Baseline Scores

Run the baseline yourself:

```bash
export OPENAI_API_KEY=sk-...
python baseline.py --model gpt-4o-mini
```

| Task | Model | Avg Score | Emails |
|------|-------|-----------|--------|
| Easy | gpt-4o-mini | 0.9500 | 3 |
| Medium | gpt-4o-mini | 0.8167 | 3 |
| Hard | gpt-4o-mini | 0.7167 | 3 |
| **Overall** | gpt-4o-mini | **0.8278** | **9** |

---

## 🚀 Setup & Usage

### Local

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env

pip install -r requirements.txt

# Run baseline
export OPENAI_API_KEY=sk-...
python baseline.py

# Launch interactive demo
python app.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env
```

### OpenEnv Validation

```bash
pip install openenv
openenv validate .
```

---

## 📁 File Structure

```
email-triage-env/
├── environment.py      # Core OpenEnv environment (Observation, Action, Reward, step/reset/state)
├── baseline.py         # Baseline inference script using OpenAI API
├── app.py              # Gradio demo for Hugging Face Spaces
├── openenv.yaml        # OpenEnv metadata and spec
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

*Built for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, 2026.*
