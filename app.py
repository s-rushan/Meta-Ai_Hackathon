"""
app.py — Email Triage OpenEnv — Hugging Face Spaces entry point.

HTTP API (for OpenEnv automated validator):
  GET  /         → 200 health check
  POST /reset    → 200 Observation JSON
  POST /step     → 200 {observation, reward, done, info}
  POST /state    → 200 state dict

Gradio UI served at /ui
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import gradio as gr

from environment import EmailTriageEnv, Action, EMAILS

# ── FastAPI ────────────────────────────────────────────────────────────────── #

api = FastAPI(title="Email Triage OpenEnv")
_env = EmailTriageEnv(task_id="easy", seed=42)
_env.reset()


@api.get("/")
async def health():
    return JSONResponse({"status": "ok", "env": "email-triage-env", "version": "1.0.0"})


@api.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    global _env
    task_id = body.get("task_id", "easy")
    seed = body.get("seed", 42)
    _env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = _env.reset()
    return JSONResponse(obs.model_dump())


@api.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
        action = Action(**body)
        obs, reward, done, info = _env.step(action)
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": {k: v for k, v in info.items() if k != "agent_action"},
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@api.post("/state")
async def state():
    return JSONResponse(_env.state())


# ── Gradio helpers ─────────────────────────────────────────────────────────── #

def load_email(task_id, email_idx):
    pool = EMAILS[task_id]
    email = pool[int(email_idx)]
    return (
        f"**From:** {email['sender']}\n\n"
        f"**Subject:** {email['subject']}\n\n"
        f"**Date:** {email['timestamp']}\n\n"
        f"---\n\n{email['body']}"
    )


def run_manual_triage(task_id, email_idx, priority, category, route, summary):
    pool = EMAILS[task_id]
    email = pool[int(email_idx)]
    env = EmailTriageEnv(task_id=task_id)
    env._current_email = email
    env._done = False
    if not priority or not category or not route:
        return load_email(task_id, email_idx), "Please select priority, category, and route."
    try:
        action = Action(priority=priority, category=category, route=route,
                        summary=summary or "No summary provided.")
        _, reward, _, info = env.step(action)
    except ValueError as e:
        return load_email(task_id, email_idx), f"Invalid action: {e}"
    correct = info["correct_label"]
    bars = {"priority":(reward.breakdown["priority"],0.40),
            "category":(reward.breakdown["category"],0.35),
            "route":(reward.breakdown["route"],0.20),
            "summary":(reward.breakdown["summary"],0.05)}
    bar_lines = "\n".join(
        f"- **{k.title()}**: {v:.2f}/{mx:.2f}  {'✅' if v==mx else '❌'}"
        for k,(v,mx) in bars.items())
    result = (
        f"## Score: **{reward.value:.4f} / 1.00**\n\n_{reward.feedback}_\n\n"
        f"### Breakdown\n{bar_lines}\n\n"
        f"### Correct Answer\n"
        f"- Priority: `{correct['priority']}`\n"
        f"- Category: `{correct['category']}`\n"
        f"- Route: `{correct['route']}`"
    )
    return load_email(task_id, email_idx), result


def run_llm_agent(api_key, api_base, model, task_id):
    if not api_key.strip():
        return "Please enter your HF_TOKEN or API key."
    try:
        import statistics
        from openai import OpenAI
        from inference import get_action
        client = OpenAI(base_url=api_base.strip() or "https://router.huggingface.co/v1",
                        api_key=api_key.strip())
        pool = EMAILS[task_id]
        results, scores = [], []
        for email_data in pool:
            env = EmailTriageEnv(task_id=task_id)
            env._current_email = email_data
            obs = env._make_observation()
            action, _ = get_action(client, obs)
            _, reward, _, info = env.step(action)
            scores.append(reward.value)
            correct = info["correct_label"]
            icon = "✅" if reward.value>=0.9 else ("⚠️" if reward.value>=0.5 else "❌")
            results.append(
                f"{icon} **{email_data['email_id']}** — _{email_data['subject'][:55]}_\n"
                f"   Agent:   `{action.priority}` / `{action.category}` / `{action.route}`\n"
                f"   Correct: `{correct['priority']}` / `{correct['category']}` / `{correct['route']}`\n"
                f"   **Score: {reward.value:.4f}** — {reward.feedback}\n"
            )
        avg = statistics.mean(scores)
        bar = "█"*int(avg*20) + "░"*(20-int(avg*20))
        return (f"## Agent Results — Task: `{task_id.upper()}`\n\n" +
                "\n".join(results) +
                f"\n---\n**Average Score: {avg:.4f}**  `|{bar}|`")
    except Exception as e:
        return f"Error: {e}"


# ── Gradio UI ──────────────────────────────────────────────────────────────── #

with gr.Blocks(title="Email Triage OpenEnv", theme=gr.themes.Soft()) as gradio_ui:
    gr.Markdown("""
    # 📧 Email Triage OpenEnv
    **Meta PyTorch OpenEnv Hackathon × Scaler SST**

    Real-world email triage environment. Agent assigns **priority**, **category**, and **route** to business emails.

    > API: `POST /reset` · `POST /step` · `POST /state`
    """)
    with gr.Tabs():
        with gr.Tab("🧪 Try It Yourself"):
            with gr.Row():
                task_dd  = gr.Dropdown(["easy","medium","hard"], value="easy", label="Task Level")
                email_dd = gr.Dropdown(["0","1","2"], value="0", label="Email Index")
                load_btn = gr.Button("📨 Load Email", variant="secondary")
            email_md = gr.Markdown("_Click Load Email to begin._")
            load_btn.click(load_email, inputs=[task_dd, email_dd], outputs=email_md)
            gr.Markdown("### Your Triage Decision")
            with gr.Row():
                priority_dd = gr.Dropdown(["urgent","high","medium","low"], label="Priority")
                category_dd = gr.Dropdown(["billing","technical_support","account","sales","spam","general"], label="Category")
                route_dd    = gr.Dropdown(["billing_team","tech_support_team","account_team","sales_team","trash","general_queue"], label="Route")
            summary_box = gr.Textbox(label="Summary (max 20 words)")
            submit_btn  = gr.Button("✅ Submit Triage", variant="primary")
            result_md   = gr.Markdown()
            submit_btn.click(run_manual_triage,
                inputs=[task_dd, email_dd, priority_dd, category_dd, route_dd, summary_box],
                outputs=[email_md, result_md])

        with gr.Tab("🤖 Run LLM Agent"):
            with gr.Row():
                api_key_box  = gr.Textbox(label="HF_TOKEN / API Key", type="password", placeholder="hf_...")
                api_base_box = gr.Textbox(label="API_BASE_URL", value="https://router.huggingface.co/v1")
                model_box    = gr.Textbox(label="MODEL_NAME", value="Qwen/Qwen2.5-72B-Instruct")
            task_agent_dd = gr.Dropdown(["easy","medium","hard"], value="easy", label="Task Level")
            run_btn  = gr.Button("🚀 Run Agent", variant="primary")
            agent_md = gr.Markdown()
            run_btn.click(run_llm_agent,
                inputs=[api_key_box, api_base_box, model_box, task_agent_dd],
                outputs=agent_md)

        with gr.Tab("📋 API & Spec"):
            gr.Markdown("""
            ## HTTP Endpoints
            | Method | Path | Description |
            |--------|------|-------------|
            | GET | `/` | Health check → 200 |
            | POST | `/reset` | Reset env → Observation JSON |
            | POST | `/step` | Action → Observation + Reward + done |
            | POST | `/state` | Current state dict |

            ## Reward Weights
            | Field | Weight | Partial Credit |
            |-------|--------|----------------|
            | Priority | 0.40 | 0.20 for ±1 off |
            | Category | 0.35 | Exact only |
            | Route | 0.20 | Exact only |
            | Summary | 0.05 | 5–20 words |
            """)


app = gr.mount_gradio_app(api, gradio_ui, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
