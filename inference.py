import os
import json
from openai import OpenAI
from env.renovation_env import RenovationEnv

# =========================
# ENV VARIABLES
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# =========================
# TASKS
# =========================
TASKS = {
    "easy": {"budget": 30000, "style": "modern"},
    "medium": {"budget": 40000, "style": "classic"},
    "hard": {"budget": 35000, "style": "modern"}
}

# =========================
# LLM ACTION
# =========================
def get_action(state):
    prompt = f"""
You are an AI interior design planner.

State:
{json.dumps(state)}

Available items:
["chair", "table", "paint", "light"]

Choose the best item.

Return ONLY JSON:
{{"action": "<item>"}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        # Safe JSON parsing
        if content.startswith("```"):
            content = content.strip("```json").strip("```")

        return json.loads(content)["action"]

    except Exception:
        return "chair"  # fallback

# =========================
# RUN TASK
# =========================
def run_task(task_name):
    env = RenovationEnv()
    rewards = []
    step = 0
    success = False
    error = None

    print(f"[START] task={task_name} env=renovation-env model={MODEL_NAME}")

    try:
        state = env.reset(TASKS[task_name])

        while True:
            step += 1

            action = get_action(state)

            state, reward, done, info = env.step(action)

            error = info.get("error", None) if info else None

            rewards.append(float(reward))

            print(
                f"[STEP] step={step} "
                f"action={action} "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"error={error if error else 'null'}"
            )

            if done:
                break

    except Exception as e:
        error = str(e)
        success = False

    finally:
        try:
            env.close()
        except:
            pass

        # =========================
        # FINAL GRADER LOGIC
        # =========================
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)

        # ✅ ADVANCED AGENT GRADER
        if task_name == "easy":
            success = score >= 0.6
        elif task_name == "medium":
            success = score >= 0.5
        else:  # hard
            success = score >= 0.4

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        print(
            f"[END] success={str(success).lower()} "
            f"steps={step} "
            f"score={score:.2f} "
            f"rewards={rewards_str}"
        )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)