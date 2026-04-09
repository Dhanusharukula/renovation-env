import os
import json
from openai import OpenAI
from env.renovation_env import RenovationEnv

# =========================
# ENV (MANDATORY)
# =========================
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

# =========================
# TASKS
# =========================
TASKS = {
    "easy": {"budget": 30000, "style": "modern"},
    "medium": {"budget": 40000, "style": "classic"},
    "hard": {"budget": 35000, "style": "modern"}
}

# =========================
# LLM ACTION (REQUIRED)
# =========================
def get_action(state):
    prompt = f"""
You are an AI interior designer.

State:
{json.dumps(state)}

Available items:
["chair", "table", "paint", "light"]

Choose best item based on style and budget.

Return ONLY JSON:
{{"action": "<item>"}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()

        # clean markdown if present
        if "```" in content:
            content = content.split("```")[-1].strip()

        return json.loads(content)["action"]

    except Exception:
        return "chair"  # safe fallback

# =========================
# RUN TASK
# =========================
def run_task(task_name):
    env = RenovationEnv()
    rewards = []
    step = 0
    success = False
    error = None

    print(f"[START] task={task_name}")

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

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)

        if task_name == "easy":
            success = score >= 0.6
        elif task_name == "medium":
            success = score >= 0.5
        else:
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
