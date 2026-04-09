import json
from env.renovation_env import RenovationEnv

# =========================
# TASKS
# =========================
TASKS = {
    "easy": {"budget": 30000, "style": "modern"},
    "medium": {"budget": 40000, "style": "classic"},
    "hard": {"budget": 35000, "style": "modern"}
}

# =========================
# RULE-BASED AGENT (NO API)
# =========================
def get_action(state):
    style = state["style"]

    # Smart selection based on style
    if style == "modern":
        for item in ["chair", "table", "light"]:
            if item not in state["items_selected"]:
                return item

    if style == "classic":
        if "paint" not in state["items_selected"]:
            return "paint"

    # fallback
    return "chair"

# =========================
# RUN TASK
# =========================
def run_task(task_name):
    env = RenovationEnv()
    rewards = []
    step = 0
    success = False
    error = None

    print(f"[START] task={task_name} env=renovation-env")

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
        # FINAL GRADER
        # =========================
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
