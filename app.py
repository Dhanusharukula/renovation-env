from fastapi import FastAPI
from pydantic import BaseModel
from env.renovation_env import RenovationEnv

app = FastAPI()

env = RenovationEnv()


# =========================
# Request Models
# =========================
class ResetRequest(BaseModel):
    budget: int
    style: str


class StepRequest(BaseModel):
    action: str


# =========================
# ENDPOINTS
# =========================

@app.post("/reset")
def reset(req: ResetRequest):
    state = env.reset({
        "budget": req.budget,
        "style": req.style
    })
    return {"state": state}


@app.post("/step")
def step(req: StepRequest):
    state, reward, done, info = env.step(req.action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def get_state():
    return {"state": env.state()}


# Optional (for browser root)
@app.get("/")
def home():
    return {"message": "API running"}