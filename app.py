from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
from env.renovation_env import RenovationEnv

app = FastAPI()
env = RenovationEnv()

# ✅ Request models
class ResetRequest(BaseModel):
    budget: Optional[int] = 30000
    style: Optional[str] = "modern"

class StepRequest(BaseModel):
    action: str

# ✅ FINAL FIXED RESET (handles BOTH cases)
@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        # Validator case (no body)
        state = env.reset()
    else:
        # Normal case (with body)
        state = env.reset({
            "budget": req.budget,
            "style": req.style
        })
    return {"state": state}

# ✅ STEP
@app.post("/step")
def step(req: StepRequest):
    state, reward, done, info = env.step(req.action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }

# ✅ STATE
@app.get("/state")
def get_state():
    return {"state": env.state()}

# ✅ ROOT
@app.get("/")
def root():
    return {"message": "API running"}
