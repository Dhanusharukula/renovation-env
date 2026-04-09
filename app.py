from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from env.renovation_env import RenovationEnv

app = FastAPI()
env = RenovationEnv()

# ✅ OPTIONAL fields (IMPORTANT FIX)
class ResetRequest(BaseModel):
    budget: Optional[int] = 30000
    style: Optional[str] = "modern"

class StepRequest(BaseModel):
    action: str

# ✅ FIXED RESET (handles missing body)
@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
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

@app.get("/")
def root():
    return {"message": "API running"}
