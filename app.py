from fastapi import FastAPI
from pydantic import BaseModel
from env.renovation_env import RenovationEnv

app = FastAPI()
env = RenovationEnv()

# ✅ Request models
class ResetRequest(BaseModel):
    budget: int
    style: str

class StepRequest(BaseModel):
    action: str

# ✅ RESET FIXED
@app.post("/reset")
def reset(req: ResetRequest):
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

# ✅ ROOT (optional)
@app.get("/")
def root():
    return {"message": "API running"}
