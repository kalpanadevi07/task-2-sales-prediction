from pydantic import BaseModel

class Features(BaseModel):
    lag_1: float
    lag_7: float
    rolling_mean_7: float   # ✅ ADD THIS

class SalesRequest(BaseModel):
    date: str
    features: Features