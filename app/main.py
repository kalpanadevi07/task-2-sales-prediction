from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI()

app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "Sales Forecast API Running 🚀"}