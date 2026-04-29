from fastapi import APIRouter
from app.schemas.request_schema import SalesRequest
from app.services.predictor import predict_sales

router = APIRouter()

@router.post("/predict")
def predict(data: SalesRequest):
    result = predict_sales(data)
    return {"predicted_sales": result}