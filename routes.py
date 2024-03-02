from fastapi import APIRouter, Response
from services import predict_outcome, predict_disease_outcome
from schemas import BaseNote, ImageFileSchema

router = APIRouter()

@router.post('/predict')
async def predict(data: BaseNote, response: Response):
    return predict_outcome(data,response)

@router.post('/get-disease-summary')
async def get_disease_summary(image_file: ImageFileSchema, response: Response):
    return predict_disease_outcome(image_file,response)