from fastapi import APIRouter, Response
from services import predict_outcome, predict_disease_outcome, predict_outcome2, predict_outcome3
from schemas import BaseNote, ImageFileSchema, BaseNote2, CropPred

router = APIRouter()

@router.post('/predict')
async def predict(data: BaseNote, response: Response):
    return predict_outcome(data,response)

@router.post('/get-disease-summary')
async def get_disease_summary(image_file: ImageFileSchema, response: Response):
    return predict_disease_outcome(image_file,response)

@router.post('/predictferti')
async def predict(data: BaseNote2, response: Response):
    return predict_outcome2(data,response)

@router.post('/predictcrop')
async def predict(data: CropPred, response: Response):
    return predict_outcome3(data,response)