from fastapi import APIRouter, Response
from services import predict_outcome
from schemas import BaseNote

router = APIRouter()

@router.post('/predict')
def predict(data: BaseNote, response: Response):
    return predict_outcome(data,response)