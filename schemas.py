from pydantic import BaseModel

class BaseNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float
    