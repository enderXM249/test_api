from pydantic import BaseModel

class BaseNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float

class ImageFileSchema(BaseModel):
    image_file_base64_str: str    
    