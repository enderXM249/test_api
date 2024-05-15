from pydantic import BaseModel

class BaseNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float

class ImageFileSchema(BaseModel):
    image_file_base64_str: str    

class BaseNote2(BaseModel):
    temparature: float 
    humidity: float 
    moisture: float 
    nitrogen: float
    potassium: float
    phosphorous: float
    soil_type_black: int = 0
    soil_type_clayey: int = 0
    soil_type_loamy: int = 0
    soil_type_red: int = 0
    soil_type_sandy: int = 0
    crop_type_barley: int = 0
    crop_type_cotton: int = 0
    crop_type_ground_nuts: int = 0
    crop_type_maize: int = 0
    crop_type_millets: int = 0
    crop_type_oil_seeds: int = 0
    crop_type_paddy: int = 0
    crop_type_pulses: int = 0
    crop_type_sugarcane: int = 0
    crop_type_tobacco: int = 0
    crop_type_wheat: int = 0	 
 
class CropPred(BaseModel):
    temperature: float
    humidity: float
    ph: float
    rainfall: float       