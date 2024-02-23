from schemas import BaseNote
from fastapi.encoders import jsonable_encoder
from fastapi import Response, status
import pickle
import os
import warnings

warnings.filterwarnings('ignore')       
CLASSIFIER_FILE_PATH = os.path.join(os.getcwd(),'classifier.pkl')
pickle_in = open(CLASSIFIER_FILE_PATH,'rb')
classifier=pickle.load(pickle_in)

def predict_outcome(data: BaseNote, response: Response):
    try:
        data = data.dict()
        variance=data['variance']
        skewness=data['skewness']
        curtosis=data['curtosis']
        entropy=data['entropy']
        prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
        prediction_outcome = 'Fake note' if prediction[0]>0.5 else 'Its a Bank note'
        return jsonable_encoder({'prediction': prediction_outcome,'status': status.HTTP_200_OK})  
    except Exception as error:    
        response.status_code=status.HTTP_400_BAD_REQUEST
        return jsonable_encoder({'error': str(error),'status': status.HTTP_400_BAD_REQUEST})
    