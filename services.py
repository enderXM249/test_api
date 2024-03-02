from schemas import BaseNote, ImageFileSchema
from fastapi.encoders import jsonable_encoder
from fastapi import Response, status
import pickle
import os
import warnings
import shutil
import pybase64
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from PIL import Image

warnings.filterwarnings('ignore')       
CLASSIFIER_FILE_PATH = os.path.join(os.getcwd(),'classifier.pkl')
MODEL_FILE_PATH = os.path.join(os.getcwd(),'resnet18v2classification_model.pth')
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
        response.status_code = status.HTTP_200_OK 
        return jsonable_encoder({'prediction': prediction_outcome,'status': status.HTTP_200_OK})  
    except Exception as error:    
        response.status_code=status.HTTP_400_BAD_REQUEST
        return jsonable_encoder({'error': str(error),'status': status.HTTP_400_BAD_REQUEST})

def predict_disease_outcome(image_file: ImageFileSchema, response: Response ):
    try:
        TEMP_IMAGE_FILE_PATH = os.path.join(os.getcwd(),'temp_image_storage','test.png')
        os.makedirs('temp_image_storage',exist_ok=True)    
        with open( TEMP_IMAGE_FILE_PATH, 'wb') as ifh:
            ifh.write(pybase64.b64decode(image_file.image_file_base64_str))
        device = "mps" if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')    
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1000)  
        model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))
        model.eval()
        new_model = models.resnet18(pretrained=True)
        new_model.fc = nn.Linear(new_model.fc.in_features, 38)  
        new_model.fc.weight.data = model.fc.weight.data[0:38]  
        new_model.fc.bias.data = model.fc.bias.data[0:38]
    
        image = Image.open(TEMP_IMAGE_FILE_PATH)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  #
        with torch.no_grad():
            output = model(input_batch)
        _, predicted_class = output.max(1)
        class_names = ['Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'] 
        predicted_class_name = class_names[predicted_class.item()]
        shutil.rmtree(os.path.join(os.getcwd(),'temp_image_storage'))
        response.status_code = status.HTTP_200_OK       
        return jsonable_encoder({'predicted_class_name': predicted_class_name ,'status': status.HTTP_200_OK})
    except Exception as error:    
        shutil.rmtree(os.path.join(os.getcwd(),'temp_image_storage'))
        response.status_code=status.HTTP_400_BAD_REQUEST
        return jsonable_encoder({'error': str(error),'status': status.HTTP_400_BAD_REQUEST})
        