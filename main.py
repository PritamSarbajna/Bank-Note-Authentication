from fastapi import FastAPI
import pickle
import pandas as pd
from BankNotes import BankNote
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


pickle_in = open('classifier.pkl', 'rb')
clf = pickle.load(pickle_in)

@app.get('/')
async def index():
    return {'msg': 'Hello world'}

@app.post('/predict')
async def predict(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    
    prediction = clf.predict([[variance,skewness,curtosis,entropy]])
    
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }