from fastapi import FastAPI
import pickle
import pandas as pd
from BankNotes import BankNote
import uvicorn

app = FastAPI()

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

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=10000)