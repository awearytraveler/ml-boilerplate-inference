from fastapi import FastAPI, HTTPException
from inference import runInference, loadModel
import validators
from validators import ValidationFailure
import pickle
import requests
import PIL
from pydantic import BaseModel
from io import BytesIO

app = FastAPI()
model = loadModel(export_dir='finetuned_model_export')
class_names = pickle.loads(open('labels.pickle', "rb").read())

class Image(BaseModel):
    url: str

@app.get("/")
async def root():
    return {"message": "Inference Service"}

@app.post("/image/")
async def imageInference(image_url: Image):
    
    res = validators.url(image_url.url)
    if isinstance(res, ValidationFailure):
        raise HTTPException(status_code=400, detail="Parameter not URL")
    
    response = requests.get(image_url.url)
    image = PIL.Image.open(BytesIO(response.content))
    res = runInference(image, model)
    res = class_names[res.argmax(axis=1)[0]]
    return {'message':res}
    
