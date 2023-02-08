# basic libs
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# cleaning data
import re
import os

# save vocabulary in files
import pickle
# pad sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,load_model
import translation
import serial

from typing import Union
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message":20}
  
@app.get("/items/{text}")
async def read_item(text):
    trans=translation.translate_eng_fr(text)
    return {
      "id" : serial.generate_serial(),
      "text" : text,
      "translate": trans,
    }