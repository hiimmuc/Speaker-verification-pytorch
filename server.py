from typing import Optional, List
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import cv2
import time
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")




@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})



@app.post("/add")
async def post_new_item(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    return {
        'information': 'Hello'
    }

if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=8111)
