from fastapi import APIRouter
from fastapi import Depends, FastAPI, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.model.vehicle import newVehicle, extractedInfo
from src.model.helper import newHelper
from src.service import vehicleInfo
import base64
from PIL import Image
import uuid
import os
import cv2
controller = APIRouter()
IMAGEDIR = "src/img/"


@controller.post("/help/")
async def verify_vehicle(file: UploadFile = File(...)):

    file_path = os.path.join(
        IMAGEDIR, file.filename)
    with open(file_path, "wb") as image:
        image.write(await file.read())
    return vehicleInfo.extractvehicleInfo(file)
