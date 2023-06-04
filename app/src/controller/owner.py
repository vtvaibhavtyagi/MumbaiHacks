import os
from datetime import datetime, timedelta
from fastapi import APIRouter
from fastapi import UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.model.vehicle import newVehicle, extractedInfo
from src.service import vehicleInfo

controller = APIRouter()
IMAGEDIR = "src/img/"


@controller.post("/register-vehicle/")
async def register_vehicle(file: UploadFile = File(...)):
    file_path = os.path.join(
        IMAGEDIR, file.filename)
    with open(file_path, "wb") as image:
        image.write(await file.read())
    return vehicleInfo.extractvehicleInfo(file)

    # return newVehExtractedInfo
