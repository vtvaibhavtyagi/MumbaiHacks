from fastapi import APIRouter
from fastapi import Depends, FastAPI, HTTPException, status, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.model.vehicle import newVehicle,extractedInfo
from src.model.helper import newHelper
from src.service import vehicleInfo

controller = APIRouter()

@controller.get("/help/")
def verify_vehicle(helper:newHelper ,verifyImage: UploadFile):
    extractedInfo = vehicleInfo.extractVehicleInfo(image=verifyImage)
    
    return extractedInfo