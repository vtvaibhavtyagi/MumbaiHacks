from datetime import datetime, timedelta
from fastapi import APIRouter
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.model.vehicle import newVehicle,extractedInfo
from src.service import vehicleInfo

controller = APIRouter()

@controller.get("/register-vehicle/{owner}")
def register_vehicle(owner: str, vehicle:newVehicle):
    newVehExtractedInfo = vehicleInfo.extractVehicleInfo(image=vehicle.image)
    
    return newVehExtractedInfo
    

