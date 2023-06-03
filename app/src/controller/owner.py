from datetime import datetime, timedelta
from fastapi import APIRouter
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from src.model.vehicle import newVehicle, extractedInfo
from src.service import vehicleInfo

controller = APIRouter()


@controller.post("/register-vehicle/")
def register_vehicle(vehicle: newVehicle):
    return vehicleInfo.extractVehicleInfo(image=vehicle.image)

    # return newVehExtractedInfo
