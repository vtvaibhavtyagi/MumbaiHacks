import uuid
import json
from fastapi import FastAPI, UploadFile
from datetime import datetime, time
from typing import List, Optional
from pydantic import BaseModel
from pydantic.types import UUID4, constr

from src.model.vehicle import extractedInfo


def extractVehicleInfo(image: UploadFile):
    veh = extractedInfo()
    
    return veh
    
    