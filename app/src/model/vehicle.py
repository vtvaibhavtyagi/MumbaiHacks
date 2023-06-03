import uuid
import json
from fastapi import FastAPI, UploadFile
from datetime import datetime, time
from typing import List, Optional
from pydantic import BaseModel
from pydantic.types import UUID4, constr


class newVehicle(BaseModel):
    vehicle_number: str
    image: UploadFile
    color: Optional[str] =None
    owner: str
    description: Optional[str] =None


class extractedInfo(BaseModel):
    extractedColor: str
    extractedVehicleNo: str