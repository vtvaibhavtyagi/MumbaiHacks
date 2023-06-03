from fastapi import FastAPI, UploadFile
from datetime import datetime, time
from typing import List, Optional
from pydantic import BaseModel


class newHelper(BaseModel):
    IPAddress: str