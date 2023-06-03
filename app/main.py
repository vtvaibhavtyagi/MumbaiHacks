import uvicorn
from typing import Union
from fastapi import FastAPI,Depends
from src.controller import owner,helper

app = FastAPI()


app.include_router(owner.controller, prefix="/api/owner", tags=["owner"],)
app.include_router(helper.controller, prefix="/api/helper", tags=["helper"])


if __name__ == "__main__":
    # create_default_usee()
    uvicorn.run(app)