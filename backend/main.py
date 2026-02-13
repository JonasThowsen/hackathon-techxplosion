from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import BuildingLayout
from sample_building import SAMPLE_BUILDING

app = FastAPI(title="Energy Waste Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/building")
def get_building() -> BuildingLayout:
    return SAMPLE_BUILDING
