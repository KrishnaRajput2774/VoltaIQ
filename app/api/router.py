from fastapi import APIRouter
from app.api.fleet import router as fleet_router
from app.api.battery import router as battery_router
from app.api.carbon import router as carbon_router
from app.api.chat import router as chat_router
from app.api.system import router as system_router
from app.api.models_router import router as models_router

# Aggregated V1 versioned router
v1_router = APIRouter()
v1_router.include_router(fleet_router)
v1_router.include_router(battery_router)
v1_router.include_router(carbon_router)
v1_router.include_router(chat_router)
v1_router.include_router(system_router)
v1_router.include_router(models_router)

# Legacy unversioned router (keeps tests passing)
api_router = APIRouter()
api_router.include_router(fleet_router)
api_router.include_router(battery_router)
api_router.include_router(carbon_router)
api_router.include_router(chat_router)
