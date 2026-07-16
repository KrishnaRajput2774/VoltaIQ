from app.services.models_service import model_service, ModelService
from app.services.carbon_service import carbon_service, CarbonService

def get_model_service() -> ModelService:
    """Dependency provider for ModelService."""
    return model_service

def get_carbon_service() -> CarbonService:
    """Dependency provider for CarbonService."""
    return carbon_service
