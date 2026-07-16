from fastapi import APIRouter, status, HTTPException
from app.schemas import ChatQueryInput, ChatQueryResponse
from app.agent.graph import run_query
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["AI Fleet Advisor"])

@router.post("/query", response_model=ChatQueryResponse, status_code=status.HTTP_200_OK,
             responses={500: {"model": ChatQueryResponse}})
async def query_fleet_advisor(payload: ChatQueryInput) -> ChatQueryResponse:
    """
    Query the LangGraph AI Fleet Advisor to get insights about battery telemetry,
    readiness schedules, carbon footprint offsets, or charging station proximity.
    """
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be blank."
        )
        
    try:
        # Run query through LangGraph Fleet Advisor
        result = run_query(
            user_message=payload.message,
            chat_history=payload.chat_history
        )
        return ChatQueryResponse(
            response=result["response"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error querying AI Fleet Advisor: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while compiling your fleet query. Please check your data logs."
        )

