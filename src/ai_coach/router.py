"""FastAPI router for AI Coach service."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import structlog

from src.ai_coach.service import AICoachService
from src.ai_coach.models import (
    CoachingRequest,
    CoachingResponse,
    ChatSession,
    ProactiveAlert
)
from src.common.models import BaseResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency
def get_ai_coach_service() -> AICoachService:
    """Get AI Coach service instance."""
    return AICoachService()


@router.post("/advice", response_model=CoachingResponse)
async def get_coaching_advice(
    request: CoachingRequest,
    service: AICoachService = Depends(get_ai_coach_service)
):
    """Get personalized financial coaching advice."""
    try:
        response = await service.get_coaching_advice(request)
        return response
    except Exception as e:
        logger.error("Error getting coaching advice", error=str(e), request=request.dict())
        raise HTTPException(status_code=500, detail="Failed to generate coaching advice")


@router.post("/chat", response_model=CoachingResponse)
async def chat_with_coach(
    request: CoachingRequest,
    service: AICoachService = Depends(get_ai_coach_service)
):
    """Chat interface for AI coach."""
    try:
        # For chat, we use the same underlying service but could add
        # session management and conversation history
        response = await service.get_coaching_advice(request)
        return response
    except Exception as e:
        logger.error("Error in chat session", error=str(e), request=request.dict())
        raise HTTPException(status_code=500, detail="Failed to process chat request")


@router.post("/proactive-alert", response_model=ProactiveAlert)
async def generate_proactive_alert(
    user_id: str,
    alert_type: str,
    context: Dict[str, Any],
    service: AICoachService = Depends(get_ai_coach_service)
):
    """Generate proactive coaching alerts."""
    try:
        alert = await service.generate_proactive_alert(user_id, alert_type, context)
        return alert
    except Exception as e:
        logger.error("Error generating proactive alert", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to generate proactive alert")


@router.get("/knowledge-base/search")
async def search_knowledge_base(
    query: str,
    limit: int = 5,
    service: AICoachService = Depends(get_ai_coach_service)
):
    """Search the financial knowledge base."""
    try:
        # Create a dummy context for search
        from src.ai_coach.models import ContextData
        context = ContextData(
            user_profile={},
            recent_transactions=[],
            jar_balances={},
            spending_patterns={},
            financial_goals=[]
        )
        
        results = await service._search_knowledge_base(query, context)
        return {"results": results[:limit]}
    except Exception as e:
        logger.error("Error searching knowledge base", error=str(e), query=query)
        raise HTTPException(status_code=500, detail="Failed to search knowledge base")


@router.post("/knowledge-base/add")
async def add_knowledge_entry(
    title: str,
    content: str,
    category: str,
    tags: List[str],
    service: AICoachService = Depends(get_ai_coach_service)
):
    """Add entry to the knowledge base."""
    try:
        # Get or create collection
        collection = service.chroma_client.get_or_create_collection(
            name=service.collection_name,
            metadata={"description": "Financial knowledge base"}
        )
        
        # Add document
        entry_id = f"{category}_{title.replace(' ', '_').lower()}"
        collection.add(
            documents=[content],
            metadatas=[{
                "title": title,
                "category": category,
                "tags": ",".join(tags)
            }],
            ids=[entry_id]
        )
        
        return BaseResponse(message=f"Knowledge entry '{title}' added successfully")
    except Exception as e:
        logger.error("Error adding knowledge entry", error=str(e), title=title)
        raise HTTPException(status_code=500, detail="Failed to add knowledge entry")


@router.get("/health")
async def health_check():
    """Health check for AI Coach service."""
    return {"status": "healthy", "service": "ai-coach"}


@router.get("/metrics")
async def get_metrics():
    """Get AI Coach service metrics."""
    # This would typically return real metrics
    return {
        "total_sessions": 1000,
        "avg_response_time": 2.5,
        "confidence_score_avg": 0.8,
        "user_satisfaction": 4.2
    } 