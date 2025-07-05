"""Pydantic models for AI Coach service."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from src.common.models import BaseResponse, JarType


class CoachingType(str, Enum):
    """Types of coaching requests."""
    SPENDING_ADVICE = "spending_advice"
    INVESTMENT_SUGGESTION = "investment_suggestion"
    BUDGET_OPTIMIZATION = "budget_optimization"
    FINANCIAL_PLANNING = "financial_planning"
    DEBT_MANAGEMENT = "debt_management"
    EMERGENCY_FUND = "emergency_fund"


class CoachingRequest(BaseModel):
    """Request for AI coaching."""
    user_id: str
    query: str
    coaching_type: CoachingType
    context: Optional[Dict[str, Any]] = None
    transaction_id: Optional[str] = None
    jar_type: Optional[JarType] = None
    amount: Optional[float] = None


class CoachingResponse(BaseResponse):
    """Response from AI coach."""
    session_id: str
    advice: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggested_actions: List[str] = Field(default_factory=list)
    related_products: List[str] = Field(default_factory=list)
    context_used: Dict[str, Any] = Field(default_factory=dict)
    follow_up_questions: List[str] = Field(default_factory=list)


class ContextData(BaseModel):
    """Context data for coaching."""
    user_profile: Dict[str, Any]
    recent_transactions: List[Dict[str, Any]]
    jar_balances: Dict[JarType, float]
    spending_patterns: Dict[str, Any]
    financial_goals: List[Dict[str, Any]]
    market_data: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role: user or assistant")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str
    user_id: str
    messages: List[ChatMessage]
    context: ContextData
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class ProactiveAlert(BaseModel):
    """Proactive coaching alert."""
    alert_id: str
    user_id: str
    alert_type: str
    message: str
    priority: str = Field(..., description="Priority: low, medium, high")
    jar_type: Optional[JarType] = None
    amount: Optional[float] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_sent: bool = False


class KnowledgeBase(BaseModel):
    """Knowledge base entry."""
    entry_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow) 