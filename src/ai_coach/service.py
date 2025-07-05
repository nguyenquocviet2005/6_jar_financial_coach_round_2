"""AI Coach service implementation."""

import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from src.common.dependencies import (
    get_bedrock_runtime_client,
    get_chroma_client,
    get_sagemaker_client,
    get_s3_client
)
from src.common.logging import get_logger
from src.ai_coach.models import (
    CoachingRequest,
    CoachingResponse,
    ContextData,
    ChatSession,
    ProactiveAlert
)

logger = get_logger(__name__)


class AICoachService:
    """AI Coach service for financial advisory."""
    
    def __init__(self):
        self.bedrock_client = get_bedrock_runtime_client()
        self.chroma_client = get_chroma_client()
        self.sagemaker_client = get_sagemaker_client()
        self.s3_client = get_s3_client()
        self.collection_name = "financial_knowledge"
        
    async def get_coaching_advice(self, request: CoachingRequest) -> CoachingResponse:
        """Get AI coaching advice for a user query."""
        try:
            session_id = str(uuid.uuid4())
            
            # 1. Aggregate context data
            context_data = await self._aggregate_context(request.user_id, request.context)
            
            # 2. Perform semantic search for relevant knowledge
            relevant_knowledge = await self._search_knowledge_base(request.query, context_data)
            
            # 3. Generate ML predictions if needed
            ml_predictions = await self._get_ml_predictions(request, context_data)
            
            # 4. Create comprehensive prompt
            prompt = self._create_coaching_prompt(request, context_data, relevant_knowledge, ml_predictions)
            
            # 5. Generate advice using Bedrock
            advice_response = await self._generate_advice(prompt)
            
            # 6. Process and structure response
            structured_response = self._process_advice_response(advice_response, context_data)
            
            # 7. Store session for follow-up
            await self._store_coaching_session(session_id, request, structured_response, context_data)
            
            return CoachingResponse(
                session_id=session_id,
                advice=structured_response["advice"],
                confidence_score=structured_response["confidence_score"],
                suggested_actions=structured_response["suggested_actions"],
                related_products=structured_response["related_products"],
                context_used=structured_response["context_used"],
                follow_up_questions=structured_response["follow_up_questions"]
            )
            
        except Exception as e:
            logger.error("Error in coaching advice generation", error=str(e), request=request.dict())
            raise
    
    async def _aggregate_context(self, user_id: str, additional_context: Optional[Dict[str, Any]] = None) -> ContextData:
        """Aggregate context data for the user."""
        # This would typically query your database for user data
        # For now, we'll simulate the data structure
        
        context_data = {
            "user_profile": {
                "user_id": user_id,
                "income": 5000,
                "risk_tolerance": "moderate",
                "financial_goals": ["emergency_fund", "retirement"]
            },
            "recent_transactions": [
                {"amount": 50, "category": "food", "jar_type": "necessities"},
                {"amount": 200, "category": "entertainment", "jar_type": "play"}
            ],
            "jar_balances": {
                "necessities": 2500,
                "financial_freedom": 500,
                "long_term_savings": 800,
                "education": 300,
                "play": 600,
                "give": 300
            },
            "spending_patterns": {
                "daily_average": 150,
                "top_categories": ["food", "transport", "entertainment"],
                "trend": "increasing"
            },
            "financial_goals": [
                {"goal": "emergency_fund", "target": 15000, "current": 5000},
                {"goal": "vacation", "target": 3000, "current": 1200}
            ]
        }
        
        if additional_context:
            context_data.update(additional_context)
            
        return ContextData(**context_data)
    
    async def _search_knowledge_base(self, query: str, context: ContextData) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        try:
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Financial knowledge base"}
            )
            
            # Perform semantic search
            results = collection.query(
                query_texts=[query],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            knowledge_items = []
            if results["documents"]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    knowledge_items.append({
                        "content": doc,
                        "metadata": metadata,
                        "relevance_score": 1 - distance  # Convert distance to similarity
                    })
            
            return knowledge_items
            
        except Exception as e:
            logger.warning("Error searching knowledge base", error=str(e))
            return []
    
    async def _get_ml_predictions(self, request: CoachingRequest, context: ContextData) -> Dict[str, Any]:
        """Get ML predictions from SageMaker endpoints."""
        predictions = {}
        
        try:
            # Spending prediction
            if request.coaching_type in ["spending_advice", "budget_optimization"]:
                spending_prediction = await self._get_spending_prediction(request.user_id, context)
                predictions["spending_forecast"] = spending_prediction
            
            # Investment recommendation
            if request.coaching_type == "investment_suggestion":
                investment_prediction = await self._get_investment_recommendation(request.user_id, context)
                predictions["investment_recommendation"] = investment_prediction
                
        except Exception as e:
            logger.warning("Error getting ML predictions", error=str(e))
            
        return predictions
    
    async def _get_spending_prediction(self, user_id: str, context: ContextData) -> Dict[str, Any]:
        """Get spending prediction from SageMaker endpoint."""
        # This would call your SageMaker endpoint
        # For now, we'll simulate a response
        return {
            "predicted_spending": 1200,
            "confidence": 0.85,
            "factors": ["historical_pattern", "season", "income_level"]
        }
    
    async def _get_investment_recommendation(self, user_id: str, context: ContextData) -> Dict[str, Any]:
        """Get investment recommendation from SageMaker endpoint."""
        # This would call your SageMaker endpoint
        # For now, we'll simulate a response
        return {
            "recommended_allocation": {
                "stocks": 0.6,
                "bonds": 0.3,
                "cash": 0.1
            },
            "risk_score": 0.7,
            "expected_return": 0.08
        }
    
    def _create_coaching_prompt(
        self,
        request: CoachingRequest,
        context: ContextData,
        knowledge: List[Dict[str, Any]],
        predictions: Dict[str, Any]
    ) -> str:
        """Create a comprehensive prompt for the LLM."""
        
        knowledge_context = "\n".join([
            f"- {item['content']}" for item in knowledge[:3]  # Top 3 relevant items
        ])
        
        prompt = f"""
You are an expert financial advisor for a 6-jar budgeting system. Provide personalized financial advice based on the following context:

USER QUERY: {request.query}
COACHING TYPE: {request.coaching_type}

USER CONTEXT:
- Income: ${context.user_profile.get('income', 0):,.2f}
- Jar Balances: {context.jar_balances}
- Recent Spending: {context.spending_patterns.get('daily_average', 0)}
- Financial Goals: {[goal['goal'] for goal in context.financial_goals]}

RELEVANT KNOWLEDGE:
{knowledge_context}

ML PREDICTIONS:
{json.dumps(predictions, indent=2)}

INSTRUCTIONS:
1. Provide clear, actionable financial advice
2. Reference the 6-jar budgeting system where appropriate
3. Suggest specific actions the user can take
4. Recommend relevant financial products if applicable
5. Keep advice personalized to the user's situation
6. Use a friendly, supportive tone

Structure your response as JSON with the following fields:
- advice: Main financial advice (string)
- confidence_score: Your confidence in the advice (0.0-1.0)
- suggested_actions: List of specific actions to take
- related_products: List of recommended financial products
- follow_up_questions: Questions to ask for more personalized advice

Response:
"""
        return prompt
    
    async def _generate_advice(self, prompt: str) -> str:
        """Generate advice using Bedrock."""
        try:
            # Call Bedrock Claude model
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            logger.error("Error generating advice with Bedrock", error=str(e))
            # Fallback response
            return json.dumps({
                "advice": "I apologize, but I'm having trouble generating personalized advice right now. Please try again later.",
                "confidence_score": 0.3,
                "suggested_actions": ["Contact customer support", "Try again in a few minutes"],
                "related_products": [],
                "follow_up_questions": []
            })
    
    def _process_advice_response(self, response: str, context: ContextData) -> Dict[str, Any]:
        """Process and structure the LLM response."""
        try:
            # Try to parse JSON response
            structured_response = json.loads(response)
            
            # Validate and set defaults
            return {
                "advice": structured_response.get("advice", "No advice available"),
                "confidence_score": max(0.0, min(1.0, structured_response.get("confidence_score", 0.5))),
                "suggested_actions": structured_response.get("suggested_actions", []),
                "related_products": structured_response.get("related_products", []),
                "follow_up_questions": structured_response.get("follow_up_questions", []),
                "context_used": {
                    "user_id": context.user_profile.get("user_id"),
                    "jar_balances": context.jar_balances,
                    "spending_trend": context.spending_patterns.get("trend")
                }
            }
            
        except json.JSONDecodeError:
            # If response is not JSON, create a structured response
            return {
                "advice": response,
                "confidence_score": 0.7,
                "suggested_actions": [],
                "related_products": [],
                "follow_up_questions": [],
                "context_used": {}
            }
    
    async def _store_coaching_session(
        self,
        session_id: str,
        request: CoachingRequest,
        response: Dict[str, Any],
        context: ContextData
    ):
        """Store coaching session for analysis and follow-up."""
        session_data = {
            "session_id": session_id,
            "user_id": request.user_id,
            "query": request.query,
            "coaching_type": request.coaching_type,
            "response": response,
            "context": context.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in database or analytics system
        # For now, we'll log it
        logger.info("Coaching session stored", session=session_data)
    
    async def generate_proactive_alert(self, user_id: str, alert_type: str, context: Dict[str, Any]) -> ProactiveAlert:
        """Generate proactive coaching alerts."""
        alert_id = str(uuid.uuid4())
        
        # Generate alert based on type and context
        if alert_type == "overspending":
            message = f"You've spent ${context.get('amount', 0):.2f} from your {context.get('jar_type', 'unknown')} jar. Consider reviewing your budget."
            priority = "high"
        elif alert_type == "low_balance":
            message = f"Your {context.get('jar_type', 'unknown')} jar balance is getting low. Time to reassess your spending priorities."
            priority = "medium"
        else:
            message = "We noticed some changes in your spending patterns. Let's review your financial goals."
            priority = "low"
        
        return ProactiveAlert(
            alert_id=alert_id,
            user_id=user_id,
            alert_type=alert_type,
            message=message,
            priority=priority,
            jar_type=context.get('jar_type'),
            amount=context.get('amount'),
            context=context
        ) 