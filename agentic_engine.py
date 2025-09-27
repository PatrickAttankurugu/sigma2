"""
Enhanced Agentic Engine for Business Model Canvas Updates
Streamlined for Seedstars Assignment - Real AI with 4-Agent Workflow

This module implements a sophisticated 4-agent workflow that analyzes 
completed actions and proposes intelligent business model updates with
idempotent behavior, auto-mode safety, and production-ready reliability.
"""

import json
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from business_models import (
    BusinessModelCanvas,
    CompletedAction,
    ActionOutcome,
    ProposedChange,
    AgentRecommendation,
    ConfidenceLevel,
    ChangeType,
    AgentStatus
)
from utils import generate_action_hash, deduplicate_changes


class AgenticOrchestrator:
    """
    Production-ready orchestrator for 4-agent workflow.
    Processes action outcomes and generates contextually relevant 
    business model canvas updates with safety mechanisms and idempotent behavior.
    """

    def __init__(
        self, 
        google_api_key: Optional[str] = None, 
        model_name: str = "gemini-2.0-flash"
    ):
        """Initialize the orchestrator with Google Gemini integration."""
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass google_api_key parameter.")

        self.model_name = model_name

        try:
            # Configure Gemini with production-ready settings
            self.llm = ChatGoogleGenerativeAI(
                api_key=self.api_key,
                model=model_name,
                temperature=0.2,
                max_output_tokens=1000,
                timeout=30,
                max_retries=2
            )

            # Initialize specialized agents
            self.action_detection_agent = ActionDetectionAgent(self.llm)
            self.outcome_analysis_agent = OutcomeAnalysisAgent(self.llm)
            self.canvas_update_agent = CanvasUpdateAgent(self.llm)
            self.next_step_agent = NextStepAgent(self.llm)

            # Track processed actions for idempotent behavior
            self.processed_actions = set()
            
            # Performance tracking
            self.processing_metrics = {}
            
        except Exception as e:
            raise ValueError(f"Failed to initialize AgenticOrchestrator: {str(e)}")

    async def process_action_outcome(
        self,
        action_data: Dict[str, Any],
        current_bmc: BusinessModelCanvas,
        status_callback: Optional[callable] = None
    ) -> AgentRecommendation:
        """
        Process a completed action through the 4-agent workflow with idempotent behavior.

        Args:
            action_data: Raw action outcome data
            current_bmc: Current business model canvas state
            status_callback: Optional callback for UI status updates

        Returns:
            Complete agent recommendation with proposed changes
        """
        start_time = time.time()
        
        try:
            # Add overall timeout for the entire workflow
            return await asyncio.wait_for(
                self._process_workflow(action_data, current_bmc, status_callback),
                timeout=120  # 2 minutes total timeout
            )
        except asyncio.TimeoutError:
            return self._create_timeout_response(action_data)
        except Exception as e:
            return self._create_error_response(action_data, str(e))
    
    async def _process_workflow(
        self,
        action_data: Dict[str, Any],
        current_bmc: BusinessModelCanvas,
        status_callback: Optional[callable] = None
    ) -> AgentRecommendation:
        """Internal workflow processing with timeout protection."""
        start_time = time.time()
        
        try:
            # Check for duplicate processing (idempotent behavior)
            action_hash = generate_action_hash(action_data)
            if action_hash in self.processed_actions:
                return self._create_duplicate_response(action_data)

            # Agent 1: Action Detection & Validation
            try:
                if status_callback:
                    status_callback("action_detection", "running")
                
                validated_action = await asyncio.wait_for(
                    self.action_detection_agent.process(action_data),
                    timeout=30
                )
                
                if status_callback:
                    status_callback("action_detection", "completed")
            except Exception as e:
                if status_callback:
                    status_callback("action_detection", "failed")
                raise Exception(f"Action detection failed: {str(e)}")

            # Agent 2: Outcome Analysis
            try:
                if status_callback:
                    status_callback("outcome_analysis", "running")
                
                business_analysis = await asyncio.wait_for(
                    self.outcome_analysis_agent.process(validated_action, current_bmc),
                    timeout=30
                )
                
                if status_callback:
                    status_callback("outcome_analysis", "completed")
            except Exception as e:
                if status_callback:
                    status_callback("outcome_analysis", "failed")
                raise Exception(f"Outcome analysis failed: {str(e)}")

            # Agent 3: Canvas Updates
            try:
                if status_callback:
                    status_callback("canvas_update", "running")
                
                proposed_changes = await asyncio.wait_for(
                    self.canvas_update_agent.process(business_analysis, current_bmc, validated_action),
                    timeout=30
                )
                
                # Apply deduplication and validation
                proposed_changes = self._validate_and_enhance_changes(
                    proposed_changes, current_bmc
                )
                
                if status_callback:
                    status_callback("canvas_update", "completed")
            except Exception as e:
                if status_callback:
                    status_callback("canvas_update", "failed")
                raise Exception(f"Canvas update failed: {str(e)}")

            # Agent 4: Next Steps
            try:
                if status_callback:
                    status_callback("next_step", "running")
                
                next_actions = await asyncio.wait_for(
                    self.next_step_agent.process(proposed_changes, current_bmc, validated_action),
                    timeout=30
                )
                
                if status_callback:
                    status_callback("next_step", "completed")
            except Exception as e:
                if status_callback:
                    status_callback("next_step", "failed")
                raise Exception(f"Next step generation failed: {str(e)}")

            # Mark as processed for idempotent behavior
            self.processed_actions.add(action_hash)

            # Compile final recommendation
            processing_time = int((time.time() - start_time) * 1000)
            
            recommendation = AgentRecommendation(
                proposed_changes=proposed_changes,
                next_actions=next_actions,
                reasoning=business_analysis.get("summary", "Analysis completed successfully"),
                confidence_level=self._determine_overall_confidence(proposed_changes),
                processing_time_ms=processing_time,
                model_version=self.model_name,
                source_action_id=validated_action.id if validated_action else action_hash
            )

            # Store metrics for monitoring
            self.processing_metrics[action_hash] = {
                "processing_time_ms": processing_time,
                "changes_count": len(proposed_changes),
                "confidence_level": recommendation.confidence_level.value,
                "timestamp": datetime.now().isoformat()
            }

            return recommendation

        except Exception as e:
            # Mark all agents as failed if we reach here
            if status_callback:
                for agent in ["action_detection", "outcome_analysis", "canvas_update", "next_step"]:
                    status_callback(agent, "failed")
            
            # Return graceful failure response
            return self._create_error_response(action_data, str(e))

    def _create_duplicate_response(self, action_data: Dict[str, Any]) -> AgentRecommendation:
        """Create response for duplicate action processing (idempotent behavior)."""
        return AgentRecommendation(
            proposed_changes=[],
            next_actions=["This action has already been processed"],
            reasoning="Idempotent behavior: This action outcome has already been analyzed and processed.",
            confidence_level=ConfidenceLevel.HIGH,
            processing_time_ms=0,
            model_version=self.model_name
        )
    
    def _create_timeout_response(self, action_data: Dict[str, Any]) -> AgentRecommendation:
        """Create response for timeout scenarios."""
        return AgentRecommendation(
            proposed_changes=[],
            next_actions=[
                "Retry the analysis with a simpler action",
                "Check API connectivity and try again",
                "Consider manual analysis for complex actions"
            ],
            reasoning="Workflow timeout: The analysis took longer than expected. This may be due to API latency or complex action data.",
            confidence_level=ConfidenceLevel.LOW,
            processing_time_ms=120000,  # 2 minutes
            model_version=self.model_name
        )
    
    def _create_error_response(self, action_data: Dict[str, Any], error_message: str) -> AgentRecommendation:
        """Create response for error scenarios."""
        return AgentRecommendation(
            proposed_changes=[],
            next_actions=[
                "Review the action data for completeness",
                "Check API key and connectivity",
                "Try with a different action or simplified data"
            ],
            reasoning=f"Processing error: {error_message}. Please review the action data and try again.",
            confidence_level=ConfidenceLevel.LOW,
            processing_time_ms=0,
            model_version=self.model_name
        )

    def _validate_and_enhance_changes(
        self, 
        changes: List[ProposedChange], 
        current_bmc: BusinessModelCanvas
    ) -> List[ProposedChange]:
        """Validate and enhance proposed changes with safety checks."""
        if not changes:
            return changes

        enhanced_changes = []
        
        for change in changes:
            try:
                # Basic validation
                if not change.proposed_value or not change.reasoning:
                    continue
                    
                # Enhance change with additional metadata
                change.impact_assessment = self._assess_change_impact(change, current_bmc)
                change.risk_factors = self._identify_risk_factors(change)
                change.validation_suggestions = self._generate_validation_suggestions(change)
                
                enhanced_changes.append(change)
            except Exception as e:
                # Skip invalid changes
                continue

        # Apply deduplication
        enhanced_changes = deduplicate_changes(enhanced_changes)
        
        # Sort by confidence score (highest first)
        enhanced_changes.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return enhanced_changes

    def _assess_change_impact(self, change: ProposedChange, current_bmc: BusinessModelCanvas) -> str:
        """Assess the business impact of a proposed change."""
        try:
            # High impact sections
            high_impact_sections = {'value_propositions', 'revenue_streams', 'customer_segments'}
            
            if change.canvas_section in high_impact_sections:
                if change.change_type == ChangeType.REMOVE:
                    return "high"
                return "medium"
            
            if change.change_type == ChangeType.REMOVE:
                return "medium"
            
            return "low"
        except Exception:
            return "medium"  # Default to medium impact if assessment fails

    def _identify_risk_factors(self, change: ProposedChange) -> List[str]:
        """Identify potential risk factors for a change."""
        risks = []
        
        try:
            if change.confidence_score < 0.7:
                risks.append("Moderate confidence level - consider additional validation")
            
            if change.change_type == ChangeType.REMOVE:
                risks.append("Removal operation - ensure no dependencies exist")
            
            if change.canvas_section in ["revenue_streams", "cost_structure"]:
                risks.append("Financial impact - review carefully before implementation")
        except Exception:
            risks.append("Risk assessment failed - manual review recommended")
        
        return risks

    def _generate_validation_suggestions(self, change: ProposedChange) -> List[str]:
        """Generate validation suggestions for a change."""
        section_suggestions = {
            "customer_segments": ["Conduct customer interviews", "Analyze user behavior data"],
            "value_propositions": ["Test with target customers", "Run A/B tests"],
            "channels": ["Pilot test new channels", "Measure channel effectiveness"],
            "revenue_streams": ["Model financial impact", "Test pricing sensitivity"],
        }
        
        return section_suggestions.get(
            change.canvas_section, 
            ["Validate with stakeholders", "Test in small pilot"]
        )

    def _determine_overall_confidence(self, changes: List[ProposedChange]) -> ConfidenceLevel:
        """Determine overall confidence level based on individual change scores."""
        if not changes:
            return ConfidenceLevel.LOW

        try:
            avg_confidence = sum(change.confidence_score for change in changes) / len(changes)

            if avg_confidence >= 0.8:
                return ConfidenceLevel.HIGH
            elif avg_confidence >= 0.6:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        except Exception:
            return ConfidenceLevel.LOW

    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for monitoring and debugging."""
        return {
            "total_processed": len(self.processed_actions),
            "recent_metrics": list(self.processing_metrics.values())[-5:],
            "model_used": self.model_name
        }

    def reset_processed_actions(self) -> None:
        """Reset processed actions cache (useful for testing)."""
        self.processed_actions.clear()
        self.processing_metrics.clear()


class ActionDetectionAgent:
    """Agent for parsing and validating completed action data."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(self, action_data: Dict[str, Any]) -> CompletedAction:
        """Process and validate raw action data."""
        try:
            system_message = SystemMessage(content="""
You are an expert business action analyst. Your task is to analyze raw action data and structure it properly.

Return ONLY a JSON object with this exact structure:
{
    "title": "Clear, descriptive action title",
    "description": "Detailed action description",
    "outcome": "successful|failed|inconclusive",
    "results_data": "Detailed findings and metrics",
    "success_metrics": {},
    "action_category": "Category classification"
}

Focus on extracting meaningful business insights from the provided data.
""")

            human_message = HumanMessage(content=f"""
Analyze this action data and return structured JSON:

Action Data: {json.dumps(action_data, indent=2)}

Return only the JSON object, no other text.
""")

            result = await self.llm.ainvoke([system_message, human_message])
            content = result.content.strip()

            # Extract JSON from response
            json_str = self._extract_json(content)
            parsed_result = json.loads(json_str)

            # Create CompletedAction object
            return CompletedAction(
                title=parsed_result.get("title", action_data.get("title", "Unknown Action")),
                description=parsed_result.get("description", action_data.get("description", "")),
                outcome=ActionOutcome(parsed_result.get("outcome", action_data.get("outcome", "inconclusive"))),
                results_data=parsed_result.get("results_data", action_data.get("results_data", "")),
                success_metrics=parsed_result.get("success_metrics", {}),
                action_category=parsed_result.get("action_category", "General"),
                completion_date=datetime.now()
            )

        except Exception as e:
            # Fallback to basic action creation
            return CompletedAction(
                title=action_data.get("title", "Action Analysis"),
                description=action_data.get("description", "No description provided"),
                outcome=ActionOutcome(action_data.get("outcome", "inconclusive")),
                results_data=action_data.get("results_data", "No results available"),
                success_metrics={},
                action_category="General",
                completion_date=datetime.now()
            )

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        try:
            # Find JSON block
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return content[start:end]
            return content
        except Exception:
            return "{}"


class OutcomeAnalysisAgent:
    """Agent for analyzing business implications of action outcomes."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(
        self,
        action: CompletedAction,
        current_bmc: BusinessModelCanvas
    ) -> Dict[str, Any]:
        """Analyze business implications with contextual intelligence."""
        try:
            system_message = SystemMessage(content="""
You are a senior business strategy consultant specializing in emerging market fintech and business model optimization.

Analyze the completed action and provide strategic insights for business model updates.

Return ONLY a JSON object with this structure:
{
    "summary": "Executive summary of key findings",
    "business_implications": {
        "value_proposition_impact": "Impact on value propositions",
        "customer_impact": "Customer experience implications",
        "revenue_impact": "Revenue and cost implications",
        "operational_impact": "Operational changes needed"
    },
    "bmc_recommendations": {
        "sections_to_update": ["list of BMC sections needing updates"],
        "priority_changes": "Most critical changes needed",
        "evidence_strength": "Strength of evidence supporting changes"
    },
    "confidence_assessment": "Overall confidence in recommendations"
}

Focus on actionable insights for African emerging market context.
""")

            # Prepare BMC context safely
            try:
                bmc_summary = {
                    "customer_segments_count": len(current_bmc.customer_segments),
                    "value_props_count": len(current_bmc.value_propositions),
                    "revenue_streams_count": len(current_bmc.revenue_streams),
                    "completeness": current_bmc.get_completeness_score()
                }
            except Exception:
                bmc_summary = {
                    "customer_segments_count": 0,
                    "value_props_count": 0,
                    "revenue_streams_count": 0,
                    "completeness": 0.0
                }

            human_message = HumanMessage(content=f"""
ANALYZE THIS COMPLETED ACTION:

Action: {action.title}
Category: {action.action_category}
Outcome: {action.outcome.value}
Description: {action.description}
Results: {action.results_data}
Success Metrics: {json.dumps(action.success_metrics)}

CURRENT BMC CONTEXT:
{json.dumps(bmc_summary, indent=2)}

Provide strategic analysis for business model optimization. Return only JSON.
""")

            result = await self.llm.ainvoke([system_message, human_message])
            content = result.content.strip()

            json_str = self._extract_json(content)
            return json.loads(json_str)

        except Exception as e:
            # Fallback analysis
            return {
                "summary": f"Analysis of {action.title} requires manual review due to processing complexity.",
                "business_implications": {
                    "value_proposition_impact": f"Action outcome: {action.outcome.value}",
                    "customer_impact": "Impact assessment needed",
                    "revenue_impact": "Financial review recommended",
                    "operational_impact": "Operational analysis required"
                },
                "bmc_recommendations": {
                    "sections_to_update": [],
                    "priority_changes": "Manual analysis recommended",
                    "evidence_strength": "low"
                },
                "confidence_assessment": "low"
            }

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return content[start:end]
            return content
        except Exception:
            return "{}"


class CanvasUpdateAgent:
    """Agent for generating specific business model canvas updates."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(
        self,
        analysis: Dict[str, Any],
        current_bmc: BusinessModelCanvas,
        action: CompletedAction
    ) -> List[ProposedChange]:
        """Generate specific, validated BMC update recommendations."""
        try:
            system_message = SystemMessage(content="""
You are a business model design expert. Generate specific, evidence-based updates to business model canvas elements.

Return ONLY a JSON object with "proposed_changes" array:
{
    "proposed_changes": [
        {
            "canvas_section": "customer_segments|value_propositions|channels|customer_relationships|revenue_streams|key_resources|key_activities|key_partnerships|cost_structure",
            "change_type": "add|modify|remove",
            "current_value": "existing value being changed (for modify/remove)",
            "proposed_value": "new value to implement",
            "reasoning": "detailed evidence-based explanation (minimum 30 characters)",
            "confidence_score": 0.85
        }
    ]
}

CONFIDENCE SCORING:
- 0.9-1.0: Strong quantitative evidence, clear success metrics
- 0.7-0.8: Good evidence, measurable outcomes  
- 0.5-0.6: Limited evidence, qualitative indicators
- Below 0.5: Weak evidence, speculation

Only propose changes with strong evidence support. Return only JSON.
""")

            # Safely get BMC section counts
            try:
                customer_segments_count = len(current_bmc.customer_segments)
                value_props_count = len(current_bmc.value_propositions)
                channels_count = len(current_bmc.channels)
                revenue_streams_count = len(current_bmc.revenue_streams)
                partnerships_count = len(current_bmc.key_partnerships)
            except Exception:
                customer_segments_count = value_props_count = channels_count = revenue_streams_count = partnerships_count = 0

            human_message = HumanMessage(content=f"""
STRATEGIC ANALYSIS:
{json.dumps(analysis, indent=2)}

CURRENT BMC SECTIONS:
Customer Segments: {customer_segments_count} items
Value Propositions: {value_props_count} items  
Channels: {channels_count} items
Revenue Streams: {revenue_streams_count} items
Key Partnerships: {partnerships_count} items

ACTION CONTEXT:
Title: {action.title}
Outcome: {action.outcome.value}
Results: {action.results_data[:300]}...

Generate evidence-based BMC updates. Return only JSON.
""")

            result = await self.llm.ainvoke([system_message, human_message])
            content = result.content.strip()

            json_str = self._extract_json(content)
            parsed_result = json.loads(json_str)

            return self._create_proposed_changes(parsed_result.get("proposed_changes", []))

        except Exception as e:
            # Return empty changes on error
            return []

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return content[start:end]
            return content
        except Exception:
            return "{}"

    def _create_proposed_changes(self, changes_data: List[Dict[str, Any]]) -> List[ProposedChange]:
        """Create validated ProposedChange objects."""
        validated_changes = []

        for change_data in changes_data:
            try:
                # Validate required fields
                required_fields = ["canvas_section", "change_type", "proposed_value", "reasoning"]
                if not all(field in change_data for field in required_fields):
                    continue

                # Create ProposedChange
                change = ProposedChange(
                    canvas_section=change_data["canvas_section"],
                    change_type=ChangeType(change_data["change_type"]),
                    current_value=change_data.get("current_value"),
                    proposed_value=change_data["proposed_value"],
                    reasoning=change_data["reasoning"],
                    confidence_score=float(change_data.get("confidence_score", 0.5))
                )

                # Validate change quality
                if self._validate_change(change):
                    validated_changes.append(change)

            except (ValueError, KeyError, TypeError):
                continue

        return validated_changes

    def _validate_change(self, change: ProposedChange) -> bool:
        """Validate the quality of a proposed change."""
        try:
            # Check reasoning quality
            if len(change.reasoning) < 20:
                return False

            # Check confidence score range
            if not (0.0 <= change.confidence_score <= 1.0):
                return False

            # Check for meaningful proposed value
            if not change.proposed_value.strip():
                return False

            return True
        except Exception:
            return False


class NextStepAgent:
    """Agent for suggesting intelligent follow-up actions."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(
        self,
        proposed_changes: List[ProposedChange],
        current_bmc: BusinessModelCanvas,
        action: CompletedAction
    ) -> List[str]:
        """Generate intelligent follow-up action recommendations."""
        try:
            system_message = SystemMessage(content="""
You are a startup strategy advisor. Suggest practical next steps based on proposed business model changes.

Return ONLY a JSON object:
{
    "next_actions": [
        "Specific, actionable next step 1",
        "Specific, actionable next step 2", 
        "Specific, actionable next step 3"
    ]
}

Focus on:
- Validation experiments for proposed changes
- Implementation steps with timelines
- Risk mitigation actions
- Measurement and monitoring setup

Return only JSON with 3-5 practical next actions.
""")

            changes_summary = self._summarize_changes(proposed_changes)

            # Safely get completeness score
            try:
                completeness = current_bmc.get_completeness_score()
            except Exception:
                completeness = 0.0

            human_message = HumanMessage(content=f"""
PROPOSED CHANGES SUMMARY:
{changes_summary}

COMPLETED ACTION CONTEXT:
Action: {action.title}
Outcome: {action.outcome.value}
Category: {action.action_category}

BMC COMPLETENESS: {completeness:.0%}

Generate 3-5 specific next actions. Return only JSON.
""")

            result = await self.llm.ainvoke([system_message, human_message])
            content = result.content.strip()

            json_str = self._extract_json(content)
            parsed_result = json.loads(json_str)

            return parsed_result.get("next_actions", self._create_fallback_actions(action))

        except Exception as e:
            return self._create_fallback_actions(action)

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return content[start:end]
            return content
        except Exception:
            return "{}"

    def _summarize_changes(self, changes: List[ProposedChange]) -> str:
        """Create a summary of proposed changes."""
        try:
            if not changes:
                return "No changes proposed"

            summary = f"{len(changes)} proposed changes:\n"
            
            for change in changes[:3]:  # Show top 3
                try:
                    section = change.canvas_section.replace("_", " ").title()
                    summary += f"• {section}: {change.change_type.value} - {change.proposed_value[:50]}...\n"
                except Exception:
                    summary += f"• Change: {change.proposed_value[:50]}...\n"

            return summary
        except Exception:
            return "Changes summary unavailable"

    def _create_fallback_actions(self, action: CompletedAction) -> List[str]:
        """Create fallback actions when AI processing fails."""
        try:
            return [
                f"Review and validate findings from {action.title}",
                "Discuss proposed changes with key stakeholders",
                "Design follow-up experiments to test assumptions",
                "Create implementation timeline for approved changes",
                "Set up monitoring metrics for change impact"
            ]
        except Exception:
            return [
                "Review and validate action findings",
                "Discuss proposed changes with stakeholders",
                "Design follow-up experiments",
                "Create implementation timeline",
                "Set up monitoring metrics"
            ]


# Export main classes
__all__ = [
    'AgenticOrchestrator',
    'ActionDetectionAgent',
    'OutcomeAnalysisAgent', 
    'CanvasUpdateAgent',
    'NextStepAgent'
]