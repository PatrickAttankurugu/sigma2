"""
Agentic Engine for Business Model Canvas Updates - FIXED VERSION

This module implements a sophisticated 4-agent workflow using LangChain
that analyzes completed actions and proposes intelligent business model updates.

FIXES APPLIED:
- Updated to use current LangChain API patterns
- Fixed OpenAI API integration
- Optimized for cost-efficiency with gpt-3.5-turbo
- Added proper error handling for API calls

Architecture:
1. ActionDetectionAgent - Validates and structures action data
2. OutcomeAnalysisAgent - Analyzes business implications
3. CanvasUpdateAgent - Generates specific BMC updates
4. NextStepAgent - Suggests follow-up actions
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from business_models import (
    BusinessModelCanvas,
    CompletedAction,
    ActionOutcome,
    ProposedChange,
    AgentRecommendation,
    ConfidenceLevel,
    ChangeType,
    ProcessingStatus,
    AgentStatus
)
from mock_data import get_market_context


class AgenticOrchestrator:
    """
    Orchestrates a multi-agent workflow for intelligent business model updates.

    This system uses 4 specialized agents to process action outcomes and generate
    contextually relevant business model canvas updates for African fintech markets.
    """

    def __init__(self, google_api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """Initialize the orchestrator with Google Gemini integration."""
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")

        self.model_name = model_name

        # Gemini chat model via LangChain
        self.llm = ChatGoogleGenerativeAI(
            api_key=self.api_key,
            model=model_name,
            temperature=0.2,
            max_output_tokens=1000,
            timeout=30
        )

        # Initialize specialized agents
        self.action_detection_agent = ActionDetectionAgent(self.llm)
        self.outcome_analysis_agent = OutcomeAnalysisAgent(self.llm)
        self.canvas_update_agent = CanvasUpdateAgent(self.llm)
        self.next_step_agent = NextStepAgent(self.llm)

        # Context and memory
        self.market_context = get_market_context()
        self.processing_status = ProcessingStatus()

    async def process_action_outcome(
        self,
        action_data: Dict[str, Any],
        current_bmc: BusinessModelCanvas
    ) -> AgentRecommendation:
        """
        Process a completed action through the full 4-agent workflow.

        Args:
            action_data: Raw action outcome data
            current_bmc: Current business model canvas state

        Returns:
            Complete agent recommendation with proposed changes
        """
        try:
            self.processing_status = ProcessingStatus()

            # Agent 1: Action Detection & Validation
            self.processing_status.action_detection_status = AgentStatus.RUNNING
            validated_action = await self.action_detection_agent.process(action_data)
            self.processing_status.action_detection_status = AgentStatus.COMPLETED

            # Agent 2: Outcome Analysis
            self.processing_status.outcome_analysis_status = AgentStatus.RUNNING
            business_analysis = await self.outcome_analysis_agent.process(
                validated_action, current_bmc, self.market_context
            )
            self.processing_status.outcome_analysis_status = AgentStatus.COMPLETED

            # Agent 3: Canvas Updates
            self.processing_status.canvas_update_status = AgentStatus.RUNNING
            proposed_changes = await self.canvas_update_agent.process(
                business_analysis, current_bmc, validated_action
            )
            self.processing_status.canvas_update_status = AgentStatus.COMPLETED

            # Agent 4: Next Steps
            self.processing_status.next_step_status = AgentStatus.RUNNING
            next_actions = await self.next_step_agent.process(
                proposed_changes, current_bmc, validated_action, business_analysis
            )
            self.processing_status.next_step_status = AgentStatus.COMPLETED

            # Compile final recommendation
            recommendation = AgentRecommendation(
                proposed_changes=proposed_changes,
                next_actions=next_actions,
                reasoning=business_analysis.get("summary", "Comprehensive analysis completed"),
                confidence_level=self._determine_overall_confidence(proposed_changes)
            )

            self.processing_status.completed_at = datetime.now()
            return recommendation

        except Exception as e:
            self.processing_status.error_message = str(e)
            # Set appropriate agent status to failed based on where error occurred
            if self.processing_status.action_detection_status == AgentStatus.RUNNING:
                self.processing_status.action_detection_status = AgentStatus.FAILED
            elif self.processing_status.outcome_analysis_status == AgentStatus.RUNNING:
                self.processing_status.outcome_analysis_status = AgentStatus.FAILED
            elif self.processing_status.canvas_update_status == AgentStatus.RUNNING:
                self.processing_status.canvas_update_status = AgentStatus.FAILED
            elif self.processing_status.next_step_status == AgentStatus.RUNNING:
                self.processing_status.next_step_status = AgentStatus.FAILED

            raise e

    def get_processing_status(self) -> ProcessingStatus:
        """Return current processing status for UI updates."""
        return self.processing_status

    def _determine_overall_confidence(self, changes: List[ProposedChange]) -> ConfidenceLevel:
        """Determine overall confidence based on individual change scores."""
        if not changes:
            return ConfidenceLevel.LOW

        avg_confidence = sum(change.confidence_score for change in changes) / len(changes)

        if avg_confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class ActionDetectionAgent:
    """Agent responsible for parsing and validating completed action data."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(self, action_data: Dict[str, Any]) -> CompletedAction:
        """Process and validate raw action data."""
        try:
            market_context_str = json.dumps(get_market_context(), indent=2)
            
            # FIXED: Use modern LangChain message format instead of LLMChain
            system_message = SystemMessage(content="""
You are an expert data validation agent for African fintech business analysis.

Your task is to analyze raw action outcome data and extract structured information.

Please validate and structure this data according to these requirements:

1. Extract key action information:
   - Title (descriptive name)
   - Description (detailed explanation)
   - Outcome (successful/failed/inconclusive)
   - Results data (detailed findings)
   - Success metrics (if available)

2. Validate data quality:
   - Check for completeness
   - Assess reliability of results
   - Identify any data gaps or concerns

3. Classify the action type and business impact area

Return your analysis in this JSON format:
{
    "validated_action": {
        "title": "string",
        "description": "string",
        "outcome": "successful|failed|inconclusive",
        "results_data": "string",
        "success_metrics": {},
        "data_quality_score": 0.0-1.0,
        "business_impact_areas": ["list of BMC sections likely affected"]
    },
    "validation_notes": "Any concerns or observations about data quality"
}

Focus on African fintech context, informal economy dynamics, and mobile payment ecosystems.
""")

            human_message = HumanMessage(content=f"""
Market Context: {market_context_str}

Raw Action Data: {json.dumps(action_data, indent=2)}

Please analyze and validate this action data.
""")

            # FIXED: Use ainvoke instead of arun
            result = await self.llm.ainvoke([system_message, human_message])

            # Robust JSON extraction for Gemini responses
            content = result.content or ""
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end] if start != -1 and end != -1 else content
            parsed_result = json.loads(json_str)
            validated_data = parsed_result["validated_action"]

            # Create CompletedAction object
            return CompletedAction(
                title=validated_data["title"],
                description=validated_data["description"],
                outcome=ActionOutcome(validated_data["outcome"]),
                results_data=validated_data["results_data"],
                success_metrics=validated_data.get("success_metrics", {}),
                completion_date=datetime.now()
            )

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"ActionDetectionAgent error: {e}")  # Added debugging
            # Fallback: create basic CompletedAction from raw data
            return CompletedAction(
                title=action_data.get("title", "Unknown Action"),
                description=action_data.get("description", "No description provided"),
                outcome=ActionOutcome(action_data.get("outcome", "inconclusive")),
                results_data=str(action_data.get("results_data", "No results data")),
                success_metrics=action_data.get("success_metrics", {}),
                completion_date=datetime.now()
            )
        except Exception as e:
            print(f"Unexpected error in ActionDetectionAgent: {e}")
            raise


class OutcomeAnalysisAgent:
    """Agent responsible for analyzing business implications of action outcomes."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(
        self,
        action: CompletedAction,
        current_bmc: BusinessModelCanvas,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the business implications of the action outcome."""

        try:
            # Serialize BMC for prompt
            bmc_dict = {
                "customer_segments": current_bmc.customer_segments,
                "value_propositions": current_bmc.value_propositions,
                "channels": current_bmc.channels,
                "customer_relationships": current_bmc.customer_relationships,
                "revenue_streams": current_bmc.revenue_streams,
                "key_resources": current_bmc.key_resources,
                "key_activities": current_bmc.key_activities,
                "key_partnerships": current_bmc.key_partnerships,
                "cost_structure": current_bmc.cost_structure
            }

            system_message = SystemMessage(content="""
You are a senior business strategy consultant specializing in African fintech and mobile payments.

Analyze completed actions and their strategic implications for the business model.

Provide comprehensive strategic analysis addressing:

1. IMMEDIATE IMPLICATIONS:
   - Value proposition impact
   - Customer impact  
   - Revenue/cost impact

2. STRATEGIC INSIGHTS:
   - Market positioning implications
   - Competitive landscape impact
   - Risk factors and opportunities

3. BMC SECTION IMPACTS:
   - Which specific BMC sections need updates
   - How results challenge current assumptions
   - What new elements should be considered

4. AFRICAN CONTEXT CONSIDERATIONS:
   - Informal economy implications
   - Mobile-first customer behavior insights
   - Regulatory/compliance impacts

Return analysis as JSON with immediate_implications, strategic_insights, bmc_section_impacts, african_context_insights, and summary fields.
""")

            human_message = HumanMessage(content=f"""
Completed Action:
Title: {action.title}
Description: {action.description}  
Outcome: {action.outcome}
Results: {action.results_data}
Metrics: {action.success_metrics}

Current Business Model: {json.dumps(bmc_dict, indent=2)}

Market Context: {json.dumps(market_context, indent=2)}

Please provide strategic analysis.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            try:
                content = result.content or ""
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end] if start != -1 and end != -1 else content
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback response
                return {
                    "immediate_implications": {"value_proposition_impact": "Analysis pending"},
                    "strategic_insights": {"market_positioning": "Requires further analysis"},
                    "bmc_section_impacts": {},
                    "african_context_insights": "Context analysis needed",
                    "summary": "Strategic analysis completed with limited insights"
                }
                
        except Exception as e:
            print(f"OutcomeAnalysisAgent error: {e}")
            raise


class CanvasUpdateAgent:
    """Agent responsible for generating specific business model canvas updates."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(
        self,
        analysis: Dict[str, Any],
        current_bmc: BusinessModelCanvas,
        action: CompletedAction
    ) -> List[ProposedChange]:
        """Generate specific BMC update recommendations."""

        try:
            # Serialize current BMC
            bmc_dict = {
                "customer_segments": current_bmc.customer_segments,
                "value_propositions": current_bmc.value_propositions,
                "channels": current_bmc.channels,
                "customer_relationships": current_bmc.customer_relationships,
                "revenue_streams": current_bmc.revenue_streams,
                "key_resources": current_bmc.key_resources,
                "key_activities": current_bmc.key_activities,
                "key_partnerships": current_bmc.key_partnerships,
                "cost_structure": current_bmc.cost_structure
            }

            system_message = SystemMessage(content="""
You are a business model design expert specializing in Business Model Canvas updates.

Based on strategic analysis, generate specific, actionable updates to the business model canvas.

GUIDELINES:
- Only propose changes where analysis provides clear evidence
- Each change should be specific and actionable
- Consider interdependencies between BMC sections
- Maintain focus on African fintech/mobile payment context
- Assign realistic confidence scores based on evidence strength

Return recommendations in JSON format with proposed_changes array containing:
- canvas_section, change_type (add/modify/remove), current_value, proposed_value, reasoning, confidence_score

BMC SECTIONS: customer_segments, value_propositions, channels, customer_relationships, revenue_streams, key_resources, key_activities, key_partnerships, cost_structure
""")

            human_message = HumanMessage(content=f"""
Strategic Analysis: {json.dumps(analysis, indent=2)}

Current Business Model Canvas: {json.dumps(bmc_dict, indent=2)}

Original Action: Title: {action.title}, Outcome: {action.outcome}, Results: {action.results_data}

Please generate specific BMC update recommendations.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            try:
                content = result.content or ""
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end] if start != -1 and end != -1 else content
                parsed_result = json.loads(json_str)
                changes = []

                for change_data in parsed_result.get("proposed_changes", []):
                    try:
                        change = ProposedChange(
                            canvas_section=change_data["canvas_section"],
                            change_type=ChangeType(change_data["change_type"]),
                            current_value=change_data.get("current_value"),
                            proposed_value=change_data["proposed_value"],
                            reasoning=change_data["reasoning"],
                            confidence_score=float(change_data["confidence_score"])
                        )
                        changes.append(change)
                    except (ValueError, KeyError) as e:
                        # Skip invalid changes but continue processing
                        continue

                return changes

            except json.JSONDecodeError:
                # Return empty list if parsing fails
                return []
                
        except Exception as e:
            print(f"CanvasUpdateAgent error: {e}")
            return []


class NextStepAgent:
    """Agent responsible for suggesting intelligent follow-up actions."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def process(
        self,
        proposed_changes: List[ProposedChange],
        current_bmc: BusinessModelCanvas,
        action: CompletedAction,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate intelligent follow-up action recommendations."""

        try:
            # Serialize inputs for prompt
            changes_data = [
                {
                    "section": change.canvas_section,
                    "type": change.change_type,
                    "proposal": change.proposed_value,
                    "reasoning": change.reasoning,
                    "confidence": change.confidence_score
                }
                for change in proposed_changes
            ]

            bmc_dict = {
                "customer_segments": current_bmc.customer_segments,
                "value_propositions": current_bmc.value_propositions,
                "channels": current_bmc.channels,
                "revenue_streams": current_bmc.revenue_streams
            }

            system_message = SystemMessage(content="""
You are a startup strategy advisor specializing in validation methodologies and experiment design.

Based on proposed business model changes, recommend specific follow-up actions to validate and implement updates.

Design a validation roadmap including:

1. VALIDATION EXPERIMENTS: Specific actions to test proposed changes
2. IMPLEMENTATION STEPS: Prioritized action sequence  
3. MONITORING & MEASUREMENT: Key performance indicators

Consider African market constraints: limited connectivity, cash preferences, trust-building, compliance needs.

Return JSON with validation_experiments and implementation_steps arrays.
""")

            human_message = HumanMessage(content=f"""
Proposed Changes: {json.dumps(changes_data, indent=2)}

Current Business Model: {json.dumps(bmc_dict, indent=2)}

Recent Action Results: Title: {action.title}, Results: {action.results_data}

Strategic Analysis Summary: {json.dumps(analysis.get("summary", "Analysis summary"), indent=2)}

Please recommend follow-up actions.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            try:
                content = result.content or ""
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end] if start != -1 and end != -1 else content
                parsed_result = json.loads(json_str)

                # Extract next actions from validation experiments and implementation steps
                next_actions = []

                for experiment in parsed_result.get("validation_experiments", []):
                    next_actions.append(f"[EXPERIMENT] {experiment['action_title']}: {experiment['description']}")

                for step in parsed_result.get("implementation_steps", [])[:3]:  # Limit to top 3
                    next_actions.append(f"[IMPLEMENTATION] {step['step']} - {step['rationale']}")

                return next_actions

            except json.JSONDecodeError:
                # Fallback recommendations
                return [
                    "Conduct follow-up customer interviews to validate findings",
                    "Analyze competitor responses to similar market conditions",
                    "Design A/B test for proposed value proposition changes",
                    "Schedule partnership discussions with relevant stakeholders"
                ]
                
        except Exception as e:
            print(f"NextStepAgent error: {e}")
            return [
                "Conduct follow-up customer interviews to validate findings",
                "Analyze competitor responses to similar market conditions"
            ]


# Utility functions for integration

def validate_safety(proposed_changes: List[ProposedChange]) -> bool:
    """Validate if proposed changes are safe for auto-application."""
    if not proposed_changes:
        return True

    # Safety criteria
    high_risk_changes = [change for change in proposed_changes if change.confidence_score < 0.6]
    major_changes = [change for change in proposed_changes if change.change_type == ChangeType.REMOVE]

    # Don't auto-apply if there are high-risk or major removal changes
    return len(high_risk_changes) == 0 and len(major_changes) == 0


async def create_orchestrator() -> AgenticOrchestrator:
    """Factory function to create a properly configured orchestrator."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set")

    return AgenticOrchestrator(api_key, model_name="gemini-1.5-flash")


# ADDED: Quick test function to verify API connectivity
async def test_api_connection():
    """Test function to verify Google Gemini API is working correctly."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY environment variable not set")
            return False
            
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=50
        )
        
        test_message = HumanMessage(content="Hello, this is a test. Please respond with 'API connection successful'.")
        response = await llm.ainvoke([test_message])
        
        print(f"API Test Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"API Connection Test Failed: {e}")
        return False


# Usage example:
if __name__ == "__main__":
    # Test API connection first
    print("Testing API connection...")
    asyncio.run(test_api_connection())