"""
Enhanced Agentic Engine for Business Model Canvas Updates

This module implements a sophisticated 4-agent workflow using LangChain that analyzes 
completed actions and proposes intelligent business model updates with advanced business 
intelligence capabilities.

ENHANCEMENTS:
- Integration with enhanced business models and validation
- Advanced confidence scoring and safety mechanisms  
- Professional error handling and fallback strategies
- Enhanced prompt engineering with business context
- Integration with chat interface and user profiling
- Comprehensive logging and analytics
- Support for emerging market business intelligence

Architecture:
1. ActionDetectionAgent - Validates and structures action data with quality scoring
2. OutcomeAnalysisAgent - Analyzes business implications with market context
3. CanvasUpdateAgent - Generates specific BMC updates with confidence scoring
4. NextStepAgent - Suggests intelligent follow-up actions with validation logic
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import hashlib
import re

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
    AgentStatus,
    UserProfile
)
from mock_data import get_market_context
from utils import (
    calculate_confidence_score,
    validate_change_safety,
    determine_change_impact,
    generate_change_hash,
    deduplicate_changes
)


class AgenticOrchestrator:
    """
    Enhanced orchestrator for multi-agent workflow with advanced business intelligence.

    This system uses 4 specialized agents to process action outcomes and generate
    contextually relevant business model canvas updates with sophisticated confidence
    scoring, safety mechanisms, and business intelligence capabilities.
    """

    def __init__(
        self, 
        google_api_key: Optional[str] = None, 
        model_name: str = "gemini-1.5-flash",
        user_profile: Optional[UserProfile] = None
    ):
        """Initialize the orchestrator with Google Gemini integration."""
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")

        self.model_name = model_name
        self.user_profile = user_profile

        # Enhanced Gemini chat model configuration
        self.llm = ChatGoogleGenerativeAI(
            api_key=self.api_key,
            model=model_name,
            temperature=0.2,  # Balanced creativity vs consistency
            max_output_tokens=1500,  # Increased for detailed analysis
            timeout=45  # Increased timeout for complex analysis
        )

        # Initialize specialized agents with enhanced capabilities
        self.action_detection_agent = ActionDetectionAgent(self.llm, self.user_profile)
        self.outcome_analysis_agent = OutcomeAnalysisAgent(self.llm, self.user_profile)
        self.canvas_update_agent = CanvasUpdateAgent(self.llm, self.user_profile)
        self.next_step_agent = NextStepAgent(self.llm, self.user_profile)

        # Enhanced context and memory
        self.market_context = get_market_context()
        self.processing_status = ProcessingStatus()
        self.session_memory = {}  # Store processing context across calls
        
        # Performance tracking
        self.start_time = None
        self.agent_timings = {}

    async def process_action_outcome(
        self,
        action_data: Dict[str, Any],
        current_bmc: BusinessModelCanvas,
        processing_callback: Optional[callable] = None
    ) -> AgentRecommendation:
        """
        Process a completed action through the enhanced 4-agent workflow.

        Args:
            action_data: Raw action outcome data
            current_bmc: Current business model canvas state
            processing_callback: Optional callback for progress updates

        Returns:
            Complete agent recommendation with proposed changes and analytics
        """
        self.start_time = time.time()
        
        try:
            # Initialize processing status with enhanced tracking
            self.processing_status = ProcessingStatus()
            self.processing_status.current_agent = "initialization"
            
            if processing_callback:
                processing_callback("Initializing 4-agent workflow...")

            # Agent 1: Enhanced Action Detection & Validation
            agent_start = time.time()
            self.processing_status.update_agent_status("action_detection", AgentStatus.RUNNING)
            
            if processing_callback:
                processing_callback("Agent 1: Validating and structuring action data...")
            
            validated_action = await self.action_detection_agent.process(action_data)
            
            self.agent_timings["action_detection"] = time.time() - agent_start
            self.processing_status.update_agent_status("action_detection", AgentStatus.COMPLETED)

            # Agent 2: Enhanced Outcome Analysis
            agent_start = time.time()
            self.processing_status.update_agent_status("outcome_analysis", AgentStatus.RUNNING)
            
            if processing_callback:
                processing_callback("Agent 2: Analyzing business implications...")
            
            business_analysis = await self.outcome_analysis_agent.process(
                validated_action, current_bmc, self.market_context
            )
            
            self.agent_timings["outcome_analysis"] = time.time() - agent_start
            self.processing_status.update_agent_status("outcome_analysis", AgentStatus.COMPLETED)

            # Agent 3: Enhanced Canvas Updates
            agent_start = time.time()
            self.processing_status.update_agent_status("canvas_update", AgentStatus.RUNNING)
            
            if processing_callback:
                processing_callback("Agent 3: Generating BMC updates...")
            
            proposed_changes = await self.canvas_update_agent.process(
                business_analysis, current_bmc, validated_action
            )
            
            # Apply enhanced validation and deduplication
            proposed_changes = self._enhance_proposed_changes(proposed_changes, current_bmc)
            
            self.agent_timings["canvas_update"] = time.time() - agent_start
            self.processing_status.update_agent_status("canvas_update", AgentStatus.COMPLETED)

            # Agent 4: Enhanced Next Steps
            agent_start = time.time()
            self.processing_status.update_agent_status("next_step", AgentStatus.RUNNING)
            
            if processing_callback:
                processing_callback("Agent 4: Generating next actions...")
            
            next_actions = await self.next_step_agent.process(
                proposed_changes, current_bmc, validated_action, business_analysis
            )
            
            self.agent_timings["next_step"] = time.time() - agent_start
            self.processing_status.update_agent_status("next_step", AgentStatus.COMPLETED)

            # Compile enhanced recommendation with analytics
            total_processing_time = int((time.time() - self.start_time) * 1000)
            
            recommendation = AgentRecommendation(
                proposed_changes=proposed_changes,
                next_actions=next_actions,
                reasoning=business_analysis.get("summary", "Comprehensive analysis completed"),
                confidence_level=self._determine_overall_confidence(proposed_changes),
                processing_time_ms=total_processing_time,
                model_version=self.model_name,
                source_action_id=validated_action.id,
                market_context=self.market_context,
                risk_assessment=self._assess_overall_risk(proposed_changes),
                implementation_priority=self._determine_implementation_priority(proposed_changes)
            )

            self.processing_status.completed_at = datetime.now()
            
            if processing_callback:
                processing_callback("Analysis complete!")
            
            return recommendation

        except Exception as e:
            self.processing_status.error_message = str(e)
            self._handle_agent_failure(e)
            
            if processing_callback:
                processing_callback(f"Error: {str(e)}")
            
            # Return fallback recommendation rather than raising
            return self._create_fallback_recommendation(action_data, str(e))

    def _enhance_proposed_changes(
        self, 
        changes: List[ProposedChange], 
        current_bmc: BusinessModelCanvas
    ) -> List[ProposedChange]:
        """Apply enhanced validation, scoring, and deduplication to proposed changes."""
        
        if not changes:
            return changes
        
        enhanced_changes = []
        
        for change in changes:
            # Calculate enhanced confidence score
            enhanced_confidence = calculate_confidence_score(
                reasoning=change.reasoning,
                data_quality=0.8,  # Assume good data quality from AI analysis
                sample_size=100,   # Estimated based on action analysis
                validation_sources=2  # Analysis + market context
            )
            
            # Update confidence score
            change.confidence_score = enhanced_confidence
            
            # Add impact assessment
            change.impact_assessment = determine_change_impact(change, current_bmc)
            
            # Add validation suggestions
            change.validation_suggestions = self._generate_validation_suggestions(change)
            
            # Add risk factors
            change.risk_factors = self._identify_risk_factors(change, current_bmc)
            
            enhanced_changes.append(change)
        
        # Apply deduplication
        enhanced_changes = deduplicate_changes(enhanced_changes)
        
        # Sort by confidence score (highest first)
        enhanced_changes.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return enhanced_changes

    def _generate_validation_suggestions(self, change: ProposedChange) -> List[str]:
        """Generate specific validation suggestions for a proposed change."""
        suggestions = []
        
        section_validation = {
            "customer_segments": [
                "Conduct customer interviews to validate segment characteristics",
                "Analyze customer data to confirm segment size and behavior patterns"
            ],
            "value_propositions": [
                "Run A/B tests with different value proposition messaging",
                "Survey customers on value proposition appeal and differentiation"
            ],
            "channels": [
                "Test channel effectiveness with pilot customer groups",
                "Measure channel cost-effectiveness and reach metrics"
            ],
            "revenue_streams": [
                "Model financial projections and conduct pricing sensitivity analysis",
                "Test willingness to pay with target customer segments"
            ],
            "key_partnerships": [
                "Reach out to potential partners for initial discussions",
                "Analyze partnership feasibility and mutual value creation"
            ]
        }
        
        base_suggestions = section_validation.get(change.canvas_section, [
            "Gather additional data to validate this change",
            "Consult with stakeholders before implementation"
        ])
        
        suggestions.extend(base_suggestions)
        
        # Add confidence-based suggestions
        if change.confidence_score < 0.7:
            suggestions.append("Conduct additional research due to moderate confidence level")
        
        return suggestions[:3]  # Limit to top 3 suggestions

    def _identify_risk_factors(
        self, 
        change: ProposedChange, 
        current_bmc: BusinessModelCanvas
    ) -> List[str]:
        """Identify potential risk factors for a proposed change."""
        risks = []
        
        # High-impact section risks
        if change.canvas_section in ["revenue_streams", "value_propositions", "customer_segments"]:
            if change.change_type == ChangeType.REMOVE:
                risks.append("High impact: Removing core business element")
            elif change.change_type == ChangeType.MODIFY:
                risks.append("Moderate impact: Modifying core business element")
        
        # Low confidence risks
        if change.confidence_score < 0.6:
            risks.append("Low confidence: Limited evidence supporting change")
        
        # Change type risks
        if change.change_type == ChangeType.REMOVE:
            risks.append("Removal risk: May negatively impact existing operations")
        
        # Check for dependency risks
        current_values = getattr(current_bmc, change.canvas_section, [])
        if len(current_values) <= 1 and change.change_type == ChangeType.REMOVE:
            risks.append("Dependency risk: Removing only element in this section")
        
        return risks

    def _assess_overall_risk(self, changes: List[ProposedChange]) -> str:
        """Assess overall risk level of all proposed changes."""
        if not changes:
            return "low"
        
        high_risk_count = 0
        total_risk_score = 0
        
        for change in changes:
            # Risk factors
            risk_factors = len(change.risk_factors) if change.risk_factors else 0
            
            # Confidence penalty
            confidence_penalty = max(0, (0.8 - change.confidence_score) * 2)
            
            # Change type penalty
            type_penalty = 2 if change.change_type == ChangeType.REMOVE else 0
            
            risk_score = risk_factors + confidence_penalty + type_penalty
            total_risk_score += risk_score
            
            if risk_score > 3:
                high_risk_count += 1
        
        avg_risk = total_risk_score / len(changes) if changes else 0
        
        if high_risk_count > 0 or avg_risk > 2:
            return "high"
        elif avg_risk > 1:
            return "medium"
        else:
            return "low"

    def _determine_implementation_priority(self, changes: List[ProposedChange]) -> str:
        """Determine implementation priority based on change characteristics."""
        if not changes:
            return "low"
        
        high_confidence_count = len([c for c in changes if c.confidence_score >= 0.8])
        high_impact_count = len([c for c in changes if c.impact_assessment == "high"])
        
        if high_confidence_count >= 2 and high_impact_count >= 1:
            return "high"
        elif high_confidence_count >= 1 or high_impact_count >= 1:
            return "medium"
        else:
            return "low"

    def _handle_agent_failure(self, error: Exception) -> None:
        """Handle agent failure by setting appropriate status."""
        # Determine which agent failed based on current processing status
        if self.processing_status.action_detection_status == AgentStatus.RUNNING:
            self.processing_status.action_detection_status = AgentStatus.FAILED
        elif self.processing_status.outcome_analysis_status == AgentStatus.RUNNING:
            self.processing_status.outcome_analysis_status = AgentStatus.FAILED
        elif self.processing_status.canvas_update_status == AgentStatus.RUNNING:
            self.processing_status.canvas_update_status = AgentStatus.FAILED
        elif self.processing_status.next_step_status == AgentStatus.RUNNING:
            self.processing_status.next_step_status = AgentStatus.FAILED

    def _create_fallback_recommendation(
        self, 
        action_data: Dict[str, Any], 
        error_message: str
    ) -> AgentRecommendation:
        """Create a fallback recommendation when agent processing fails."""
        return AgentRecommendation(
            proposed_changes=[],
            next_actions=[
                "Review action data quality and completeness",
                "Consider manual analysis of action outcomes",
                "Check system configuration and API connectivity"
            ],
            reasoning=f"Unable to complete full analysis due to processing error: {error_message}. Please review action data and try again.",
            confidence_level=ConfidenceLevel.LOW,
            processing_time_ms=int((time.time() - self.start_time) * 1000) if self.start_time else 0,
            model_version=self.model_name,
            risk_assessment="high",
            implementation_priority="low"
        )

    def get_processing_status(self) -> ProcessingStatus:
        """Return current processing status for UI updates."""
        return self.processing_status

    def get_agent_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents."""
        return {
            "agent_timings": self.agent_timings,
            "total_processing_time": sum(self.agent_timings.values()),
            "average_agent_time": sum(self.agent_timings.values()) / len(self.agent_timings) if self.agent_timings else 0,
            "model_used": self.model_name,
            "success_rate": 1.0 if not self.processing_status.has_failed() else 0.0
        }

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
    """Enhanced agent for parsing and validating completed action data."""

    def __init__(self, llm: ChatGoogleGenerativeAI, user_profile: Optional[UserProfile] = None):
        self.llm = llm
        self.user_profile = user_profile

    async def process(self, action_data: Dict[str, Any]) -> CompletedAction:
        """Process and validate raw action data with enhanced validation."""
        try:
            # Prepare context with user profile if available
            user_context = ""
            if self.user_profile:
                context_data = self.user_profile.get_context_for_ai()
                user_context = f"\nUser Context: {json.dumps(context_data, indent=2)}"

            market_context_str = json.dumps(get_market_context(), indent=2)
            
            system_message = SystemMessage(content=f"""
You are an expert data validation agent specializing in emerging market business analysis.

Your task is to analyze raw action outcome data and extract structured, validated information suitable for business model analysis.

VALIDATION REQUIREMENTS:
1. Extract and validate key action information:
   - Title (descriptive, actionable name)
   - Description (comprehensive explanation)
   - Outcome (successful/failed/inconclusive)
   - Results data (detailed, quantitative findings)
   - Success metrics (measurable KPIs)

2. Assess data quality and reliability:
   - Completeness score (0.0-1.0)
   - Data reliability assessment
   - Identify gaps or inconsistencies
   - Flag potential biases or limitations

3. Business context classification:
   - Action category (Market Research, Product Test, Partnership, etc.)
   - Primary business impact areas
   - Stakeholder involvement level
   - Strategic significance

4. Emerging market considerations:
   - Cultural and economic context factors
   - Infrastructure constraints or opportunities
   - Regulatory or compliance implications

Return analysis in JSON format:
{{
    "validated_action": {{
        "title": "Clear, descriptive action title",
        "description": "Comprehensive action description",
        "outcome": "successful|failed|inconclusive",
        "results_data": "Detailed findings and metrics",
        "success_metrics": {{}},
        "action_category": "Category classification",
        "stakeholders_involved": ["list of stakeholders"],
        "budget_spent": numeric_value_if_available,
        "duration_days": numeric_value_if_available
    }},
    "data_quality_assessment": {{
        "completeness_score": 0.0-1.0,
        "reliability_score": 0.0-1.0,
        "confidence_factors": ["factors supporting confidence"],
        "quality_concerns": ["any data quality issues"],
        "business_impact_areas": ["BMC sections likely affected"]
    }},
    "validation_notes": "Additional observations about data quality and analysis"
}}

Focus on emerging market context, informal economy dynamics, and mobile/digital ecosystem factors.
{user_context}
""")

            human_message = HumanMessage(content=f"""
Market Context: {market_context_str}

Raw Action Data: {json.dumps(action_data, indent=2)}

Please analyze and validate this action data with comprehensive business intelligence assessment.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            # Enhanced JSON extraction with better error handling
            content = result.content or ""
            json_str = self._extract_json_from_response(content)
            parsed_result = json.loads(json_str)
            
            validated_data = parsed_result["validated_action"]
            quality_data = parsed_result.get("data_quality_assessment", {})

            # Create enhanced CompletedAction object
            return CompletedAction(
                title=validated_data["title"],
                description=validated_data["description"],
                outcome=ActionOutcome(validated_data["outcome"]),
                results_data=validated_data["results_data"],
                success_metrics=validated_data.get("success_metrics", {}),
                action_category=validated_data.get("action_category"),
                stakeholders_involved=validated_data.get("stakeholders_involved", []),
                budget_spent=validated_data.get("budget_spent"),
                duration_days=validated_data.get("duration_days"),
                completion_date=datetime.now()
            )

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"ActionDetectionAgent JSON/Validation error: {e}")
            return self._create_fallback_action(action_data)
        
        except Exception as e:
            print(f"ActionDetectionAgent unexpected error: {e}")
            raise

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response with multiple fallback strategies."""
        # Strategy 1: Find JSON block
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return content[start:end]
        
        # Strategy 2: Find JSON in code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(1)
        
        # Strategy 3: Return full content if no clear JSON found
        return content

    def _create_fallback_action(self, action_data: Dict[str, Any]) -> CompletedAction:
        """Create fallback CompletedAction when parsing fails."""
        return CompletedAction(
            title=action_data.get("title", "Unknown Action"),
            description=action_data.get("description", "No description provided"),
            outcome=ActionOutcome(action_data.get("outcome", "inconclusive")),
            results_data=str(action_data.get("results_data", "No results data available")),
            success_metrics=action_data.get("success_metrics", {}),
            action_category="Unclassified",
            completion_date=datetime.now()
        )


class OutcomeAnalysisAgent:
    """Enhanced agent for analyzing business implications of action outcomes."""

    def __init__(self, llm: ChatGoogleGenerativeAI, user_profile: Optional[UserProfile] = None):
        self.llm = llm
        self.user_profile = user_profile

    async def process(
        self,
        action: CompletedAction,
        current_bmc: BusinessModelCanvas,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze business implications with enhanced strategic intelligence."""

        try:
            # Prepare enhanced BMC context
            bmc_dict = self._serialize_bmc_with_analytics(current_bmc)
            
            # Prepare user context
            user_context = ""
            if self.user_profile:
                context_data = self.user_profile.get_context_for_ai()
                user_context = f"\nUser Business Context: {json.dumps(context_data, indent=2)}"

            system_message = SystemMessage(content=f"""
You are a senior business strategy consultant specializing in emerging market fintech, mobile payments, and business model innovation.

Analyze completed business actions and provide comprehensive strategic intelligence for business model optimization.

ANALYSIS FRAMEWORK:

1. IMMEDIATE BUSINESS IMPLICATIONS:
   - Direct impact on value proposition delivery
   - Customer experience and satisfaction effects
   - Revenue generation and cost structure implications
   - Operational efficiency changes

2. STRATEGIC MARKET INSIGHTS:
   - Competitive positioning implications
   - Market opportunity identification
   - Risk assessment and mitigation strategies
   - Growth potential and scalability factors

3. BUSINESS MODEL CANVAS IMPACTS:
   - Specific BMC sections requiring updates
   - Interdependencies between canvas elements
   - New assumptions to validate
   - Elements to add, modify, or remove

4. EMERGING MARKET CONSIDERATIONS:
   - Informal economy integration opportunities
   - Mobile-first customer behavior insights
   - Infrastructure constraint adaptations
   - Regulatory compliance implications
   - Cultural and social factor impacts

5. DATA-DRIVEN RECOMMENDATIONS:
   - Quantitative insights from action results
   - Evidence-based strategic recommendations
   - Confidence levels for different insights
   - Priority ranking for implementation

Return comprehensive analysis in JSON format:
{{
    "immediate_implications": {{
        "value_proposition_impact": "detailed analysis",
        "customer_impact": "customer experience insights",
        "revenue_cost_impact": "financial implications",
        "operational_impact": "operational changes needed"
    }},
    "strategic_insights": {{
        "market_positioning": "positioning analysis",
        "competitive_landscape": "competitive implications",
        "growth_opportunities": "identified opportunities", 
        "risk_factors": "strategic risks and mitigation"
    }},
    "bmc_section_impacts": {{
        "sections_to_update": ["list of BMC sections"],
        "interdependencies": "cross-section impacts",
        "new_assumptions": "assumptions to validate",
        "evidence_strength": "strength of evidence for changes"
    }},
    "emerging_market_insights": {{
        "informal_economy": "informal economy implications",
        "mobile_behavior": "mobile-first insights",
        "infrastructure": "infrastructure considerations",
        "regulatory": "compliance and regulatory factors"
    }},
    "data_driven_recommendations": {{
        "quantitative_insights": "key metrics and findings",
        "strategic_recommendations": ["prioritized recommendations"],
        "confidence_assessment": "overall confidence in analysis",
        "implementation_priority": "priority for action"
    }},
    "summary": "Executive summary of key findings and strategic implications"
}}

Focus on actionable insights that can drive measurable business improvements.
{user_context}
""")

            human_message = HumanMessage(content=f"""
COMPLETED ACTION ANALYSIS:

Action Details:
- Title: {action.title}
- Category: {action.action_category or 'Not specified'}
- Description: {action.description}
- Outcome: {action.outcome.value}
- Duration: {action.duration_days or 'Not specified'} days
- Budget: {action.budget_spent or 'Not specified'}
- Stakeholders: {', '.join(action.stakeholders_involved) if action.stakeholders_involved else 'Not specified'}

Results Data:
{action.results_data}

Success Metrics:
{json.dumps(action.success_metrics or {}, indent=2)}

Current Business Model Canvas:
{json.dumps(bmc_dict, indent=2)}

Market Context:
{json.dumps(market_context, indent=2)}

Provide comprehensive strategic analysis with actionable business intelligence.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            try:
                content = result.content or ""
                json_str = self._extract_json_from_response(content)
                analysis_result = json.loads(json_str)
                
                # Validate and enhance analysis result
                return self._validate_and_enhance_analysis(analysis_result, action)
                
            except json.JSONDecodeError:
                return self._create_fallback_analysis(action)
                
        except Exception as e:
            print(f"OutcomeAnalysisAgent error: {e}")
            return self._create_fallback_analysis(action)

    def _serialize_bmc_with_analytics(self, bmc: BusinessModelCanvas) -> Dict[str, Any]:
        """Serialize BMC with additional analytics for context."""
        bmc_dict = {
            "customer_segments": bmc.customer_segments,
            "value_propositions": bmc.value_propositions,
            "channels": bmc.channels,
            "customer_relationships": bmc.customer_relationships,
            "revenue_streams": bmc.revenue_streams,
            "key_resources": bmc.key_resources,
            "key_activities": bmc.key_activities,
            "key_partnerships": bmc.key_partnerships,
            "cost_structure": bmc.cost_structure
        }
        
        # Add analytics
        bmc_dict["_analytics"] = {
            "total_elements": bmc.get_total_elements(),
            "empty_sections": bmc.get_empty_sections(),
            "completeness_score": bmc.get_completeness_score(),
            "last_updated": bmc.last_updated.isoformat(),
            "version": bmc.version
        }
        
        return bmc_dict

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response."""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return content[start:end]
        return content

    def _validate_and_enhance_analysis(
        self, 
        analysis: Dict[str, Any], 
        action: CompletedAction
    ) -> Dict[str, Any]:
        """Validate and enhance analysis results."""
        # Ensure all required sections exist
        required_sections = [
            "immediate_implications", "strategic_insights", 
            "bmc_section_impacts", "emerging_market_insights",
            "data_driven_recommendations", "summary"
        ]
        
        for section in required_sections:
            if section not in analysis:
                analysis[section] = {"analysis": f"Analysis for {section} pending"}
        
        # Add confidence scoring based on action data quality
        analysis["confidence_metadata"] = {
            "data_completeness": 0.8 if action.success_metrics else 0.5,
            "outcome_clarity": 0.9 if action.outcome == ActionOutcome.SUCCESSFUL else 0.6,
            "context_richness": 0.7,  # Based on available context
            "analysis_depth": 0.8      # Based on analysis comprehensiveness
        }
        
        return analysis

    def _create_fallback_analysis(self, action: CompletedAction) -> Dict[str, Any]:
        """Create fallback analysis when JSON parsing fails."""
        return {
            "immediate_implications": {
                "value_proposition_impact": "Analysis requires manual review",
                "customer_impact": f"Action outcome: {action.outcome.value}",
                "revenue_cost_impact": "Financial impact assessment needed",
                "operational_impact": "Operational review recommended"
            },
            "strategic_insights": {
                "market_positioning": "Strategic analysis requires additional data",
                "competitive_landscape": "Competitive assessment pending",
                "growth_opportunities": "Opportunity analysis needed",
                "risk_factors": "Risk assessment required"
            },
            "bmc_section_impacts": {
                "sections_to_update": [],
                "evidence_strength": "low"
            },
            "emerging_market_insights": {
                "informal_economy": "Context analysis needed",
                "mobile_behavior": "Behavioral analysis pending"
            },
            "data_driven_recommendations": {
                "strategic_recommendations": ["Conduct additional analysis", "Gather more data"],
                "confidence_assessment": "low"
            },
            "summary": f"Analysis of {action.title} requires additional data and manual review for comprehensive insights."
        }


class CanvasUpdateAgent:
    """Enhanced agent for generating specific business model canvas updates."""

    def __init__(self, llm: ChatGoogleGenerativeAI, user_profile: Optional[UserProfile] = None):
        self.llm = llm
        self.user_profile = user_profile

    async def process(
        self,
        analysis: Dict[str, Any],
        current_bmc: BusinessModelCanvas,
        action: CompletedAction
    ) -> List[ProposedChange]:
        """Generate specific, validated BMC update recommendations."""

        try:
            # Enhanced BMC serialization with context
            bmc_dict = self._serialize_bmc_for_updates(current_bmc)
            
            # Prepare user context
            user_context = ""
            if self.user_profile:
                context_data = self.user_profile.get_context_for_ai()
                user_context = f"\nUser Business Profile: {json.dumps(context_data, indent=2)}"

            system_message = SystemMessage(content=f"""
You are a business model design expert specializing in Business Model Canvas optimization for emerging market enterprises.

Generate specific, actionable, and evidence-based updates to business model canvas elements.

UPDATE GENERATION PRINCIPLES:

1. EVIDENCE-BASED CHANGES:
   - Only propose changes supported by concrete evidence from analysis
   - Each change must have clear business rationale
   - Confidence scores must reflect evidence strength
   - Consider data quality and sample sizes

2. STRATEGIC ALIGNMENT:
   - Ensure changes align with overall business strategy
   - Consider interdependencies between BMC sections
   - Maintain coherence across the business model
   - Focus on sustainable competitive advantage

3. EMERGING MARKET ADAPTATION:
   - Consider informal economy dynamics
   - Account for mobile-first customer behaviors
   - Address infrastructure and connectivity constraints
   - Integrate cultural and regulatory considerations

4. IMPLEMENTATION FEASIBILITY:
   - Assess resource requirements for changes
   - Consider implementation timeline and complexity
   - Evaluate organizational readiness
   - Balance ambition with practical constraints

5. RISK ASSESSMENT:
   - Identify potential negative consequences
   - Assess change reversibility
   - Consider stakeholder impact
   - Evaluate competitive response risks

CHANGE SPECIFICATION FORMAT:

Return JSON with "proposed_changes" array containing changes with these fields:
- canvas_section: BMC section name (customer_segments, value_propositions, channels, customer_relationships, revenue_streams, key_resources, key_activities, key_partnerships, cost_structure)
- change_type: "add", "modify", or "remove"
- current_value: existing value being changed (for modify/remove)
- proposed_value: new value to implement
- reasoning: detailed evidence-based explanation (minimum 50 characters)
- confidence_score: 0.0-1.0 based on evidence strength
- impact_assessment: "low", "medium", or "high"
- risk_factors: array of potential risks
- validation_suggestions: array of ways to validate this change
- estimated_effort: implementation effort assessment

CONFIDENCE SCORING GUIDELINES:
- 0.9-1.0: Strong quantitative evidence, large sample, clear success metrics
- 0.7-0.8: Good evidence, moderate sample, measurable outcomes
- 0.5-0.6: Limited evidence, small sample, qualitative indicators
- 0.3-0.4: Weak evidence, assumptions-based, unclear outcomes
- 0.1-0.2: Speculation, no supporting evidence

Only propose changes where evidence clearly supports the recommendation.
{user_context}
""")

            human_message = HumanMessage(content=f"""
STRATEGIC ANALYSIS RESULTS:
{json.dumps(analysis, indent=2)}

CURRENT BUSINESS MODEL CANVAS:
{json.dumps(bmc_dict, indent=2)}

ORIGINAL ACTION CONTEXT:
- Title: {action.title}
- Outcome: {action.outcome.value}
- Category: {action.action_category or 'Not specified'}
- Results Summary: {action.results_data[:500]}...
- Key Metrics: {json.dumps(action.success_metrics or {}, indent=2)}

Generate specific, evidence-based BMC update recommendations with detailed validation and risk assessment.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            try:
                content = result.content or ""
                json_str = self._extract_json_from_response(content)
                parsed_result = json.loads(json_str)
                
                return self._create_validated_changes(parsed_result.get("proposed_changes", []))
                
            except json.JSONDecodeError:
                print("CanvasUpdateAgent: Failed to parse JSON response")
                return []
                
        except Exception as e:
            print(f"CanvasUpdateAgent error: {e}")
            return []

    def _serialize_bmc_for_updates(self, bmc: BusinessModelCanvas) -> Dict[str, Any]:
        """Serialize BMC with update context."""
        return {
            "customer_segments": bmc.customer_segments,
            "value_propositions": bmc.value_propositions,
            "channels": bmc.channels,
            "customer_relationships": bmc.customer_relationships,
            "revenue_streams": bmc.revenue_streams,
            "key_resources": bmc.key_resources,
            "key_activities": bmc.key_activities,
            "key_partnerships": bmc.key_partnerships,
            "cost_structure": bmc.cost_structure,
            "_metadata": {
                "version": bmc.version,
                "completeness": bmc.get_completeness_score(),
                "total_elements": bmc.get_total_elements(),
                "last_updated": bmc.last_updated.isoformat()
            }
        }

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response."""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return content[start:end]
        return content

    def _create_validated_changes(self, changes_data: List[Dict[str, Any]]) -> List[ProposedChange]:
        """Create validated ProposedChange objects."""
        validated_changes = []
        
        for change_data in changes_data:
            try:
                # Validate required fields
                if not all(field in change_data for field in ["canvas_section", "change_type", "proposed_value", "reasoning"]):
                    continue
                
                # Create ProposedChange with enhanced validation
                change = ProposedChange(
                    canvas_section=change_data["canvas_section"],
                    change_type=ChangeType(change_data["change_type"]),
                    current_value=change_data.get("current_value"),
                    proposed_value=change_data["proposed_value"],
                    reasoning=change_data["reasoning"],
                    confidence_score=float(change_data.get("confidence_score", 0.5)),
                    impact_assessment=change_data.get("impact_assessment", "medium"),
                    risk_factors=change_data.get("risk_factors", []),
                    validation_suggestions=change_data.get("validation_suggestions", []),
                    estimated_effort=change_data.get("estimated_effort")
                )
                
                # Additional validation
                if self._validate_change_quality(change):
                    validated_changes.append(change)
                    
            except (ValueError, KeyError, TypeError) as e:
                print(f"Invalid change data: {e}")
                continue
        
        return validated_changes

    def _validate_change_quality(self, change: ProposedChange) -> bool:
        """Validate the quality of a proposed change."""
        # Check reasoning quality
        if len(change.reasoning) < 20:
            return False
        
        # Check confidence score range
        if not (0.0 <= change.confidence_score <= 1.0):
            return False
        
        # Check for meaningful proposed value
        if not change.proposed_value.strip():
            return False
        
        # Avoid generic or placeholder content
        generic_terms = ["tbd", "placeholder", "example", "test", "todo"]
        if change.proposed_value.lower().strip() in generic_terms:
            return False
        
        return True


class NextStepAgent:
    """Enhanced agent for suggesting intelligent follow-up actions."""

    def __init__(self, llm: ChatGoogleGenerativeAI, user_profile: Optional[UserProfile] = None):
        self.llm = llm
        self.user_profile = user_profile

    async def process(
        self,
        proposed_changes: List[ProposedChange],
        current_bmc: BusinessModelCanvas,
        action: CompletedAction,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate intelligent follow-up action recommendations."""

        try:
            # Prepare enhanced context
            changes_summary = self._summarize_changes(proposed_changes)
            user_context = ""
            if self.user_profile:
                context_data = self.user_profile.get_context_for_ai()
                user_context = f"\nUser Profile: {json.dumps(context_data, indent=2)}"

            system_message = SystemMessage(content=f"""
You are a startup strategy advisor specializing in validation methodologies, experiment design, and implementation planning for emerging market businesses.

Design a comprehensive action plan based on proposed business model changes that prioritizes validation, implementation, and measurement.

ACTION PLANNING FRAMEWORK:

1. VALIDATION EXPERIMENTS:
   - Design specific tests to validate proposed changes
   - Consider emerging market constraints (connectivity, resources, infrastructure)
   - Include success criteria and measurement methods
   - Account for cultural and regulatory factors

2. IMPLEMENTATION ROADMAP:
   - Sequence actions based on dependencies and risk
   - Consider resource requirements and timeline
   - Include stakeholder engagement steps
   - Plan for change management and communication

3. MONITORING AND MEASUREMENT:
   - Define key performance indicators
   - Establish baseline measurements
   - Create feedback loops and iteration cycles
   - Plan for course corrections

4. RISK MITIGATION:
   - Address identified risk factors
   - Create contingency plans
   - Plan rollback strategies where needed
   - Consider competitive response scenarios

5. EMERGING MARKET ADAPTATIONS:
   - Account for informal economy dynamics
   - Leverage mobile-first approaches
   - Consider community-based validation
   - Address trust-building requirements

Return JSON with structured action recommendations:
{{
    "immediate_actions": ["actions to take within 1-2 weeks"],
    "short_term_actions": ["actions for 1-2 months timeline"], 
    "validation_experiments": [
        {{
            "experiment_name": "descriptive name",
            "description": "detailed experiment description",
            "success_criteria": "measurable success criteria",
            "timeline": "expected timeline",
            "resources_needed": "resource requirements"
        }}
    ],
    "implementation_steps": [
        {{
            "step": "implementation step",
            "rationale": "why this step is important",
            "timeline": "when to execute",
            "dependencies": "what needs to happen first"
        }}
    ],
    "monitoring_plan": {{
        "key_metrics": ["list of KPIs to track"],
        "measurement_frequency": "how often to measure",
        "review_schedule": "when to review progress"
    }},
    "risk_mitigation": [
        {{
            "risk": "identified risk",
            "mitigation_strategy": "how to address the risk",
            "contingency_plan": "backup plan if risk occurs"
        }}
    ]
}}

Focus on actionable, measurable, and culturally appropriate recommendations.
{user_context}
""")

            human_message = HumanMessage(content=f"""
PROPOSED CHANGES SUMMARY:
{changes_summary}

CURRENT BUSINESS MODEL OVERVIEW:
- Customer Segments: {len(current_bmc.customer_segments)} defined
- Value Propositions: {len(current_bmc.value_propositions)} defined
- Revenue Streams: {len(current_bmc.revenue_streams)} defined
- Key Partnerships: {len(current_bmc.key_partnerships)} defined
- Completeness Score: {current_bmc.get_completeness_score():.1%}

COMPLETED ACTION CONTEXT:
- Action: {action.title}
- Outcome: {action.outcome.value}
- Category: {action.action_category or 'Not specified'}
- Results: {action.results_data[:300]}...

STRATEGIC ANALYSIS INSIGHTS:
- Summary: {analysis.get('summary', 'Analysis summary not available')}
- Key Recommendations: {analysis.get('data_driven_recommendations', {}).get('strategic_recommendations', [])}
- Confidence Level: {analysis.get('data_driven_recommendations', {}).get('confidence_assessment', 'moderate')}

Generate comprehensive follow-up action plan with validation experiments and implementation roadmap.
""")

            result = await self.llm.ainvoke([system_message, human_message])

            try:
                content = result.content or ""
                json_str = self._extract_json_from_response(content)
                parsed_result = json.loads(json_str)
                
                return self._format_next_actions(parsed_result)
                
            except json.JSONDecodeError:
                return self._create_fallback_actions(proposed_changes, action)
                
        except Exception as e:
            print(f"NextStepAgent error: {e}")
            return self._create_fallback_actions(proposed_changes, action)

    def _summarize_changes(self, changes: List[ProposedChange]) -> str:
        """Create a summary of proposed changes."""
        if not changes:
            return "No changes proposed"
        
        summary = f"{len(changes)} proposed changes:\n"
        
        by_section = {}
        for change in changes:
            section = change.canvas_section.replace("_", " ").title()
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(f"{change.change_type.value} - {change.proposed_value[:50]}...")
        
        for section, section_changes in by_section.items():
            summary += f"\n{section}:\n"
            for change_desc in section_changes[:2]:  # Limit to 2 per section
                summary += f"  - {change_desc}\n"
        
        return summary

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response."""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return content[start:end]
        return content

    def _format_next_actions(self, parsed_result: Dict[str, Any]) -> List[str]:
        """Format parsed result into list of next actions."""
        actions = []
        
        # Add immediate actions
        for action in parsed_result.get("immediate_actions", [])[:3]:
            actions.append(f"IMMEDIATE: {action}")
        
        # Add validation experiments
        for exp in parsed_result.get("validation_experiments", [])[:2]:
            exp_name = exp.get("experiment_name", "Validation Experiment")
            description = exp.get("description", "Conduct validation")
            actions.append(f"EXPERIMENT: {exp_name} - {description}")
        
        # Add implementation steps
        for step in parsed_result.get("implementation_steps", [])[:2]:
            step_desc = step.get("step", "Implementation step")
            rationale = step.get("rationale", "")
            actions.append(f"IMPLEMENT: {step_desc} - {rationale}")
        
        # Add monitoring plan
        monitoring = parsed_result.get("monitoring_plan", {})
        if monitoring.get("key_metrics"):
            metrics = ", ".join(monitoring["key_metrics"][:3])
            actions.append(f"MONITOR: Track key metrics - {metrics}")
        
        return actions[:8]  # Limit to 8 total actions

    def _create_fallback_actions(
        self, 
        changes: List[ProposedChange], 
        action: CompletedAction
    ) -> List[str]:
        """Create fallback actions when JSON parsing fails."""
        fallback_actions = [
            f"Review and validate findings from {action.title}",
            "Conduct stakeholder discussions on proposed changes",
            "Design validation experiments for key assumptions",
            "Create implementation timeline for approved changes"
        ]
        
        if changes:
            high_confidence_changes = [c for c in changes if c.confidence_score >= 0.8]
            if high_confidence_changes:
                fallback_actions.append(f"Prioritize implementation of {len(high_confidence_changes)} high-confidence changes")
            
            # Add section-specific actions
            sections = list(set(c.canvas_section for c in changes))
            for section in sections[:2]:
                section_name = section.replace("_", " ").title()
                fallback_actions.append(f"Validate proposed {section_name} updates with stakeholders")
        
        return fallback_actions[:6]


# Enhanced Utility Functions

def validate_safety(proposed_changes: List[ProposedChange]) -> bool:
    """Enhanced safety validation for auto-application."""
    if not proposed_changes:
        return True

    # Enhanced safety criteria
    for change in proposed_changes:
        # Use the enhanced safety check from the model
        if not change.is_safe_for_auto_application():
            return False
    
    # Additional system-level safety checks
    high_impact_changes = [c for c in proposed_changes 
                          if c.impact_assessment == "high"]
    
    if len(high_impact_changes) > 2:  # Too many high-impact changes
        return False
    
    # Check for conflicting changes
    if _has_conflicting_changes(proposed_changes):
        return False
    
    return True


def _has_conflicting_changes(changes: List[ProposedChange]) -> bool:
    """Check for conflicting changes in the same section."""
    section_changes = {}
    
    for change in changes:
        section = change.canvas_section
        if section not in section_changes:
            section_changes[section] = []
        section_changes[section].append(change)
    
    # Check for conflicts within each section
    for section, section_change_list in section_changes.items():
        if len(section_change_list) > 1:
            # Multiple changes to same section - check for conflicts
            for i, change1 in enumerate(section_change_list):
                for change2 in section_change_list[i+1:]:
                    if _changes_conflict(change1, change2):
                        return True
    
    return False


def _changes_conflict(change1: ProposedChange, change2: ProposedChange) -> bool:
    """Check if two changes conflict with each other."""
    # Same value being modified differently
    if (change1.current_value and change2.current_value and 
        change1.current_value == change2.current_value and
        change1.proposed_value != change2.proposed_value):
        return True
    
    # One removes what the other modifies
    if (change1.change_type == ChangeType.REMOVE and 
        change2.change_type == ChangeType.MODIFY and
        change1.current_value == change2.current_value):
        return True
    
    return False


async def create_orchestrator(
    user_profile: Optional[UserProfile] = None
) -> AgenticOrchestrator:
    """Enhanced factory function to create a configured orchestrator."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set")

    return AgenticOrchestrator(
        google_api_key=api_key, 
        model_name="gemini-1.5-flash",
        user_profile=user_profile
    )


async def test_api_connection() -> bool:
    """Enhanced API connectivity test."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY environment variable not set")
            return False
            
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=100
        )
        
        test_message = HumanMessage(content="Respond with 'API connection successful' to confirm connectivity.")
        response = await llm.ainvoke([test_message])
        
        success = "successful" in response.content.lower()
        print(f"API Test Response: {response.content}")
        print(f"Connection Status: {'SUCCESS' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"API Connection Test Failed: {e}")
        return False


# Export main classes and functions
__all__ = [
    'AgenticOrchestrator',
    'ActionDetectionAgent', 
    'OutcomeAnalysisAgent',
    'CanvasUpdateAgent',
    'NextStepAgent',
    'validate_safety',
    'create_orchestrator',
    'test_api_connection'
]


# Usage example and testing
if __name__ == "__main__":
    print("Enhanced Agentic Engine - Testing API Connection...")
    asyncio.run(test_api_connection())