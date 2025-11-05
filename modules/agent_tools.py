"""
Agent Tools for Multi-Agent System

This module provides specialized tools for each agent in the multi-agent system:
- Strategy Agent: Business analysis tools
- Market Research Agent: Market and competitor research tools
- Product Agent: Value proposition analysis tools
- Execution Agent: Action analysis and recommendation tools
"""

from typing import Dict, List, Any, Optional
from langchain_core.tools import StructuredTool
from langchain_community.utilities import GoogleSearchAPIWrapper
import json
import os
from modules.bmc_canvas import BusinessModelCanvas
from modules.utils import get_logger

logger = get_logger(__name__)


class AgentTools:
    """Factory class for creating agent-specific tools"""

    def __init__(self, bmc: BusinessModelCanvas):
        self.bmc = bmc
        # Initialize Google Search if API keys are available
        self.search_wrapper = None
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
            try:
                self.search_wrapper = GoogleSearchAPIWrapper()
            except Exception as e:
                logger.warning(f"Could not initialize Google Search: {e}")

    # ========== Strategy Agent Tools ==========

    def analyze_bmc_coherence(self, context: str = "") -> str:
        """
        Analyze the Business Model Canvas for internal coherence and alignment.
        Returns insights about how well different BMC components work together.
        """
        try:
            customer_segments = self.bmc.get_section('customer_segments')
            value_propositions = self.bmc.get_section('value_propositions')
            business_models = self.bmc.get_section('business_models')
            market_opportunities = self.bmc.get_section('market_opportunities')

            stage = self.bmc.get_business_stage()
            risks = self.bmc.get_risk_assessment()

            analysis = {
                "bmc_summary": {
                    "customer_segments": len(customer_segments),
                    "value_propositions": len(value_propositions),
                    "business_models": len(business_models),
                    "market_opportunities": len(market_opportunities)
                },
                "business_stage": stage,
                "identified_risks": risks,
                "coherence_notes": []
            }

            # Check for alignment issues
            if len(customer_segments) == 0:
                analysis["coherence_notes"].append("No customer segments defined - critical gap")
            if len(value_propositions) == 0:
                analysis["coherence_notes"].append("No value propositions defined - critical gap")
            if len(customer_segments) > 0 and len(value_propositions) == 0:
                analysis["coherence_notes"].append("Customer segments exist but no value propositions to serve them")
            if len(market_opportunities) > len(customer_segments):
                analysis["coherence_notes"].append("More market opportunities than customer segments - may need focus")

            return json.dumps(analysis, indent=2)

        except Exception as e:
            logger.error(f"Error in BMC coherence analysis: {e}")
            return json.dumps({"error": str(e)})

    def detect_business_stage(self, context: str = "") -> str:
        """
        Detect the current business stage based on BMC content and context.
        Returns: validation, growth, or scale with reasoning.
        """
        try:
            stage = self.bmc.get_business_stage()

            # Get all BMC content for context
            all_items = []
            for section in ['customer_segments', 'value_propositions', 'business_models', 'market_opportunities']:
                all_items.extend(self.bmc.get_section(section))

            result = {
                "detected_stage": stage,
                "reasoning": f"Based on keywords and business context in BMC items",
                "bmc_items_analyzed": len(all_items),
                "recommendations_by_stage": {
                    "validation": "Focus on customer discovery, MVP testing, and hypothesis validation",
                    "growth": "Focus on scaling customer acquisition, optimizing channels, and retention",
                    "scale": "Focus on international expansion, automation, and competitive advantages"
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error detecting business stage: {e}")
            return json.dumps({"error": str(e)})

    # ========== Market Research Agent Tools ==========

    def search_market_trends(self, query: str) -> str:
        """
        Search for market trends and insights related to the business domain.
        Uses Google Search to find relevant market information.
        """
        try:
            if not self.search_wrapper:
                return json.dumps({
                    "error": "Search not available. Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.",
                    "fallback": "Consider researching market trends for: " + query
                })

            results = self.search_wrapper.results(query, num_results=5)

            formatted_results = {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "link": r.get("link", "")
                    }
                    for r in results
                ]
            }

            return json.dumps(formatted_results, indent=2)

        except Exception as e:
            logger.error(f"Error searching market trends: {e}")
            return json.dumps({
                "error": str(e),
                "suggestion": "Manual research recommended for: " + query
            })

    def analyze_customer_segments(self, context: str = "") -> str:
        """
        Analyze the defined customer segments for completeness and specificity.
        Provides recommendations for improving customer segment definitions.
        """
        try:
            segments = self.bmc.get_section('customer_segments')

            analysis = {
                "total_segments": len(segments),
                "segments": segments,
                "quality_assessment": [],
                "recommendations": []
            }

            for segment in segments:
                quality = {
                    "segment": segment,
                    "has_demographic": any(word in segment.lower() for word in ['age', 'old', 'young', 'male', 'female', 'income']),
                    "has_geographic": any(word in segment.lower() for word in ['urban', 'rural', 'city', 'country', 'region', 'area']),
                    "has_psychographic": any(word in segment.lower() for word in ['owner', 'professional', 'enthusiast', 'beginner']),
                    "specificity_score": len(segment.split()) / 20.0  # Rough score based on detail
                }
                analysis["quality_assessment"].append(quality)

            if len(segments) == 0:
                analysis["recommendations"].append("Define at least one customer segment to focus efforts")
            elif len(segments) > 5:
                analysis["recommendations"].append("Consider focusing on fewer segments for initial validation")

            return json.dumps(analysis, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing customer segments: {e}")
            return json.dumps({"error": str(e)})

    # ========== Product Agent Tools ==========

    def analyze_value_propositions(self, context: str = "") -> str:
        """
        Analyze value propositions for uniqueness, clarity, and customer alignment.
        """
        try:
            value_props = self.bmc.get_section('value_propositions')
            customer_segments = self.bmc.get_section('customer_segments')

            analysis = {
                "total_value_propositions": len(value_props),
                "value_propositions": value_props,
                "quality_metrics": [],
                "alignment_with_customers": len(customer_segments) > 0,
                "recommendations": []
            }

            for vp in value_props:
                metrics = {
                    "value_proposition": vp,
                    "has_quantifiable_benefit": any(char.isdigit() or word in vp.lower() for char in vp for word in ['faster', 'cheaper', 'better', 'more', 'less', 'save', 'reduce', 'increase']),
                    "mentions_pain_point": any(word in vp.lower() for word in ['problem', 'pain', 'challenge', 'difficulty', 'issue', 'struggle']),
                    "has_differentiation": any(word in vp.lower() for word in ['unique', 'only', 'first', 'exclusive', 'proprietary', 'innovative']),
                    "clarity_score": min(1.0, len(vp.split()) / 15.0)  # Optimal around 15 words
                }
                analysis["quality_metrics"].append(metrics)

            if len(value_props) == 0:
                analysis["recommendations"].append("Define clear value propositions that address customer pain points")

            if len(customer_segments) == 0:
                analysis["recommendations"].append("Define customer segments to ensure value propositions are targeted")

            return json.dumps(analysis, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing value propositions: {e}")
            return json.dumps({"error": str(e)})

    def assess_competitive_positioning(self, context: str = "") -> str:
        """
        Assess competitive positioning based on value propositions and market opportunities.
        """
        try:
            value_props = self.bmc.get_section('value_propositions')
            market_opps = self.bmc.get_section('market_opportunities')

            assessment = {
                "competitive_elements": [],
                "differentiation_strength": "unknown",
                "positioning_recommendations": []
            }

            # Look for competitive keywords in value props
            competitive_keywords = ['unique', 'only', 'first', 'exclusive', 'better', 'faster', 'cheaper', 'innovative', 'proprietary']

            for vp in value_props:
                found_keywords = [kw for kw in competitive_keywords if kw in vp.lower()]
                if found_keywords:
                    assessment["competitive_elements"].append({
                        "value_proposition": vp,
                        "competitive_keywords": found_keywords
                    })

            # Assess strength
            if len(assessment["competitive_elements"]) >= 2:
                assessment["differentiation_strength"] = "strong"
            elif len(assessment["competitive_elements"]) == 1:
                assessment["differentiation_strength"] = "moderate"
            else:
                assessment["differentiation_strength"] = "weak"
                assessment["positioning_recommendations"].append(
                    "Strengthen differentiation by highlighting unique benefits or approaches"
                )

            # Check market opportunities alignment
            if len(market_opps) == 0:
                assessment["positioning_recommendations"].append(
                    "Define market opportunities to understand competitive landscape"
                )

            return json.dumps(assessment, indent=2)

        except Exception as e:
            logger.error(f"Error assessing competitive positioning: {e}")
            return json.dumps({"error": str(e)})

    # ========== Execution Agent Tools ==========

    def analyze_action_impact(self, action_data: str) -> str:
        """
        Analyze the impact of a completed action/experiment on the business model.
        Expects action_data to be a JSON string with keys: title, description, outcome, metrics
        """
        try:
            action = json.loads(action_data) if isinstance(action_data, str) else action_data

            # Extract outcome sentiment
            outcome = action.get('outcome', '').lower()
            outcome_type = 'inconclusive'

            if any(word in outcome for word in ['success', 'positive', 'exceed', 'achieved', 'validated', 'confirmed']):
                outcome_type = 'successful'
            elif any(word in outcome for word in ['fail', 'negative', 'below', 'did not', 'invalid', 'rejected']):
                outcome_type = 'failed'

            impact = {
                "action_title": action.get('title', 'Unknown'),
                "outcome_type": outcome_type,
                "impact_areas": [],
                "recommended_bmc_changes": [],
                "confidence_level": "medium"
            }

            # Determine which BMC sections this action affects
            title_lower = action.get('title', '').lower()
            description_lower = action.get('description', '').lower()

            if any(word in title_lower or word in description_lower for word in ['customer', 'segment', 'user', 'audience', 'target']):
                impact["impact_areas"].append("customer_segments")

            if any(word in title_lower or word in description_lower for word in ['value', 'proposition', 'benefit', 'feature', 'solution']):
                impact["impact_areas"].append("value_propositions")

            if any(word in title_lower or word in description_lower for word in ['pricing', 'revenue', 'model', 'monetization', 'channel']):
                impact["impact_areas"].append("business_models")

            if any(word in title_lower or word in description_lower for word in ['market', 'opportunity', 'demand', 'trend', 'competition']):
                impact["impact_areas"].append("market_opportunities")

            # Generate recommendations based on outcome
            if outcome_type == 'successful':
                impact["recommended_bmc_changes"].append({
                    "action": "reinforce",
                    "reasoning": "Success indicates this approach is working - consider scaling or doubling down"
                })
                impact["confidence_level"] = "high"
            elif outcome_type == 'failed':
                impact["recommended_bmc_changes"].append({
                    "action": "pivot",
                    "reasoning": "Failure suggests need to adjust approach or assumptions"
                })
                impact["confidence_level"] = "high"
            else:
                impact["recommended_bmc_changes"].append({
                    "action": "clarify",
                    "reasoning": "Inconclusive results suggest need for more targeted experiments"
                })
                impact["confidence_level"] = "low"

            return json.dumps(impact, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing action impact: {e}")
            return json.dumps({"error": str(e)})

    def generate_next_steps(self, analysis_context: str) -> str:
        """
        Generate actionable next steps based on analysis context.
        Expects analysis_context to include action results and current business stage.
        """
        try:
            stage = self.bmc.get_business_stage()

            next_steps = {
                "business_stage": stage,
                "recommended_actions": [],
                "prioritization_criteria": "Impact vs Effort matrix"
            }

            # Stage-specific recommendations
            if stage == "validation":
                next_steps["recommended_actions"].extend([
                    {
                        "title": "Conduct additional customer interviews",
                        "priority": "high",
                        "timeline": "1-2 weeks",
                        "effort": "medium"
                    },
                    {
                        "title": "Test MVP with early adopters",
                        "priority": "high",
                        "timeline": "2-4 weeks",
                        "effort": "high"
                    },
                    {
                        "title": "Validate pricing assumptions",
                        "priority": "medium",
                        "timeline": "1-2 weeks",
                        "effort": "low"
                    }
                ])
            elif stage == "growth":
                next_steps["recommended_actions"].extend([
                    {
                        "title": "Optimize customer acquisition channels",
                        "priority": "high",
                        "timeline": "2-3 weeks",
                        "effort": "medium"
                    },
                    {
                        "title": "Implement retention program",
                        "priority": "high",
                        "timeline": "3-4 weeks",
                        "effort": "high"
                    },
                    {
                        "title": "Expand to adjacent customer segments",
                        "priority": "medium",
                        "timeline": "4-6 weeks",
                        "effort": "high"
                    }
                ])
            else:  # scale
                next_steps["recommended_actions"].extend([
                    {
                        "title": "Explore international market expansion",
                        "priority": "high",
                        "timeline": "8-12 weeks",
                        "effort": "high"
                    },
                    {
                        "title": "Automate operational processes",
                        "priority": "medium",
                        "timeline": "6-8 weeks",
                        "effort": "medium"
                    },
                    {
                        "title": "Build strategic partnerships",
                        "priority": "high",
                        "timeline": "4-8 weeks",
                        "effort": "medium"
                    }
                ])

            return json.dumps(next_steps, indent=2)

        except Exception as e:
            logger.error(f"Error generating next steps: {e}")
            return json.dumps({"error": str(e)})

    # ========== Tool Creation Methods ==========

    def get_strategy_tools(self) -> List[StructuredTool]:
        """Get tools for the Strategy Agent"""
        return [
            StructuredTool.from_function(
                func=self.analyze_bmc_coherence,
                name="analyze_bmc_coherence",
                description="Analyze Business Model Canvas for internal coherence and alignment. Use this to understand how well different BMC components work together."
            ),
            StructuredTool.from_function(
                func=self.detect_business_stage,
                name="detect_business_stage",
                description="Detect current business stage (validation/growth/scale). Use this to understand what phase the business is in and what strategies are appropriate."
            )
        ]

    def get_market_research_tools(self) -> List[StructuredTool]:
        """Get tools for the Market Research Agent"""
        tools = [
            StructuredTool.from_function(
                func=self.analyze_customer_segments,
                name="analyze_customer_segments",
                description="Analyze defined customer segments for completeness and quality. Use this to evaluate how well customer segments are defined."
            )
        ]

        if self.search_wrapper:
            tools.append(
                StructuredTool.from_function(
                    func=self.search_market_trends,
                    name="search_market_trends",
                    description="Search for market trends and insights. Input should be a search query about market trends, competitors, or industry insights."
                )
            )

        return tools

    def get_product_tools(self) -> List[StructuredTool]:
        """Get tools for the Product Agent"""
        return [
            StructuredTool.from_function(
                func=self.analyze_value_propositions,
                name="analyze_value_propositions",
                description="Analyze value propositions for uniqueness, clarity, and customer alignment. Use this to evaluate product-market fit."
            ),
            StructuredTool.from_function(
                func=self.assess_competitive_positioning,
                name="assess_competitive_positioning",
                description="Assess competitive positioning based on value propositions and market opportunities. Use this to understand differentiation."
            )
        ]

    def get_execution_tools(self) -> List[StructuredTool]:
        """Get tools for the Execution Agent"""
        return [
            StructuredTool.from_function(
                func=self.analyze_action_impact,
                name="analyze_action_impact",
                description="Analyze the impact of a completed action/experiment. Input should be JSON string with keys: title, description, outcome, metrics."
            ),
            StructuredTool.from_function(
                func=self.generate_next_steps,
                name="generate_next_steps",
                description="Generate actionable next steps based on analysis. Input should include action results and business context."
            )
        ]
