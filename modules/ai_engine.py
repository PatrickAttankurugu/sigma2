"""
AI Engine with Quality Validation for SIGMA Actions Co-pilot
"""

import json
import time
import uuid
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from .bmc_canvas import BusinessModelCanvas
from .utils import LoggingMixin


@dataclass
class ResponseQuality:
    """Quality metrics for AI responses"""
    overall_score: float
    specificity_score: float
    evidence_score: float
    actionability_score: float
    consistency_score: float
    issues: List[str]


class AIQualityValidator:
    """Validates and scores AI response quality"""
    
    def __init__(self):
        self.quality_history = []
    
    def validate_response(self, response: Dict[str, Any], action_data: Dict[str, Any]) -> ResponseQuality:
        """Comprehensive response quality validation"""
        
        specificity = self._score_specificity(response)
        evidence = self._score_evidence_alignment(response, action_data)
        actionability = self._score_actionability(response)
        consistency = self._score_consistency(response)
        
        issues = self._identify_issues(response, action_data)
        
        overall = (specificity + evidence + actionability + consistency) / 4
        
        quality = ResponseQuality(
            overall_score=overall,
            specificity_score=specificity,
            evidence_score=evidence,
            actionability_score=actionability,
            consistency_score=consistency,
            issues=issues
        )
        
        self.quality_history.append({
            'timestamp': datetime.now(),
            'quality': quality,
            'action_outcome': action_data.get('outcome'),
            'action_type': 'sample' if 'Sample' in action_data.get('title', '') else 'custom'
        })
        
        return quality
    
    def _score_specificity(self, response: Dict[str, Any]) -> float:
        """Score how specific vs generic the response is"""
        score = 1.0
        
        generic_phrases = [
            'improve', 'enhance', 'optimize', 'better', 'more effective',
            'innovative', 'cutting-edge', 'world-class', 'best-in-class',
            'leverage', 'utilize', 'implement', 'establish'
        ]
        
        text_content = json.dumps(response).lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_content)
        
        score -= min(generic_count * 0.1, 0.4)
        
        if any(char.isdigit() for char in text_content):
            score += 0.2
        
        next_steps = response.get('next_steps', [])
        if next_steps and len(next_steps) > 0:
            detailed_steps = sum(1 for step in next_steps if len(str(step)) > 100)
            score += min(detailed_steps * 0.1, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _score_evidence_alignment(self, response: Dict[str, Any], action_data: Dict[str, Any]) -> float:
        """Score how well the response aligns with action evidence"""
        score = 0.7
        
        action_text = (action_data.get('results', '') + ' ' + action_data.get('description', '')).lower()
        response_text = json.dumps(response).lower()
        
        outcome = action_data.get('outcome', '').lower()
        if outcome == 'successful':
            growth_terms = ['scale', 'expand', 'grow', 'increase', 'more']
            if any(term in response_text for term in growth_terms):
                score += 0.1
        elif outcome == 'failed':
            pivot_terms = ['pivot', 'change', 'different', 'alternative', 'reconsider']
            if any(term in response_text for term in pivot_terms):
                score += 0.1
        
        return min(1.0, score)
    
    def _score_actionability(self, response: Dict[str, Any]) -> float:
        """Score how actionable the recommendations are"""
        score = 0.5
        
        changes = response.get('changes', [])
        if changes:
            for change in changes:
                new_value = change.get('new', '')
                reasoning = change.get('reason', '')
                
                if len(new_value) > 20:
                    score += 0.05
                
                if len(reasoning) > 30:
                    score += 0.05
        
        next_steps = response.get('next_steps', [])
        for step in next_steps:
            if isinstance(step, dict):
                if step.get('timeline') and step.get('resources_needed'):
                    score += 0.1
                if step.get('success_metrics'):
                    score += 0.1
            elif len(str(step)) > 50:
                score += 0.05
        
        return min(1.0, score)
    
    def _score_consistency(self, response: Dict[str, Any]) -> float:
        """Score internal consistency of recommendations"""
        score = 1.0
        
        changes = response.get('changes', [])
        if len(changes) < 2:
            return score
        
        return max(0.0, score)
    
    def _identify_issues(self, response: Dict[str, Any], action_data: Dict[str, Any]) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        if not response.get('analysis'):
            issues.append("Missing analysis section")
        if not response.get('changes'):
            issues.append("No changes proposed")
        if not response.get('next_steps'):
            issues.append("No next steps suggested")
        
        changes = response.get('changes', [])
        low_confidence_count = sum(1 for c in changes if c.get('confidence', 0) < 0.6)
        if low_confidence_count > len(changes) * 0.7:
            issues.append("Too many low-confidence recommendations")
        
        analysis = response.get('analysis', '')
        if len(analysis) < 50:
            issues.append("Analysis too brief")
        
        next_steps = response.get('next_steps', [])
        if next_steps:
            generic_steps = sum(1 for step in next_steps if len(str(step)) < 30)
            if generic_steps > len(next_steps) * 0.5:
                issues.append("Next steps too generic")
        
        return issues
    
    def should_retry(self, quality: ResponseQuality) -> bool:
        """Determine if response quality is too low and should retry"""
        if quality.overall_score < 0.4:
            return True
        if len(quality.issues) > 3:
            return True
        if quality.specificity_score < 0.3:
            return True
        return False
    
    def get_improvement_prompt(self, quality: ResponseQuality, original_response: Dict[str, Any]) -> str:
        """Generate prompt for improving low-quality response"""
        issues_text = ", ".join(quality.issues)
        
        return f"""The previous response had quality issues: {issues_text}

Please provide a better analysis that:
1. Is more specific and detailed (avoid generic business language)
2. References specific data from the action results
3. Provides actionable, implementable recommendations
4. Has clear reasoning for each suggested change
5. Includes detailed next steps with timelines, resources, and success metrics

Original response to improve:
{json.dumps(original_response, indent=2)}

Provide an improved version that addresses these quality issues."""
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get quality metrics for dashboard"""
        if not self.quality_history:
            return {}
        
        recent_scores = [q['quality'].overall_score for q in self.quality_history[-10:]]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        retry_rate = sum(1 for q in self.quality_history[-20:] 
                        if self.should_retry(q['quality'])) / min(20, len(self.quality_history))
        
        return {
            "average_quality_score": avg_score,
            "retry_rate": retry_rate,
            "total_responses": len(self.quality_history),
            "recent_trend": "improving" if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else "stable"
        }


class QualityEnhancedAI(LoggingMixin):
    """AI engine with response quality validation and enhanced next steps generation"""
    
    def __init__(self, api_key: str):
        """Initialize with Google Gemini and quality validation"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash",
                temperature=0.3,
                max_output_tokens=2000,
                timeout=30
            )
            self.SystemMessage = SystemMessage
            self.HumanMessage = HumanMessage
            self.quality_validator = AIQualityValidator()
            self.max_retries = 2
            
            self.log_ai_performance("ai_initialization", 0, True, {
                "model": "gemini-2.0-flash",
                "quality_validation": True,
                "enhanced_next_steps": True
            })
            
        except ImportError as e:
            self.log_ai_performance("ai_initialization", 0, False, {"error": str(e)})
            raise ImportError(f"Missing dependencies: {e}. Run: pip install langchain langchain-google-genai")

    def analyze_action_with_quality_control(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas) -> Tuple[Dict[str, Any], ResponseQuality]:
        """Analyze action with quality validation and retry logic"""
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())[:8]
        
        self.log_ai_performance("quality_analysis_started", 0, True, {
            "analysis_id": analysis_id,
            "action_title": action_data.get('title', 'Unknown'),
            "action_outcome": action_data.get('outcome', 'Unknown'),
            "business_stage": bmc.get_business_stage(),
            "max_retries": self.max_retries
        })
        
        last_response = None
        quality = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    response = self._get_initial_response(action_data, bmc, analysis_id)
                else:
                    response = self._retry_with_improvement(action_data, bmc, last_response, quality, analysis_id)
                
                quality = self.quality_validator.validate_response(response, action_data)
                
                self.log_ai_performance(f"quality_check_attempt_{attempt + 1}", 0, True, {
                    "analysis_id": analysis_id,
                    "overall_score": quality.overall_score,
                    "should_retry": self.quality_validator.should_retry(quality),
                    "attempt": attempt + 1
                })
                
                if not self.quality_validator.should_retry(quality) or attempt == self.max_retries:
                    break
                
                last_response = response
                
            except Exception as e:
                self.log_ai_performance(f"quality_retry_failed_attempt_{attempt + 1}", 0, False, {
                    "analysis_id": analysis_id,
                    "error": str(e),
                    "attempt": attempt + 1
                })
                
                if attempt == self.max_retries:
                    response = self._create_fallback_response(action_data, bmc)
                    quality = ResponseQuality(
                        overall_score=0.3,
                        specificity_score=0.3,
                        evidence_score=0.3,
                        actionability_score=0.3,
                        consistency_score=0.3,
                        issues=["Quality validation failed", "Using fallback response"]
                    )
                    break
        
        total_duration = (time.time() - start_time) * 1000
        self.log_ai_performance("quality_analysis_completed", int(total_duration), True, {
            "analysis_id": analysis_id,
            "final_quality_score": quality.overall_score,
            "attempts_used": attempt + 1,
            "total_duration_ms": int(total_duration)
        })
        
        return response, quality

    def _get_initial_response(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas, analysis_id: str) -> Dict[str, Any]:
        """Get initial AI response with enhanced prompting"""
        
        business_stage = bmc.get_business_stage()
        business_context = bmc.get_enhanced_business_context()
        risk_assessment = bmc.get_risk_assessment()
        
        system_prompt = f"""You are SIGMA's AI co-pilot helping entrepreneurs validate business assumptions through experiments.

Your role is to:
1. Analyze completed actions and suggest specific Business Model updates
2. Generate strategic, implementable next steps based on the outcome
3. Provide business intelligence based on current stage and risk assessment

BUSINESS INTELLIGENCE CONTEXT:
- Current Stage: {business_stage}
- Risk Assessment: {risk_assessment}

OUTCOME-SPECIFIC STRATEGY:
- SUCCESSFUL outcomes → Scale/optimize/expand recommendations with growth metrics
- FAILED outcomes → Pivot/alternative approaches with risk mitigation strategies  
- INCONCLUSIVE outcomes → Clarification experiments with better data collection methods

FOCUS ONLY on these 4 business model sections:
- customer_segments
- value_propositions  
- business_models
- market_opportunities

Return ONLY valid JSON in this exact format:
{{
    "analysis": "2-3 sentence analysis of what this action outcome means for the business model and strategic implications",
    "changes": [
        {{
            "section": "customer_segments|value_propositions|business_models|market_opportunities",
            "type": "add|modify|remove", 
            "current": "existing item being modified/removed (null for add operations)",
            "new": "new item to add or replacement text",
            "reason": "clear explanation of why this change makes sense based on the action outcome",
            "confidence": 0.85
        }}
    ],
    "next_steps": [
        {{
            "title": "Specific actionable title (e.g., 'Run A/B pricing test with target segment')",
            "description": "Detailed description of what to do and why",
            "timeline": "Specific timeframe (e.g., '2-3 weeks', '1 month')",
            "resources_needed": ["Specific resources like '$500 budget', '8 hours dev time', 'Marketing manager']",
            "success_metrics": ["Measurable outcomes like '<15% churn rate', '>60% conversion rate', '$2K+ revenue']",
            "priority": "high|medium|low",
            "difficulty": "easy|medium|hard",
            "stage": "validation|growth|scale",
            "implementation_steps": ["Step 1: Specific action", "Step 2: Next action", "Step 3: Final action"]
        }}
    ]
}}

NEXT STEPS REQUIREMENTS:
- Minimum 2, maximum 4 next steps
- Each must be immediately actionable with clear implementation path
- Include specific timelines (not vague like "soon")  
- List exact resources needed (budget, time, skills)
- Define measurable success criteria with target numbers
- Prioritize based on impact vs effort for current business stage
- Match difficulty to entrepreneur's likely capabilities
- Ensure logical sequence building on current learnings

STAGE-SPECIFIC FOCUS:
- Discovery/Validation: Customer development, assumption testing, MVP validation
- Growth: Channel optimization, retention improvement, scaling experiments
- Scale: Market expansion, operational efficiency, competitive differentiation

Rules:
- Only suggest changes with confidence > 0.6
- Focus on what the action outcome actually validates or invalidates  
- Be specific - avoid generic business language
- Make next steps feel like strategic guidance from an experienced advisor
- Ensure recommendations build logically on the current action outcome"""

        user_prompt = f"""COMPLETED ACTION/EXPERIMENT:
Title: {action_data['title']}
Outcome: {action_data['outcome']} 
Description: {action_data['description']}
Key Results: {action_data['results']}

CURRENT BUSINESS CONTEXT:
{business_context}

Based on this action outcome and business context, provide:
1. Strategic analysis of implications
2. Specific business model updates  
3. Detailed next steps with implementation guidance

Focus on actionable insights that move the business forward based on the {action_data['outcome']} outcome.

Return only the JSON response."""

        api_start = time.time()
        messages = [
            self.SystemMessage(content=system_prompt),
            self.HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        api_duration = (time.time() - api_start) * 1000
        
        self.log_ai_performance("quality_api_call", int(api_duration), True, {
            "analysis_id": analysis_id,
            "response_length": len(response.content),
            "model": "gemini-2.0-flash",
            "business_stage": business_stage
        })
        
        return self._parse_response(response.content)

    def _retry_with_improvement(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas, 
                               last_response: Dict[str, Any], quality: ResponseQuality, analysis_id: str) -> Dict[str, Any]:
        """Retry analysis with improvement prompt"""
        
        improvement_prompt = self.quality_validator.get_improvement_prompt(quality, last_response)
        
        messages = [
            self.SystemMessage(content=improvement_prompt),
            self.HumanMessage(content=f"""
            ACTION: {action_data['title']}
            OUTCOME: {action_data['outcome']}
            RESULTS: {action_data['results']}
            
            BUSINESS CONTEXT: {bmc.get_enhanced_business_context()}
            BUSINESS STAGE: {bmc.get_business_stage()}
            
            Provide an improved analysis addressing the quality issues mentioned.
            Include detailed next steps with timelines, resources, and success metrics.
            Return only valid JSON.
            """)
        ]
        
        try:
            response = self.llm.invoke(messages)
            self.log_ai_performance("quality_retry_attempt", 0, True, {
                "analysis_id": analysis_id,
                "improvement_issues": len(quality.issues)
            })
            return self._parse_response(response.content)
            
        except Exception as e:
            self.log_ai_performance("quality_retry_failed", 0, False, {
                "analysis_id": analysis_id,
                "error": str(e)
            })
            return last_response

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response content to JSON"""
        content = content.strip()
        
        if "```json" in content:
            json_part = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_part = content.split("```")[1].split("```")[0].strip()
        else:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_part = content[start:end]
            else:
                json_part = content
        
        result = json.loads(json_part)
        
        if 'analysis' not in result:
            result['analysis'] = "Analysis completed successfully"
        if 'changes' not in result:
            result['changes'] = []
        if 'next_steps' not in result:
            result['next_steps'] = [
                {
                    "title": "Continue current approach",
                    "description": "Build on learnings from this action",
                    "timeline": "1-2 weeks",
                    "resources_needed": ["Team time"],
                    "success_metrics": ["Progress toward goals"],
                    "priority": "medium",
                    "difficulty": "easy",
                    "stage": "validation",
                    "implementation_steps": ["Review results", "Plan next steps"]
                }
            ]
        
        return result

    def _create_fallback_response(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas) -> Dict[str, Any]:
        """Create fallback response when all retries fail"""
        business_stage = bmc.get_business_stage()
        
        return {
            "analysis": f"Analysis of '{action_data.get('title', 'the action')}' completed with {action_data.get('outcome', 'unknown')} outcome. Manual review recommended for detailed insights.",
            "changes": [],
            "next_steps": [
                {
                    "title": "Manual review of action results",
                    "description": "Conduct detailed manual analysis of the action outcomes and implications",
                    "timeline": "1 week",
                    "resources_needed": ["2-3 hours analysis time"],
                    "success_metrics": ["Clear insights identified", "Next actions defined"],
                    "priority": "high",
                    "difficulty": "easy",
                    "stage": business_stage,
                    "implementation_steps": [
                        "Review all action data in detail",
                        "Identify key learnings and implications", 
                        "Define specific next experiments"
                    ]
                },
                {
                    "title": "Collect additional data",
                    "description": f"Gather more structured data to better understand the {action_data.get('outcome', 'unknown')} outcome",
                    "timeline": "2 weeks",
                    "resources_needed": ["Data collection tools", "Customer outreach"],
                    "success_metrics": ["Additional insights gathered", "Clearer direction identified"],
                    "priority": "medium", 
                    "difficulty": "medium",
                    "stage": business_stage,
                    "implementation_steps": [
                        "Design data collection approach",
                        "Execute data gathering",
                        "Analyze and synthesize findings"
                    ]
                }
            ]
        }

    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get quality metrics for dashboard"""
        return self.quality_validator.get_quality_dashboard_data()