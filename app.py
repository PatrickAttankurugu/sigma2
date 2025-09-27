"""
SIGMA Agentic AI Actions Co-pilot - Complete with AI Quality Validation
Demonstrates: Action → AI Analysis → BMC Updates → Next Steps

Seedstars Senior AI Engineer Assignment - Option 2
Enhanced with: Improved prompts, visual indicators, change preview, error handling, 
comprehensive logging, and AI response quality validation system
"""

import streamlit as st
import os
import json
import logging
import time
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure comprehensive logging
def setup_logging():
    """Configure structured logging for the application"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/sigma_copilot_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create specialized loggers
    app_logger = logging.getLogger("sigma.app")
    ai_logger = logging.getLogger("sigma.ai")
    bmc_logger = logging.getLogger("sigma.bmc")
    metrics_logger = logging.getLogger("sigma.metrics")
    quality_logger = logging.getLogger("sigma.quality")
    
    return app_logger, ai_logger, bmc_logger, metrics_logger, quality_logger

# Initialize loggers
app_logger, ai_logger, bmc_logger, metrics_logger, quality_logger = setup_logging()

@dataclass
class ResponseQuality:
    """Quality metrics for AI responses"""
    overall_score: float  # 0.0 to 1.0
    specificity_score: float  # How specific vs generic
    evidence_score: float  # How well supported by action data
    actionability_score: float  # How actionable the recommendations are
    consistency_score: float  # Internal consistency
    issues: List[str]  # List of quality issues found

class AIQualityValidator:
    """Validates and scores AI response quality"""
    
    def __init__(self):
        self.quality_history = []
        self.improvement_patterns = {}
    
    def validate_response(self, response: Dict[str, Any], action_data: Dict[str, Any]) -> ResponseQuality:
        """Comprehensive response quality validation"""
        
        # Score different aspects
        specificity = self._score_specificity(response)
        evidence = self._score_evidence_alignment(response, action_data)
        actionability = self._score_actionability(response)
        consistency = self._score_consistency(response)
        
        # Identify issues
        issues = self._identify_issues(response, action_data)
        
        # Calculate overall score
        overall = (specificity + evidence + actionability + consistency) / 4
        
        quality = ResponseQuality(
            overall_score=overall,
            specificity_score=specificity,
            evidence_score=evidence,
            actionability_score=actionability,
            consistency_score=consistency,
            issues=issues
        )
        
        # Track for improvement
        self.quality_history.append({
            'timestamp': datetime.now(),
            'quality': quality,
            'action_outcome': action_data.get('outcome'),
            'action_type': 'sample' if any(keyword in action_data.get('title', '').lower() 
                                         for keyword in ['trasacco', 'cctv', 'ghana police']) else 'custom'
        })
        
        # Log quality metrics
        quality_logger.info("QUALITY_ASSESSMENT", extra={
            "overall_score": overall,
            "specificity_score": specificity,
            "evidence_score": evidence,
            "actionability_score": actionability,
            "consistency_score": consistency,
            "issues_count": len(issues),
            "action_outcome": action_data.get('outcome')
        })
        
        return quality
    
    def _score_specificity(self, response: Dict[str, Any]) -> float:
        """Score how specific vs generic the response is"""
        score = 1.0
        
        # Check for generic phrases
        generic_phrases = [
            'improve', 'enhance', 'optimize', 'better', 'more effective',
            'innovative', 'cutting-edge', 'world-class', 'best-in-class',
            'leverage', 'utilize', 'implement', 'establish'
        ]
        
        text_content = json.dumps(response).lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_content)
        
        # Penalize generic language
        score -= min(generic_count * 0.1, 0.4)
        
        # Reward specific metrics, numbers, names
        if re.search(r'\d+%|\$\d+|\d+ months?|\d+ years?', text_content):
            score += 0.2
        
        # Reward specific company/location names
        africa_specific = ['ghana', 'lagos', 'accra', 'nairobi', 'african', 'west africa']
        if any(term in text_content for term in africa_specific):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_evidence_alignment(self, response: Dict[str, Any], action_data: Dict[str, Any]) -> float:
        """Score how well the response aligns with action evidence"""
        score = 0.7  # Base score
        
        action_text = (action_data.get('results', '') + ' ' + action_data.get('description', '')).lower()
        response_text = json.dumps(response).lower()
        
        # Extract key metrics/facts from action
        metrics = re.findall(r'(\d+%|\d+/\d+|\$\d+[kmb]?)', action_text)
        
        # Check if response references specific action data
        for metric in metrics:
            if metric in response_text:
                score += 0.1
        
        # Check outcome alignment
        outcome = action_data.get('outcome', '').lower()
        if outcome == 'successful':
            # Should suggest growth/scaling changes
            growth_terms = ['scale', 'expand', 'grow', 'increase', 'more']
            if any(term in response_text for term in growth_terms):
                score += 0.1
        elif outcome == 'failed':
            # Should suggest pivot/fix changes
            pivot_terms = ['pivot', 'change', 'different', 'alternative', 'reconsider']
            if any(term in response_text for term in pivot_terms):
                score += 0.1
        
        return min(1.0, score)
    
    def _score_actionability(self, response: Dict[str, Any]) -> float:
        """Score how actionable the recommendations are"""
        score = 0.5  # Base score
        
        # Check if changes are specific and implementable
        changes = response.get('changes', [])
        if not changes:
            return 0.2
        
        for change in changes:
            new_value = change.get('new', '')
            reasoning = change.get('reason', '')
            
            # Reward specific, measurable changes
            if re.search(r'\d+|\$|%|month|year|camera|customer', new_value.lower()):
                score += 0.1
            
            # Reward clear reasoning
            if len(reasoning) > 30 and any(word in reasoning.lower() for word in ['because', 'since', 'data shows', 'results indicate']):
                score += 0.05
        
        # Check next experiments
        experiments = response.get('next_experiments', [])
        for exp in experiments:
            if len(exp) > 20 and any(word in exp.lower() for word in ['test', 'try', 'pilot', 'interview', 'survey']):
                score += 0.05
        
        return min(1.0, score)
    
    def _score_consistency(self, response: Dict[str, Any]) -> float:
        """Score internal consistency of recommendations"""
        score = 1.0
        
        changes = response.get('changes', [])
        if len(changes) < 2:
            return score
        
        # Check for conflicting recommendations
        sections_modified = [change.get('section') for change in changes]
        
        # Look for logical conflicts
        # Example: Adding premium customers but reducing prices
        customer_changes = [c for c in changes if c.get('section') == 'customer_segments']
        revenue_changes = [c for c in changes if c.get('section') == 'revenue_streams']
        
        for cust_change in customer_changes:
            if 'premium' in cust_change.get('new', '').lower():
                for rev_change in revenue_changes:
                    if any(term in rev_change.get('new', '').lower() for term in ['reduce', 'lower', 'cheaper', 'discount']):
                        score -= 0.3  # Conflict detected
        
        return max(0.0, score)
    
    def _identify_issues(self, response: Dict[str, Any], action_data: Dict[str, Any]) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        # Check for missing fields
        if not response.get('analysis'):
            issues.append("Missing analysis section")
        if not response.get('changes'):
            issues.append("No changes proposed")
        if not response.get('next_experiments'):
            issues.append("No next experiments suggested")
        
        # Check for very low confidence
        changes = response.get('changes', [])
        low_confidence_count = sum(1 for c in changes if c.get('confidence', 0) < 0.6)
        if low_confidence_count > len(changes) * 0.7:
            issues.append("Too many low-confidence recommendations")
        
        # Check for generic responses
        analysis = response.get('analysis', '')
        if len(analysis) < 50:
            issues.append("Analysis too brief")
        
        generic_analysis = ['this action', 'the business', 'the company', 'improvement', 'optimization']
        if sum(1 for phrase in generic_analysis if phrase in analysis.lower()) > 3:
            issues.append("Analysis too generic")
        
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
5. Includes specific next experiments with measurable outcomes

EXAMPLE OF HIGH-QUALITY RESPONSE:
{{
    "analysis": "The 89% prediction accuracy at Trasacco Estates validates our core AI value proposition, while the $1,800 monthly revenue potential from 200 homes proves strong willingness to pay at current pricing levels.",
    "changes": [
        {{
            "section": "customer_segments",
            "type": "add",
            "current": null,
            "new": "Premium gated communities with 200+ homes in Accra metropolitan area",
            "reason": "Trasacco pilot shows this specific segment has validated demand with 91% renewal intent",
            "confidence": 0.91
        }}
    ],
    "next_experiments": [
        "Test pricing sensitivity with $12/camera rate in 2 similar communities",
        "Pilot expansion to East Legon or Airport Residential communities"
    ]
}}

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

class LoggingMixin:
    """Mixin to add structured logging capabilities"""
    
    @staticmethod
    def log_user_action(action_type: str, details: Dict[str, Any]):
        """Log user interactions"""
        app_logger.info(f"USER_ACTION: {action_type}", extra={
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
    
    @staticmethod 
    def log_ai_performance(operation: str, duration_ms: int, success: bool, details: Dict[str, Any]):
        """Log AI operation performance"""
        ai_logger.info(f"AI_PERFORMANCE: {operation}", extra={
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
    
    @staticmethod
    def log_bmc_change(section: str, change_type: str, confidence: float, auto_applied: bool):
        """Log BMC modifications"""
        bmc_logger.info(f"BMC_CHANGE: {section}.{change_type}", extra={
            "section": section,
            "change_type": change_type,
            "confidence": confidence,
            "auto_applied": auto_applied,
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    def log_session_metrics(session_id: str, metrics: Dict[str, Any]):
        """Log session-level metrics"""
        metrics_logger.info(f"SESSION_METRICS: {session_id}", extra={
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })

# Simple Business Model Canvas Class with Logging
class BusinessModelCanvas(LoggingMixin):
    """Simple BMC with 9 sections and comprehensive logging"""
    
    def __init__(self):
        # Initialize with SEMA (AI surveillance startup) sample data
        self.customer_segments = [
            "Tech-savvy homeowners in urban gated communities (ages 35-60)",
            "Property management companies managing 500+ residential units",
            "High-net-worth individuals with existing security infrastructure"
        ]
        self.value_propositions = [
            "AI-powered predictive security that prevents crimes before they happen",
            "24/7 automated surveillance monitoring with instant mobile alerts",
            "Integration with existing CCTV systems to add predictive capabilities"
        ]
        self.channels = [
            "Direct sales through dedicated teams targeting gated communities",
            "Online marketing including social media and LinkedIn outreach",
            "Partnerships with property developers and security companies"
        ]
        self.customer_relationships = [
            "Personal assistance through dedicated account managers",
            "24/7 technical support for system monitoring and maintenance",
            "Regular training sessions for security personnel and homeowners"
        ]
        self.revenue_streams = [
            "Monthly SaaS subscriptions - Basic tier at $4 per camera",
            "Premium subscription tier at $10 per month with advanced analytics",
            "One-time installation and setup services ($200-500 per property)"
        ]
        self.key_resources = [
            "Skilled AI developers with computer vision expertise",
            "Cloud computing infrastructure on AWS for real-time processing",
            "Proprietary AI algorithms for predictive crime detection"
        ]
        self.key_activities = [
            "Software development for predictive crime detection algorithms",
            "Real-time data analysis from CCTV feeds using computer vision",
            "Customer support and technical training for security systems"
        ]
        self.key_partnerships = [
            "Ghana Digital Centres Limited for business incubation support",
            "Amazon Web Services (AWS) for cloud computing infrastructure",
            "Hikvision Ghana for security camera hardware and support"
        ]
        self.cost_structure = [
            "Cloud hosting and infrastructure costs (~$6,000 monthly)",
            "Team salaries and benefits for developers (~$20,000 monthly)",
            "Marketing and customer acquisition expenses (~$8,000 monthly)"
        ]
        
        # Log BMC initialization
        self.log_user_action("bmc_initialized", {
            "total_sections": 9,
            "total_items": sum(len(getattr(self, section)) for section in self.get_section_names())
        })

    def get_section_names(self) -> List[str]:
        """Get all BMC section names"""
        return [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]

    def get_section(self, section_name: str) -> List[str]:
        """Get items from a BMC section"""
        return getattr(self, section_name, [])

    def update_section(self, section_name: str, items: List[str]):
        """Update a BMC section with new items and log the change"""
        old_items = getattr(self, section_name, [])
        setattr(self, section_name, items)
        
        # Log the section update
        self.log_user_action("bmc_section_updated", {
            "section": section_name,
            "old_count": len(old_items),
            "new_count": len(items),
            "items_added": len(items) - len(old_items)
        })

    def get_all_sections(self) -> Dict[str, List[str]]:
        """Get all BMC sections as dictionary"""
        return {section: getattr(self, section) for section in self.get_section_names()}

    def get_completeness_score(self) -> float:
        """Calculate BMC completeness percentage"""
        filled_sections = sum(1 for section in self.get_section_names() if getattr(self, section))
        return filled_sections / len(self.get_section_names())

# Quality-Enhanced AI Engine with Validation and Retry Logic
class QualityEnhancedAI(LoggingMixin):
    """AI engine with response quality validation and improvement"""
    
    def __init__(self, api_key: str):
        """Initialize with Google Gemini and quality validation"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash",
                temperature=0.3,
                max_output_tokens=1500,
                timeout=30
            )
            self.SystemMessage = SystemMessage
            self.HumanMessage = HumanMessage
            self.quality_validator = AIQualityValidator()
            self.max_retries = 2
            
            # Log successful AI initialization
            self.log_ai_performance("ai_initialization", 0, True, {
                "model": "gemini-2.0-flash",
                "api_key_length": len(api_key),
                "quality_validation": True
            })
            
        except ImportError as e:
            self.log_ai_performance("ai_initialization", 0, False, {"error": str(e)})
            raise ImportError(f"Missing dependencies: {e}. Run: pip install langchain langchain-google-genai")

    def analyze_action_with_quality_control(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas) -> Tuple[Dict[str, Any], ResponseQuality]:
        """Analyze action with quality validation and retry logic"""
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())[:8]
        
        # Log analysis start
        self.log_ai_performance("quality_analysis_started", 0, True, {
            "analysis_id": analysis_id,
            "action_title": action_data.get('title', 'Unknown'),
            "action_outcome": action_data.get('outcome', 'Unknown'),
            "max_retries": self.max_retries
        })
        
        last_response = None
        quality = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Get AI response
                if attempt == 0:
                    response = self._get_initial_response(action_data, bmc, analysis_id)
                else:
                    response = self._retry_with_improvement(action_data, bmc, last_response, quality, analysis_id)
                
                # Validate quality
                quality = self.quality_validator.validate_response(response, action_data)
                
                # Log quality metrics for this attempt
                self.log_ai_performance(f"quality_check_attempt_{attempt + 1}", 0, True, {
                    "analysis_id": analysis_id,
                    "overall_score": quality.overall_score,
                    "specificity_score": quality.specificity_score,
                    "evidence_score": quality.evidence_score,
                    "actionability_score": quality.actionability_score,
                    "issues_count": len(quality.issues),
                    "should_retry": self.quality_validator.should_retry(quality),
                    "attempt": attempt + 1
                })
                
                # Check if good enough or max retries reached
                if not self.quality_validator.should_retry(quality) or attempt == self.max_retries:
                    break
                
                last_response = response
                
            except Exception as e:
                # Log retry failure
                self.log_ai_performance(f"quality_retry_failed_attempt_{attempt + 1}", 0, False, {
                    "analysis_id": analysis_id,
                    "error": str(e),
                    "attempt": attempt + 1
                })
                
                if attempt == self.max_retries:
                    # Return fallback response on final failure
                    response = self._create_fallback_response(action_data)
                    quality = ResponseQuality(
                        overall_score=0.3,
                        specificity_score=0.3,
                        evidence_score=0.3,
                        actionability_score=0.3,
                        consistency_score=0.3,
                        issues=["Quality validation failed", "Using fallback response"]
                    )
                    break
        
        # Log final analysis completion
        total_duration = (time.time() - start_time) * 1000
        self.log_ai_performance("quality_analysis_completed", int(total_duration), True, {
            "analysis_id": analysis_id,
            "final_quality_score": quality.overall_score,
            "attempts_used": attempt + 1,
            "total_duration_ms": int(total_duration),
            "quality_improved": attempt > 0
        })
        
        return response, quality

    def _get_initial_response(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas, analysis_id: str) -> Dict[str, Any]:
        """Get initial AI response with enhanced prompting"""
        
        # Enhanced system prompt with few-shot examples
        system_prompt = """You are SIGMA's AI co-pilot helping founders validate business assumptions through experiments.

Analyze completed actions and suggest specific Business Model Canvas updates based on what was learned.

EXAMPLE ANALYSIS 1:
Action: "Customer interviews with 50 Lagos fintech users"
Outcome: "Failed - 80% couldn't afford $15/month subscription"
Analysis: "This invalidates our pricing assumption and suggests we're targeting wrong customer segment or need freemium model."
Changes: [
    {
        "section": "customer_segments",
        "type": "modify",
        "current": "Middle-income Lagos professionals",
        "new": "Budget-conscious Lagos users needing micro-payment solutions",
        "reason": "Interview data shows price sensitivity much higher than assumed",
        "confidence": 0.89
    },
    {
        "section": "revenue_streams", 
        "type": "add",
        "current": null,
        "new": "Freemium model with premium features at $3/month",
        "reason": "80% expressed willingness to pay $3/month for basic features",
        "confidence": 0.82
    }
]

EXAMPLE ANALYSIS 2:
Action: "3-month pilot with 200 farmers using AgriTech platform"
Outcome: "Successful - 45% yield increase, 92% satisfaction"
Analysis: "Strong validation of core value proposition and market fit for smallholder farmers."
Changes: [
    {
        "section": "value_propositions",
        "type": "modify", 
        "current": "Improve farm productivity through technology",
        "new": "Proven 45% yield increase through AI-powered farming recommendations",
        "reason": "Pilot data provides specific, measurable value proposition",
        "confidence": 0.95
    }
]

Return ONLY valid JSON in this exact format:
{
    "analysis": "2-3 sentence analysis of what this action outcome means for the business model",
    "changes": [
        {
            "section": "customer_segments|value_propositions|channels|customer_relationships|revenue_streams|key_resources|key_activities|key_partnerships|cost_structure",
            "type": "add|modify|remove", 
            "current": "existing item being modified/removed (null for add operations)",
            "new": "new item to add or replacement text",
            "reason": "clear explanation of why this change makes sense based on the action outcome",
            "confidence": 0.85
        }
    ],
    "next_experiments": [
        "Specific actionable next experiment to validate further assumptions",
        "Another logical next step to build on these learnings"
    ]
}

Rules:
- Only suggest changes with confidence > 0.6
- Focus on what the action outcome actually validates or invalidates
- Be specific - avoid generic suggestions
- Limit to 3-4 most important changes maximum
- Confidence scores should reflect evidence strength: 0.9+ for clear quantitative validation, 0.7-0.8 for solid qualitative insights, 0.6-0.7 for reasonable inferences"""

        # Create concise BMC summary
        current_bmc = f"""Current Business Model Canvas Summary:
Customer Segments: {len(bmc.customer_segments)} segments defined
Value Propositions: {len(bmc.value_propositions)} propositions  
Channels: {len(bmc.channels)} channels
Revenue Streams: {len(bmc.revenue_streams)} revenue models
Key Resources: {len(bmc.key_resources)} resources
Key Activities: {len(bmc.key_activities)} activities
Key Partnerships: {len(bmc.key_partnerships)} partnerships
Cost Structure: {len(bmc.cost_structure)} cost components

Key Current Elements:
- Primary Customer: {bmc.customer_segments[0] if bmc.customer_segments else 'Not defined'}
- Main Value Prop: {bmc.value_propositions[0] if bmc.value_propositions else 'Not defined'}
- Top Revenue Stream: {bmc.revenue_streams[0] if bmc.revenue_streams else 'Not defined'}"""

        user_prompt = f"""COMPLETED ACTION/EXPERIMENT:
Title: {action_data['title']}
Outcome: {action_data['outcome']} 
Description: {action_data['description']}
Key Results: {action_data['results']}

{current_bmc}

Based on this action outcome, what specific updates should be made to the Business Model Canvas? What assumptions were validated or invalidated?

Return only the JSON response."""

        # Call Gemini API with timing
        api_start = time.time()
        messages = [
            self.SystemMessage(content=system_prompt),
            self.HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        api_duration = (time.time() - api_start) * 1000
        
        # Log API call performance
        self.log_ai_performance("quality_api_call", int(api_duration), True, {
            "analysis_id": analysis_id,
            "response_length": len(response.content),
            "model": "gemini-2.0-flash"
        })
        
        return self._parse_response(response.content)

    def _retry_with_improvement(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas, 
                               last_response: Dict[str, Any], quality: ResponseQuality, analysis_id: str) -> Dict[str, Any]:
        """Retry analysis with improvement prompt"""
        
        improvement_prompt = self.quality_validator.get_improvement_prompt(quality, last_response)
        
        # Use improvement prompt as system message
        messages = [
            self.SystemMessage(content=improvement_prompt),
            self.HumanMessage(content=f"""
            ACTION: {action_data['title']}
            OUTCOME: {action_data['outcome']}
            RESULTS: {action_data['results']}
            
            BMC CONTEXT: {len(bmc.get_all_sections())} sections with varying completion
            
            Provide an improved analysis addressing the quality issues mentioned.
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
            return last_response  # Fall back to previous response

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response content to JSON"""
        content = content.strip()
        
        # Extract JSON from response
        if "```json" in content:
            json_part = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_part = content.split("```")[1].split("```")[0].strip()
        else:
            # Try to find JSON object in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_part = content[start:end]
            else:
                json_part = content
        
        # Parse JSON response
        result = json.loads(json_part)
        
        # Validate required fields
        if 'analysis' not in result:
            result['analysis'] = "Analysis completed successfully"
        if 'changes' not in result:
            result['changes'] = []
        if 'next_experiments' not in result:
            result['next_experiments'] = ["Continue testing current approach"]
        
        return result

    def _create_fallback_response(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback response when all retries fail"""
        return {
            "analysis": f"Analysis of '{action_data.get('title', 'the action')}' completed with {action_data.get('outcome', 'unknown')} outcome. Manual review recommended for detailed insights.",
            "changes": [],
            "next_experiments": [
                "Review action results manually for key insights",
                "Consider running similar experiment with more structured data collection"
            ]
        }

    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get quality metrics for dashboard"""
        return self.quality_validator.get_quality_dashboard_data()

# Enhanced UI Helper Functions
def display_confidence_indicator(confidence: float, label: str = "Confidence"):
    """Display visual confidence indicator with progress bar"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Color-coded progress bar
        if confidence >= 0.8:
            st.success(f"**{label}:** High ({confidence:.0%})")
        elif confidence >= 0.7:
            st.warning(f"**{label}:** Medium ({confidence:.0%})")
        else:
            st.error(f"**{label}:** Low ({confidence:.0%})")
    
    with col2:
        st.progress(confidence)

def display_quality_indicator(quality: ResponseQuality):
    """Display AI response quality indicator"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if quality.overall_score >= 0.8:
            st.success(f"**AI Quality:** Excellent ({quality.overall_score:.0%})")
        elif quality.overall_score >= 0.6:
            st.warning(f"**AI Quality:** Good ({quality.overall_score:.0%})")
        elif quality.overall_score >= 0.4:
            st.info(f"**AI Quality:** Improved via retry ({quality.overall_score:.0%})")
        else:
            st.error(f"**AI Quality:** Basic ({quality.overall_score:.0%})")
    
    with col2:
        st.progress(quality.overall_score)
    
    # Show quality details in expander
    if quality.issues:
        with st.expander("Quality Details", expanded=False):
            st.write("**Quality Breakdown:**")
            st.write(f"• Specificity: {quality.specificity_score:.0%}")
            st.write(f"• Evidence Alignment: {quality.evidence_score:.0%}")
            st.write(f"• Actionability: {quality.actionability_score:.0%}")
            st.write(f"• Consistency: {quality.consistency_score:.0%}")
            if quality.issues:
                st.write("**Issues Addressed:**")
                for issue in quality.issues:
                    st.write(f"• {issue}")

def preview_change(change: Dict[str, Any], current_items: List[str]) -> Dict[str, List[str]]:
    """Generate before/after preview for a proposed change"""
    before = current_items.copy()
    after = current_items.copy()
    
    if change["type"] == "add":
        after.append(change["new"])
    elif change["type"] == "modify" and change.get("current"):
        try:
            idx = after.index(change["current"])
            after[idx] = change["new"]
        except ValueError:
            after.append(change["new"])
    elif change["type"] == "remove" and change.get("current"):
        try:
            after.remove(change["current"])
        except ValueError:
            pass
    
    return {"before": before, "after": after}

def display_change_preview(changes: List[Dict[str, Any]], bmc: BusinessModelCanvas):
    """Display before/after preview for all proposed changes"""
    if not changes:
        return
    
    st.subheader("Preview Changes")
    st.caption("See how your Business Model Canvas will look after applying these changes")
    
    for i, change in enumerate(changes):
        section_display = change['section'].replace('_', ' ').title()
        current_items = bmc.get_section(change['section'])
        preview = preview_change(change, current_items)
        
        with st.expander(f"{section_display} - {change['type'].title()}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before:**")
                if preview["before"]:
                    for item in preview["before"]:
                        if change["type"] == "modify" and item == change.get("current"):
                            st.markdown(f"• ~~{item}~~ *(will be changed)*")
                        elif change["type"] == "remove" and item == change.get("current"):
                            st.markdown(f"• ~~{item}~~ *(will be removed)*")
                        else:
                            st.write(f"• {item}")
                else:
                    st.write("*No items*")
            
            with col2:
                st.write("**After:**")
                if preview["after"]:
                    for item in preview["after"]:
                        if change["type"] == "add" and item == change["new"]:
                            st.markdown(f"• **{item}** *(new)*")
                        elif change["type"] == "modify" and item == change["new"]:
                            st.markdown(f"• **{item}** *(updated)*")
                        else:
                            st.write(f"• {item}")
                else:
                    st.write("*No items*")

# Session metrics tracking
class SessionMetrics(LoggingMixin):
    """Track session-level metrics and usage patterns"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.actions_analyzed = 0
        self.changes_proposed = 0
        self.changes_applied = 0
        self.auto_mode_usage = 0
        self.sample_actions_used = 0
        self.custom_actions_used = 0
        self.quality_retries = 0
        
        # Log session start
        self.log_session_metrics(self.session_id, {
            "event": "session_started",
            "start_time": self.start_time.isoformat()
        })
    
    def record_action_analyzed(self, action_type: str, outcome: str, quality_score: float = None, retries_used: int = 0):
        """Record an action analysis"""
        self.actions_analyzed += 1
        if action_type == "sample":
            self.sample_actions_used += 1
        else:
            self.custom_actions_used += 1
        
        if retries_used > 0:
            self.quality_retries += 1
        
        self.log_user_action("action_analyzed", {
            "session_id": self.session_id,
            "action_type": action_type,
            "outcome": outcome,
            "total_analyzed": self.actions_analyzed,
            "quality_score": quality_score,
            "retries_used": retries_used
        })
    
    def record_changes_proposed(self, count: int, avg_confidence: float):
        """Record proposed changes"""
        self.changes_proposed += count
        
        self.log_user_action("changes_proposed", {
            "session_id": self.session_id,
            "count": count,
            "avg_confidence": avg_confidence,
            "total_proposed": self.changes_proposed
        })
    
    def record_changes_applied(self, count: int, auto_applied: bool):
        """Record applied changes"""
        self.changes_applied += count
        if auto_applied:
            self.auto_mode_usage += 1
        
        self.log_bmc_change("multiple", "apply", 1.0, auto_applied)
        self.log_user_action("changes_applied", {
            "session_id": self.session_id,
            "count": count,
            "auto_applied": auto_applied,
            "total_applied": self.changes_applied
        })
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "actions_analyzed": self.actions_analyzed,
            "changes_proposed": self.changes_proposed,
            "changes_applied": self.changes_applied,
            "auto_mode_usage": self.auto_mode_usage,
            "quality_retries": self.quality_retries,
            "sample_vs_custom": {
                "sample_actions": self.sample_actions_used,
                "custom_actions": self.custom_actions_used
            },
            "engagement_score": min(self.actions_analyzed * 2 + self.changes_applied, 10)
        }

# Sample actions for quick testing
def get_sample_actions():
    """Return sample actions for demo purposes"""
    return {
        "Trasacco Estates Pilot (Successful)": {
            "title": "3-Month AI Surveillance Pilot at Trasacco Estates",
            "outcome": "Successful",
            "description": "Deployed SEMA's predictive surveillance system across Trasacco Estates Phase 4, covering 200 homes with 150 existing CCTV cameras. Implemented AI-driven threat detection and real-time alerts.",
            "results": """
EXCEPTIONAL PILOT PERFORMANCE:

Security Impact:
• 89% crime prediction accuracy (exceeded 75% target)
• 23 security incidents prevented in 3 months  
• False positive rate: Only 12% (industry standard 35%)
• System uptime: 99.7% across all camera feeds

Customer Validation:
• 91% resident satisfaction score (surveyed 180 households)
• 87% activated mobile alerts within first month
• Property management: "Game-changing technology"
• 91% renewal intent for permanent installation

Business Metrics:
• Monthly recurring revenue potential: $1,800 from this community
• Customer acquisition cost: $45 per household (20% below budget)
• 5 qualified referrals generated from word-of-mouth
• Property value increase: 8% cited by real estate agents
"""
        },
        
        "CCTV Integration Testing (Failed)": {
            "title": "Legacy CCTV System Integration with 5 Security Companies",
            "outcome": "Failed", 
            "description": "Attempted to integrate SEMA's AI algorithms with existing CCTV systems used by 5 major Ghanaian security companies to demonstrate plug-and-play compatibility.",
            "results": """
INTEGRATION FAILURE ANALYSIS:

Technical Issues:
• Only 2 out of 5 security company systems successfully integrated (40%)
• 3 companies using proprietary Chinese camera protocols incompatible
• 60% of existing cameras output in incompatible video formats
• Legacy DVR systems cannot support cloud integration requirements

Market Reality:
• 78% of installations are over 5 years old (legacy systems)
• Security companies resistant to cloud-based solutions (privacy concerns)
• Network security policies prevent third-party cloud access
• Underestimated diversity of existing infrastructure in Ghana

Business Impact:
• Market size reduced by 65% (incompatible customers)
• Additional $15,000 development costs for compatibility layer
• 6-8 month delay in partnership expansion strategy
• Must pivot from retrofit market to new installation focus
"""
        },
        
        "Ghana Police Partnership (Inconclusive)": {
            "title": "Ghana Police Service Smart City Partnership Discussions",
            "outcome": "Inconclusive",
            "description": "4-month negotiations with Ghana Police Service to integrate SEMA's technology into their Smart City crime prevention initiative involving multiple government departments.",
            "results": """
MIXED PARTNERSHIP SIGNALS:

Positive Indicators:
• Strong technical endorsement from GPS Technology Division
• Written support from 3 Regional Police Commanders
• World Bank Smart Cities fund shows co-financing interest
• Minister of Interior expressed public support

Significant Challenges:
• 18-month government tender process required
• Data privacy concerns from Attorney General's office
• Budget allocation pending National Assembly approval
• Competition from 2 international firms with gov relationships
• Leadership change during negotiation period

Current Status:
• Technical requirements: 95% agreement
• Commercial terms: 60% consensus  
• Legal framework: 40% complete
• Potential contract value: $2.4M over 3 years
• Success probability assessment: 45%
"""
        }
    }

# Streamlit App Configuration
st.set_page_config(
    page_title="SIGMA Agentic AI Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main Streamlit application with comprehensive logging and quality validation"""
    
    # Initialize session metrics
    if 'session_metrics' not in st.session_state:
        st.session_state.session_metrics = SessionMetrics()
    
    # Header
    st.title("SIGMA Agentic AI Actions Co-pilot")
    st.markdown("**Seedstars Assignment**: Complete experiments → AI updates your business model → Get next steps")
    
    # Check API key with enhanced error handling
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        app_logger.error("No Google API key found in environment")
        st.error("No Google API key found")
        st.code("""
# Create .env file with:
GOOGLE_API_KEY=your_actual_google_api_key_here

# Get API key from: https://makersuite.google.com/app/apikey
        """)
        st.stop()
    elif api_key == "your_google_api_key_here":
        app_logger.error("Placeholder API key detected")
        st.error("Please replace the placeholder API key in your .env file with your actual Google API key")
        st.stop()

    # Initialize session state
    if 'bmc' not in st.session_state:
        st.session_state.bmc = BusinessModelCanvas()
    if 'ai' not in st.session_state:
        try:
            st.session_state.ai = QualityEnhancedAI(api_key)
        except Exception as e:
            app_logger.error(f"Failed to initialize AI: {e}")
            st.error(f"Failed to initialize AI: {e}")
            st.stop()
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False
    if 'last_recommendation' not in st.session_state:
        st.session_state.last_recommendation = None
    if 'last_quality' not in st.session_state:
        st.session_state.last_quality = None

    # Sidebar with enhanced metrics including quality data
    with st.sidebar:
        st.subheader("Session Info")
        metrics = st.session_state.session_metrics.get_session_summary()
        
        # Session metrics
        st.write(f"Session ID: `{metrics['session_id']}`")
        st.write(f"Actions Analyzed: {metrics['actions_analyzed']}")
        st.write(f"Changes Applied: {metrics['changes_applied']}")
        st.write(f"Duration: {metrics['duration_seconds']:.0f}s")
        
        # Quality metrics
        quality_data = st.session_state.ai.get_quality_dashboard_data()
        if quality_data:
            st.subheader("AI Quality Metrics")
            st.metric("Response Quality", f"{quality_data['average_quality_score']:.0%}")
            st.metric("Retry Rate", f"{quality_data['retry_rate']:.0%}")
            st.metric("Total Responses", quality_data['total_responses'])
            if metrics['quality_retries'] > 0:
                st.metric("Quality Improvements", metrics['quality_retries'])
        
        if st.button("View Logs"):
            st.write("Check logs/ directory for detailed logging")

    # Auto-mode toggle with enhanced description and logging
    auto_mode_changed = st.toggle(
        "**Auto-mode**: Apply high-confidence changes (>80%) automatically", 
        value=st.session_state.auto_mode,
        help="When enabled, changes with >80% confidence will be applied automatically to your BMC"
    )
    
    if auto_mode_changed != st.session_state.auto_mode:
        st.session_state.auto_mode = auto_mode_changed
        app_logger.info(f"Auto-mode toggled: {auto_mode_changed}")

    # Main layout
    col1, col2 = st.columns([1.5, 1.2])
    
    # Left Column: Business Model Canvas Display
    with col1:
        st.subheader("Current Business Model Canvas")
        st.caption("Real-time view of your business model (SEMA AI Surveillance startup)")
        
        # BMC sections with icons
        bmc_sections = [
            ("key_partnerships", "Key Partnerships"),
            ("key_activities", "Key Activities"), 
            ("key_resources", "Key Resources"),
            ("value_propositions", "Value Propositions"),
            ("customer_relationships", "Customer Relationships"),
            ("channels", "Channels"),
            ("customer_segments", "Customer Segments"),
            ("cost_structure", "Cost Structure"),
            ("revenue_streams", "Revenue Streams")
        ]
        
        # Display BMC in 3x3 grid layout
        for i in range(0, 9, 3):
            cols = st.columns(3)
            
            for j, (section_key, section_title) in enumerate(bmc_sections[i:i+3]):
                with cols[j]:
                    st.markdown(f"**{section_title}**")
                    
                    items = st.session_state.bmc.get_section(section_key)
                    if items:
                        for item in items:
                            st.markdown(f"• {item}")
                    else:
                        st.markdown("*No items defined*")
                    
                    st.markdown("")  # Add spacing

    # Right Column: Action Input & Analysis
    with col2:
        st.subheader("Log Completed Action")
        
        # Sample action selector
        sample_actions = get_sample_actions()
        use_sample = st.selectbox(
            "Choose sample action or create custom:",
            ["Custom Action"] + list(sample_actions.keys())
        )
        
        if use_sample != "Custom Action":
            # Use selected sample action
            action_data = sample_actions[use_sample]
            action_type = "sample"
            
            st.info(f"**Sample Action Selected:** {action_data['title']}")
            
            with st.expander("View Action Details", expanded=False):
                st.write(f"**Outcome:** {action_data['outcome']}")
                st.write(f"**Description:** {action_data['description']}")
                st.write("**Results:**")
                st.code(action_data['results'])
        
        else:
            # Custom action form
            action_type = "custom"
            with st.form("custom_action_form"):
                st.write("**Create Custom Action:**")
                
                action_data = {
                    "title": st.text_input(
                        "Action/Experiment Title", 
                        placeholder="e.g., Customer interviews in Lagos"
                    ),
                    "outcome": st.selectbox("Outcome", ["Successful", "Failed", "Inconclusive"]),
                    "description": st.text_area(
                        "What did you do?", 
                        placeholder="Describe the action/experiment you completed"
                    ),
                    "results": st.text_area(
                        "Results & Key Learnings", 
                        placeholder="What did you learn? Include metrics, feedback, insights..."
                    )
                }
                
                form_submitted = st.form_submit_button("Use Custom Action")
                
                if form_submitted and not all([action_data["title"], action_data["description"], action_data["results"]]):
                    st.error("Please fill in all fields for custom action")
                    st.stop()
        
        # Analyze Action Button
        if st.button("Analyze Action & Update BMC", use_container_width=True, type="primary"):
            if not action_data.get("title") or not action_data.get("results"):
                st.error("Action title and results are required")
            else:
                with st.spinner("AI analyzing your action with quality validation..."):
                    # Get AI recommendation with quality control
                    recommendation, quality = st.session_state.ai.analyze_action_with_quality_control(
                        action_data, st.session_state.bmc
                    )
                    
                    st.session_state.last_recommendation = recommendation
                    st.session_state.last_quality = quality
                    
                    # Record metrics including quality data
                    retries_used = 1 if quality.overall_score < 0.6 else 0  # Estimate retries based on quality
                    st.session_state.session_metrics.record_action_analyzed(
                        action_type, action_data.get('outcome', 'Unknown'), 
                        quality.overall_score, retries_used
                    )
                    
                    # Record changes proposed
                    if recommendation["changes"]:
                        avg_confidence = sum(c.get('confidence', 0) for c in recommendation["changes"]) / len(recommendation["changes"])
                        st.session_state.session_metrics.record_changes_proposed(len(recommendation["changes"]), avg_confidence)
                    
                    # Display AI Analysis with Quality Indicator
                    st.success("Analysis Complete!")
                    
                    # Show quality indicator
                    display_quality_indicator(quality)
                    
                    with st.expander("AI Analysis", expanded=True):
                        st.write(recommendation["analysis"])
                    
                    # Show proposed changes with enhanced visuals
                    if recommendation["changes"]:
                        st.subheader("Proposed BMC Updates")
                        
                        # Display change preview
                        display_change_preview(recommendation["changes"], st.session_state.bmc)
                        
                        high_confidence_changes = []
                        changes_applied = 0
                        
                        # Show individual changes with confidence indicators
                        st.subheader("Individual Changes")
                        for i, change in enumerate(recommendation["changes"]):
                            confidence = change.get("confidence", 0)
                            
                            # Track high confidence changes
                            if confidence >= 0.8:
                                high_confidence_changes.append(change)
                            
                            # Display change with visual confidence indicator
                            section_display = change['section'].replace('_', ' ').title()
                            
                            with st.container():
                                st.markdown(f"**{section_display}** - {change['type'].title()}")
                                
                                # Visual confidence indicator
                                display_confidence_indicator(confidence)
                                
                                st.write(f"**Change:** {change['new']}")
                                st.write(f"**Reasoning:** {change['reason']}")
                                st.markdown("---")
                        
                        # Auto-mode application
                        if st.session_state.auto_mode and high_confidence_changes:
                            st.info(f"**Auto-mode Active**: Applying {len(high_confidence_changes)} high-confidence changes...")
                            
                            for change in high_confidence_changes:
                                section = change["section"]
                                current_items = st.session_state.bmc.get_section(section)
                                
                                if change["type"] == "add":
                                    if change["new"] not in current_items:
                                        current_items.append(change["new"])
                                        changes_applied += 1
                                        
                                elif change["type"] == "modify" and change.get("current"):
                                    try:
                                        idx = current_items.index(change["current"])
                                        current_items[idx] = change["new"]
                                        changes_applied += 1
                                    except ValueError:
                                        current_items.append(change["new"])
                                        changes_applied += 1
                                        
                                elif change["type"] == "remove" and change.get("current"):
                                    try:
                                        current_items.remove(change["current"])
                                        changes_applied += 1
                                    except ValueError:
                                        pass
                                
                                st.session_state.bmc.update_section(section, current_items)
                            
                            if changes_applied > 0:
                                st.session_state.session_metrics.record_changes_applied(changes_applied, auto_applied=True)
                                st.success(f"Auto-applied {changes_applied} high-confidence changes!")
                                time.sleep(1)  # Brief pause before rerun
                                st.rerun()
                        
                        # Manual controls (if auto-mode is off or there are non-auto changes)
                        elif not st.session_state.auto_mode:
                            col_apply, col_reject = st.columns(2)
                            
                            with col_apply:
                                if st.button("Apply All Changes", use_container_width=True):
                                    for change in recommendation["changes"]:
                                        if change.get("confidence", 0) >= 0.6:  # Apply medium+ confidence
                                            section = change["section"]
                                            current_items = st.session_state.bmc.get_section(section)
                                            
                                            if change["type"] == "add":
                                                if change["new"] not in current_items:
                                                    current_items.append(change["new"])
                                                    changes_applied += 1
                                            elif change["type"] == "modify" and change.get("current"):
                                                try:
                                                    idx = current_items.index(change["current"])
                                                    current_items[idx] = change["new"]
                                                    changes_applied += 1
                                                except ValueError:
                                                    current_items.append(change["new"])
                                                    changes_applied += 1
                                            elif change["type"] == "remove" and change.get("current"):
                                                try:
                                                    current_items.remove(change["current"])
                                                    changes_applied += 1
                                                except ValueError:
                                                    pass
                                            
                                            st.session_state.bmc.update_section(section, current_items)
                                    
                                    if changes_applied > 0:
                                        st.session_state.session_metrics.record_changes_applied(changes_applied, auto_applied=False)
                                    st.success(f"Applied {changes_applied} changes to your business model!")
                                    time.sleep(1)  # Brief pause before rerun
                                    st.rerun()
                            
                            with col_reject:
                                if st.button("Reject Changes", use_container_width=True):
                                    app_logger.info("User rejected all proposed changes")
                                    st.info("Changes rejected. BMC remains unchanged.")
                    
                    else:
                        st.info("No BMC changes suggested based on this action.")
                    
                    # Show next experiments
                    if recommendation.get("next_experiments"):
                        st.subheader("Suggested Next Experiments")
                        for i, experiment in enumerate(recommendation["next_experiments"], 1):
                            st.write(f"**{i}.** {experiment}")

    # Footer with enhanced session summary including quality metrics
    st.markdown("---")
    
    # Display session metrics summary
    metrics = st.session_state.session_metrics.get_session_summary()
    quality_data = st.session_state.ai.get_quality_dashboard_data()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Actions Analyzed", metrics['actions_analyzed'])
    with col2:
        st.metric("Changes Applied", metrics['changes_applied'])
    with col3:
        st.metric("Session Duration", f"{metrics['duration_seconds']:.0f}s")
    with col4:
        st.metric("Engagement Score", f"{metrics['engagement_score']}/10")
    with col5:
        if quality_data:
            st.metric("AI Quality", f"{quality_data['average_quality_score']:.0%}")
        else:
            st.metric("AI Quality", "N/A")
    
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>SIGMA Agentic AI Actions Co-pilot</strong> | 
        Seedstars Senior AI Engineer Assignment | 
        Enhanced: Prompts, visuals, preview, error handling, logging, AI quality validation
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()