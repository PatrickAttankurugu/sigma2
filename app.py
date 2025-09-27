"""
SIGMA Agentic AI Actions Co-pilot - Enhanced with Comprehensive Logging
Demonstrates: Action → AI Analysis → BMC Updates → Next Steps

Seedstars Senior AI Engineer Assignment - Option 2
Enhanced with: Improved prompts, visual indicators, change preview, error handling, comprehensive logging
"""

import streamlit as st
import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import uuid

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
    
    return app_logger, ai_logger, bmc_logger, metrics_logger

# Initialize loggers
app_logger, ai_logger, bmc_logger, metrics_logger = setup_logging()

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

# Enhanced AI Engine with Comprehensive Logging
class SimpleAI(LoggingMixin):
    """Enhanced AI engine with improved prompting, error handling, and comprehensive logging"""
    
    def __init__(self, api_key: str):
        """Initialize with Google Gemini and log configuration"""
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
            
            # Log successful AI initialization
            self.log_ai_performance("ai_initialization", 0, True, {
                "model": "gemini-2.0-flash",
                "api_key_length": len(api_key)
            })
            
        except ImportError as e:
            self.log_ai_performance("ai_initialization", 0, False, {"error": str(e)})
            raise ImportError(f"Missing dependencies: {e}. Run: pip install langchain langchain-google-genai")

    def analyze_action(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas) -> Dict[str, Any]:
        """Analyze completed action and suggest BMC updates with comprehensive logging"""
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())[:8]
        
        # Log analysis start
        self.log_ai_performance("analysis_started", 0, True, {
            "analysis_id": analysis_id,
            "action_title": action_data.get('title', 'Unknown'),
            "action_outcome": action_data.get('outcome', 'Unknown'),
            "action_length": len(action_data.get('results', '')),
            "bmc_completeness": bmc.get_completeness_score()
        })
        
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

        try:
            # Call Gemini API with timing
            api_start = time.time()
            messages = [
                self.SystemMessage(content=system_prompt),
                self.HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            api_duration = (time.time() - api_start) * 1000
            
            # Log API call performance
            self.log_ai_performance("api_call", int(api_duration), True, {
                "analysis_id": analysis_id,
                "response_length": len(response.content),
                "model": "gemini-2.0-flash"
            })
            
            content = response.content.strip()
            
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
            
            # Calculate total analysis time
            total_duration = (time.time() - start_time) * 1000
            
            # Log successful analysis
            self.log_ai_performance("analysis_completed", int(total_duration), True, {
                "analysis_id": analysis_id,
                "changes_count": len(result['changes']),
                "avg_confidence": sum(c.get('confidence', 0) for c in result['changes']) / max(len(result['changes']), 1),
                "high_confidence_changes": sum(1 for c in result['changes'] if c.get('confidence', 0) >= 0.8),
                "next_experiments_count": len(result['next_experiments'])
            })
            
            return result
            
        except json.JSONDecodeError as e:
            duration = (time.time() - start_time) * 1000
            self.log_ai_performance("json_parsing_failed", int(duration), False, {
                "analysis_id": analysis_id,
                "error": str(e),
                "raw_response_length": len(content) if 'content' in locals() else 0
            })
            return self._create_parsing_error_response(str(e))
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log_ai_performance("analysis_failed", int(duration), False, {
                "analysis_id": analysis_id,
                "error_type": type(e).__name__,
                "error": str(e)
            })
            return self._create_api_error_response(str(e))

    def _create_parsing_error_response(self, error_detail: str) -> Dict[str, Any]:
        """Create user-friendly response for JSON parsing errors"""
        return {
            "analysis": "AI response was unclear. This sometimes happens with complex actions. Try simplifying your action description or running the analysis again.",
            "changes": [],
            "next_experiments": ["Simplify action description and try again", "Break complex actions into smaller experiments"]
        }

    def _create_api_error_response(self, error_detail: str) -> Dict[str, Any]:
        """Create user-friendly response for API errors"""
        if "rate limit" in error_detail.lower():
            return {
                "analysis": "AI is experiencing high demand. Please wait 30-60 seconds and try again.",
                "changes": [],
                "next_experiments": ["Wait a moment and retry the analysis"]
            }
        elif "api key" in error_detail.lower() or "authentication" in error_detail.lower():
            return {
                "analysis": "API key issue detected. Please check your Google API key in the .env file and ensure it's valid.",
                "changes": [],
                "next_experiments": ["Verify API key configuration", "Get a new API key from Google AI Studio"]
            }
        elif "timeout" in error_detail.lower():
            return {
                "analysis": "Analysis timed out. This usually happens with very long action descriptions. Try simplifying your input.",
                "changes": [],
                "next_experiments": ["Shorten action description", "Focus on key results only"]
            }
        else:
            return {
                "analysis": f"Technical issue occurred during analysis. Please try again or contact support if the problem persists.",
                "changes": [],
                "next_experiments": ["Try the analysis again", "Check internet connection"]
            }

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
        
        # Log session start
        self.log_session_metrics(self.session_id, {
            "event": "session_started",
            "start_time": self.start_time.isoformat()
        })
    
    def record_action_analyzed(self, action_type: str, outcome: str):
        """Record an action analysis"""
        self.actions_analyzed += 1
        if action_type == "sample":
            self.sample_actions_used += 1
        else:
            self.custom_actions_used += 1
        
        self.log_user_action("action_analyzed", {
            "session_id": self.session_id,
            "action_type": action_type,
            "outcome": outcome,
            "total_analyzed": self.actions_analyzed
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
    """Main Streamlit application with comprehensive logging"""
    
    # Initialize session metrics
    if 'session_metrics' not in st.session_state:
        st.session_state.session_metrics = SessionMetrics()
    
    # Header
    st.title("SIGMA Agentic AI Actions Co-pilot")
    st.markdown("**Seedstars Assignment**: Complete experiments → AI updates your business model → Get next steps")
    
    # Optional: Display session info in sidebar for debugging
    with st.sidebar:
        st.subheader("Session Info")
        metrics = st.session_state.session_metrics.get_session_summary()
        st.write(f"Session ID: `{metrics['session_id']}`")
        st.write(f"Actions Analyzed: {metrics['actions_analyzed']}")
        st.write(f"Changes Applied: {metrics['changes_applied']}")
        st.write(f"Duration: {metrics['duration_seconds']:.0f}s")
        
        if st.button("View Logs"):
            st.write("Check logs/ directory for detailed logging")
    
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
            st.session_state.ai = SimpleAI(api_key)
        except Exception as e:
            app_logger.error(f"Failed to initialize AI: {e}")
            st.error(f"Failed to initialize AI: {e}")
            st.stop()
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False
    if 'last_recommendation' not in st.session_state:
        st.session_state.last_recommendation = None

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
                # Record action analysis start
                st.session_state.session_metrics.record_action_analyzed(action_type, action_data.get('outcome', 'Unknown'))
                
                with st.spinner("AI analyzing your action..."):
                    # Get AI recommendation
                    recommendation = st.session_state.ai.analyze_action(action_data, st.session_state.bmc)
                    st.session_state.last_recommendation = recommendation
                    
                    # Record changes proposed
                    if recommendation["changes"]:
                        avg_confidence = sum(c.get('confidence', 0) for c in recommendation["changes"]) / len(recommendation["changes"])
                        st.session_state.session_metrics.record_changes_proposed(len(recommendation["changes"]), avg_confidence)
                    
                    # Display AI Analysis
                    st.success("Analysis Complete!")
                    
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

    # Footer with session summary
    st.markdown("---")
    
    # Display session metrics summary
    metrics = st.session_state.session_metrics.get_session_summary()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Actions Analyzed", metrics['actions_analyzed'])
    with col2:
        st.metric("Changes Applied", metrics['changes_applied'])
    with col3:
        st.metric("Session Duration", f"{metrics['duration_seconds']:.0f}s")
    with col4:
        st.metric("Engagement Score", f"{metrics['engagement_score']}/10")
    
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>SIGMA Agentic AI Actions Co-pilot</strong> | 
        Seedstars Senior AI Engineer Assignment | 
        Enhanced: Prompts, visuals, preview, error handling, comprehensive logging
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()