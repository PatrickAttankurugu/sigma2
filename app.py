"""
SIGMA Agentic AI Actions Co-pilot - Minimal Prototype
Demonstrates: Action → AI Analysis → BMC Updates → Next Steps

Seedstars Senior AI Engineer Assignment - Option 2
"""

import streamlit as st
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple Business Model Canvas Class
class BusinessModelCanvas:
    """Simple BMC with 9 sections - no complex validation"""
    
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

    def get_section(self, section_name: str) -> List[str]:
        """Get items from a BMC section"""
        return getattr(self, section_name, [])

    def update_section(self, section_name: str, items: List[str]):
        """Update a BMC section with new items"""
        setattr(self, section_name, items)

    def get_all_sections(self) -> Dict[str, List[str]]:
        """Get all BMC sections as dictionary"""
        return {
            'customer_segments': self.customer_segments,
            'value_propositions': self.value_propositions,
            'channels': self.channels,
            'customer_relationships': self.customer_relationships,
            'revenue_streams': self.revenue_streams,
            'key_resources': self.key_resources,
            'key_activities': self.key_activities,
            'key_partnerships': self.key_partnerships,
            'cost_structure': self.cost_structure
        }

# Simple AI Engine
class SimpleAI:
    """Minimal AI engine using single Gemini call"""
    
    def __init__(self, api_key: str):
        """Initialize with Google Gemini"""
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
            
        except ImportError as e:
            st.error(f"Missing dependencies: {e}")
            st.error("Run: pip install langchain langchain-google-genai")
            st.stop()

    def analyze_action(self, action_data: Dict[str, Any], bmc: BusinessModelCanvas) -> Dict[str, Any]:
        """Analyze completed action and suggest BMC updates"""
        
        system_prompt = """You are SIGMA's AI co-pilot helping founders validate business assumptions through experiments.

Analyze the completed action and suggest specific Business Model Canvas updates based on what was learned.

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
- Limit to 3-4 most important changes maximum"""

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
            # Call Gemini API
            messages = [
                self.SystemMessage(content=system_prompt),
                self.HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
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
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "analysis": f"AI response parsing failed: {str(e)}",
                "changes": [],
                "next_experiments": ["Try the analysis again with clearer action description"]
            }
        except Exception as e:
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "changes": [],
                "next_experiments": ["Check API connectivity and try again"]
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
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 SIGMA Agentic AI Actions Co-pilot")
    st.markdown("**Seedstars Assignment**: Complete experiments → AI updates your business model → Get next steps")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        st.error("⚠️ Set your GOOGLE_API_KEY in the .env file")
        st.code("""
# Create .env file with:
GOOGLE_API_KEY=your_actual_google_api_key_here
        """)
        st.stop()

    # Initialize session state
    if 'bmc' not in st.session_state:
        st.session_state.bmc = BusinessModelCanvas()
    if 'ai' not in st.session_state:
        try:
            st.session_state.ai = SimpleAI(api_key)
        except Exception as e:
            st.error(f"Failed to initialize AI: {e}")
            st.stop()
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False
    if 'last_recommendation' not in st.session_state:
        st.session_state.last_recommendation = None

    # Auto-mode toggle
    st.session_state.auto_mode = st.toggle(
        "🚀 **Auto-mode**: Apply high-confidence changes (>80%) automatically", 
        value=st.session_state.auto_mode,
        help="When enabled, changes with >80% confidence will be applied automatically"
    )

    # Main layout
    col1, col2 = st.columns([1.5, 1.2])
    
    # Left Column: Business Model Canvas Display
    with col1:
        st.subheader("📊 Current Business Model Canvas")
        st.caption("Real-time view of your business model (SEMA AI Surveillance startup)")
        
        # BMC sections with icons
        bmc_sections = [
            ("key_partnerships", "🤝", "Key Partnerships"),
            ("key_activities", "⚙️", "Key Activities"), 
            ("key_resources", "💎", "Key Resources"),
            ("value_propositions", "💡", "Value Propositions"),
            ("customer_relationships", "🤝", "Customer Relationships"),
            ("channels", "📡", "Channels"),
            ("customer_segments", "👥", "Customer Segments"),
            ("cost_structure", "💸", "Cost Structure"),
            ("revenue_streams", "💰", "Revenue Streams")
        ]
        
        # Display BMC in 3x3 grid layout
        for i in range(0, 9, 3):
            cols = st.columns(3)
            
            for j, (section_key, icon, section_title) in enumerate(bmc_sections[i:i+3]):
                with cols[j]:
                    st.markdown(f"**{icon} {section_title}**")
                    
                    items = st.session_state.bmc.get_section(section_key)
                    if items:
                        for item in items:
                            st.markdown(f"• {item}")
                    else:
                        st.markdown("*No items defined*")
                    
                    st.markdown("")  # Add spacing

    # Right Column: Action Input & Analysis
    with col2:
        st.subheader("🎯 Log Completed Action")
        
        # Sample action selector
        sample_actions = get_sample_actions()
        use_sample = st.selectbox(
            "Choose sample action or create custom:",
            ["Custom Action"] + list(sample_actions.keys())
        )
        
        if use_sample != "Custom Action":
            # Use selected sample action
            action_data = sample_actions[use_sample]
            
            st.info(f"**Sample Action Selected:** {action_data['title']}")
            
            with st.expander("📋 View Action Details", expanded=False):
                st.write(f"**Outcome:** {action_data['outcome']}")
                st.write(f"**Description:** {action_data['description']}")
                st.write("**Results:**")
                st.code(action_data['results'])
        
        else:
            # Custom action form
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
                
                form_submitted = st.form_submit_button("📝 Use Custom Action")
                
                if form_submitted and not all([action_data["title"], action_data["description"], action_data["results"]]):
                    st.error("Please fill in all fields for custom action")
                    st.stop()
        
        # Analyze Action Button
        if st.button("🚀 Analyze Action & Update BMC", use_container_width=True, type="primary"):
            if not action_data.get("title") or not action_data.get("results"):
                st.error("Action title and results are required")
            else:
                with st.spinner("🧠 AI analyzing your action..."):
                    # Get AI recommendation
                    recommendation = st.session_state.ai.analyze_action(action_data, st.session_state.bmc)
                    st.session_state.last_recommendation = recommendation
                    
                    # Display AI Analysis
                    st.success("✅ Analysis Complete!")
                    
                    with st.expander("🧠 AI Analysis", expanded=True):
                        st.write(recommendation["analysis"])
                    
                    # Show proposed changes
                    if recommendation["changes"]:
                        st.subheader("📝 Proposed BMC Updates")
                        
                        high_confidence_changes = []
                        changes_applied = 0
                        
                        for i, change in enumerate(recommendation["changes"]):
                            confidence = change.get("confidence", 0)
                            
                            # Confidence indicator
                            if confidence >= 0.9:
                                confidence_color = "🟢"
                                confidence_label = "Very High"
                            elif confidence >= 0.8:
                                confidence_color = "🟢"
                                confidence_label = "High"
                                high_confidence_changes.append(change)
                            elif confidence >= 0.7:
                                confidence_color = "🟡"
                                confidence_label = "Medium"
                            else:
                                confidence_color = "🔴"
                                confidence_label = "Low"
                            
                            # Display change
                            section_display = change['section'].replace('_', ' ').title()
                            
                            st.markdown(f"""
                            **{confidence_color} {section_display}** - {change['type'].title()} ({confidence_label} confidence: {confidence:.0%})
                            
                            **Change:** {change['new']}
                            
                            **Reasoning:** {change['reason']}
                            """)
                            
                            st.markdown("---")
                        
                        # Auto-mode application
                        if st.session_state.auto_mode and high_confidence_changes:
                            st.info(f"🚀 **Auto-mode Active**: Applying {len(high_confidence_changes)} high-confidence changes...")
                            
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
                                st.success(f"✅ Auto-applied {changes_applied} high-confidence changes!")
                                st.rerun()
                        
                        # Manual controls (if auto-mode is off or there are non-auto changes)
                        elif not st.session_state.auto_mode:
                            col_apply, col_reject = st.columns(2)
                            
                            with col_apply:
                                if st.button("✅ Apply All Changes", use_container_width=True):
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
                                    
                                    st.success(f"✅ Applied {changes_applied} changes to your business model!")
                                    st.rerun()
                            
                            with col_reject:
                                if st.button("❌ Reject Changes", use_container_width=True):
                                    st.info("Changes rejected. BMC remains unchanged.")
                    
                    else:
                        st.info("No BMC changes suggested based on this action.")
                    
                    # Show next experiments
                    if recommendation.get("next_experiments"):
                        st.subheader("🔬 Suggested Next Experiments")
                        for i, experiment in enumerate(recommendation["next_experiments"], 1):
                            st.write(f"**{i}.** {experiment}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>SIGMA Agentic AI Actions Co-pilot</strong> | 
        Seedstars Senior AI Engineer Assignment | 
        Demonstrates: Action Analysis → BMC Updates → Next Steps
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()