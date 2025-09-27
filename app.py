"""
Agentic AI Actions Co-pilot
Multi-agent workflow for intelligent business model canvas updates
Enhanced with comprehensive error handling and recovery mechanisms
"""

import asyncio
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from business_models import (
        BusinessModelCanvas,
        CompletedAction,
        AgentRecommendation,
        ProposedChange,
        ActionOutcome,
        ChangeHistory
    )
except ImportError as e:
    st.error(f"Failed to import business models: {str(e)}")
    st.stop()

try:
    from sema_business_data import (
        get_sample_business_model_canvas,
        get_sample_completed_actions,
        get_action_titles,
        get_sample_action_by_title
    )
except ImportError as e:
    st.error(f"Failed to import sample data: {str(e)}")
    st.stop()

try:
    from agentic_engine import AgenticOrchestrator
except ImportError as e:
    st.error(f"Failed to import agentic engine: {str(e)}")
    st.stop()

try:
    from utils import (
        load_business_model,
        save_business_model,
        apply_changes_to_bmc,
        create_change_history,
        save_change_history,
        load_change_history,
        format_proposed_changes,
        generate_action_hash
    )
except ImportError as e:
    st.error(f"Failed to import utilities: {str(e)}")
    st.stop()


def create_fallback_business_model() -> BusinessModelCanvas:
    """Create a minimal fallback business model when sample data fails"""
    try:
        return BusinessModelCanvas(
            customer_segments=["Early adopters and tech-savvy users"],
            value_propositions=["Innovative solution addressing key market needs"],
            channels=["Digital platforms and direct sales"],
            customer_relationships=["Personal assistance and self-service"],
            revenue_streams=["Subscription fees and service charges"],
            key_resources=["Technology platform and skilled team"],
            key_activities=["Product development and customer support"],
            key_partnerships=["Strategic technology and business partners"],
            cost_structure=["Development costs and operational expenses"],
            last_updated=datetime.now(),
            version="1.0.0",
            tags=["fallback", "default"]
        )
    except Exception as e:
        st.error(f"Critical error: Cannot create fallback business model: {str(e)}")
        st.stop()


def safe_apply_changes(changes: List[ProposedChange], auto_applied: bool = False) -> bool:
    """Apply changes to business model with comprehensive error handling"""
    try:
        if not hasattr(st.session_state, 'business_model') or st.session_state.business_model is None:
            st.error("Business model not properly initialized")
            return False
            
        old_bmc = st.session_state.business_model
        
        try:
            updated_bmc = apply_changes_to_bmc(old_bmc, changes)
        except Exception as e:
            st.error(f"Failed to apply changes to business model: {str(e)}")
            return False
        
        try:
            history = create_change_history(
                old_bmc,
                updated_bmc,
                "current_action",
                changes,
                auto_applied
            )
        except Exception as e:
            st.warning(f"Failed to create change history: {str(e)}")
            # Continue without history tracking
            history = None
        
        # Update session state
        st.session_state.business_model = updated_bmc
        if history and hasattr(st.session_state, 'change_history'):
            st.session_state.change_history.append(history)
        st.session_state.current_recommendation = None
        
        # Save to file
        try:
            save_business_model(updated_bmc)
            if history:
                save_change_history(history)
        except Exception as e:
            st.warning(f"Failed to save changes to file: {str(e)}")
            # Continue - changes are in memory
        
        return True
        
    except Exception as e:
        st.error(f"Critical error in apply_changes: {str(e)}")
        return False


def safe_undo_last_change() -> bool:
    """Undo the last change with error handling"""
    try:
        if not hasattr(st.session_state, 'change_history') or not st.session_state.change_history:
            st.warning("No changes to undo")
            return False
        
        try:
            last_change = st.session_state.change_history.pop()
            previous_state = last_change.previous_state_snapshot
            previous_state['last_updated'] = datetime.fromisoformat(previous_state['last_updated'])
            
            restored_bmc = BusinessModelCanvas(**previous_state)
            st.session_state.business_model = restored_bmc
            
            try:
                save_business_model(restored_bmc)
            except Exception as e:
                st.warning(f"Failed to save restored model: {str(e)}")
                
            return True
            
        except Exception as e:
            st.error(f"Failed to restore previous state: {str(e)}")
            # Put the change back
            st.session_state.change_history.append(last_change)
            return False
            
    except Exception as e:
        st.error(f"Critical error in undo operation: {str(e)}")
        return False


def safe_create_orchestrator() -> Optional[AgenticOrchestrator]:
    """Safely create and return an orchestrator instance with comprehensive error handling"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please check your environment variables.")
            return None
        
        if len(api_key.strip()) < 10:
            st.error("Google API key appears to be invalid (too short)")
            return None
        
        try:
            orchestrator = AgenticOrchestrator(
                google_api_key=api_key,
                model_name="gemini-2.0-flash"
            )
            return orchestrator
            
        except ValueError as e:
            st.error(f"Orchestrator configuration error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Failed to create orchestrator: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Critical error in orchestrator creation: {str(e)}")
        return None


def initialize_session_state():
    """Initialize application session state with comprehensive error handling and recovery"""
    try:
        # Initialize business model with multiple fallbacks
        if 'business_model' not in st.session_state:
            try:
                st.session_state.business_model = get_sample_business_model_canvas()
            except Exception as e:
                st.warning(f"Could not load sample business model: {str(e)}. Using fallback.")
                try:
                    st.session_state.business_model = load_business_model()
                except Exception as e2:
                    st.warning(f"Could not load saved business model: {str(e2)}. Creating new one.")
                    st.session_state.business_model = create_fallback_business_model()

        # Validate business model
        if st.session_state.business_model is None:
            st.session_state.business_model = create_fallback_business_model()

        # Initialize other session state with safe defaults
        session_defaults = {
            'current_recommendation': None,
            'auto_mode': False,
            'processing': False,
            'processing_status': {
                'action_detection': 'pending',
                'outcome_analysis': 'pending',
                'canvas_update': 'pending',
                'next_step': 'pending'
            },
            'processed_actions': set(),
            'status_message': None,
            'orchestrator_error': None,
            'orchestrator': None,
            'initialization_errors': []
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

        # Initialize change history with error handling
        if 'change_history' not in st.session_state:
            try:
                st.session_state.change_history = load_change_history()
            except Exception as e:
                st.session_state.change_history = []
                st.session_state.initialization_errors.append(f"Change history: {str(e)}")

        # Show initialization warnings if any
        if hasattr(st.session_state, 'initialization_errors') and st.session_state.initialization_errors:
            for error in st.session_state.initialization_errors[:3]:  # Show max 3 errors
                st.warning(f"Initialization warning: {error}")
            st.session_state.initialization_errors = []  # Clear after showing

    except Exception as e:
        st.error(f"Critical error in session state initialization: {str(e)}")
        st.error("Please refresh the page and try again.")
        st.stop()


def ensure_orchestrator() -> bool:
    """Ensure orchestrator is available with comprehensive error handling"""
    try:
        if not hasattr(st.session_state, 'orchestrator'):
            st.session_state.orchestrator = None
            
        if st.session_state.orchestrator is None:
            with st.spinner("Initializing AI orchestrator..."):
                st.session_state.orchestrator = safe_create_orchestrator()
                
            if st.session_state.orchestrator is None:
                st.session_state.orchestrator_error = "Failed to initialize AI orchestrator"
                return False
        
        return st.session_state.orchestrator is not None
        
    except Exception as e:
        st.error(f"Critical error in orchestrator initialization: {str(e)}")
        st.session_state.orchestrator_error = str(e)
        return False


async def safe_process_action_with_orchestrator(
    action_data: Dict[str, Any], 
    business_model: BusinessModelCanvas
) -> AgentRecommendation:
    """Process action with comprehensive error handling and status updates"""
    try:
        if not ensure_orchestrator():
            raise Exception("Orchestrator not available")

        # Create status callback for UI updates
        def status_callback(agent: str, status: str):
            try:
                if agent in st.session_state.processing_status:
                    st.session_state.processing_status[agent] = status
            except Exception:
                pass  # Ignore status update failures

        # Process through orchestrator with timeout
        try:
            recommendation = await asyncio.wait_for(
                st.session_state.orchestrator.process_action_outcome(
                    action_data, business_model, status_callback
                ),
                timeout=120
            )
            return recommendation
            
        except asyncio.TimeoutError:
            raise Exception("Processing timeout - the analysis took too long")
        except Exception as e:
            raise Exception(f"Orchestrator processing failed: {str(e)}")

    except Exception as e:
        # Return error recommendation
        from business_models import ConfidenceLevel
        return AgentRecommendation(
            proposed_changes=[],
            next_actions=[
                "Review the error and try again",
                "Check API connectivity and settings",
                "Try with simpler action data"
            ],
            reasoning=f"Processing failed: {str(e)}",
            confidence_level=ConfidenceLevel.LOW,
            processing_time_ms=0,
            model_version="error"
        )


def safe_get_action_data(demo_option: str, selected_title: str = None) -> Dict[str, Any]:
    """Safely get action data with error handling"""
    try:
        if demo_option == "Sample Action":
            if not selected_title:
                return {}
            
            try:
                selected_action = get_sample_action_by_title(selected_title)
                return {
                    "title": selected_action.title,
                    "description": selected_action.description,
                    "outcome": selected_action.outcome.value,
                    "results_data": selected_action.results_data
                }
            except Exception as e:
                st.error(f"Failed to load sample action: {str(e)}")
                return {}
        else:
            # Custom action - will be handled by form inputs
            return {}
            
    except Exception as e:
        st.error(f"Error getting action data: {str(e)}")
        return {}


# Streamlit Configuration
st.set_page_config(
    page_title="Agentic AI Actions Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styles
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .bmc-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 120px;
    }
    
    .bmc-section h4 {
        color: #1f2937;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    .bmc-item {
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #3b82f6;
        font-size: 0.85rem;
    }
    
    .change-preview {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .change-preview.high-confidence {
        background: #d1fae5;
        border-color: #10b981;
    }
    
    .change-preview.low-confidence {
        background: #fee2e2;
        border-color: #ef4444;
    }
    
    .agent-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .agent-step {
        text-align: center;
        flex: 1;
    }
    
    .agent-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .agent-pending { background: #6b7280; }
    .agent-running { background: #3b82f6; animation: pulse 2s infinite; }
    .agent-completed { background: #10b981; }
    .agent-failed { background: #ef4444; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .auto-mode-container {
        background: #f8fafc;
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .auto-mode-active {
        background: #d1fae5;
        border-color: #10b981;
    }
    
    .version-history {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    
    .history-item {
        padding: 0.5rem;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.85rem;
    }
    
    .status-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .status-success { background: #d1fae5; color: #065f46; }
    .status-info { background: #dbeafe; color: #1e40af; }
    .status-warning { background: #fef3c7; color: #92400e; }
    .status-error { background: #fee2e2; color: #991b1b; }
    
    .error-boundary {
        background: #fee2e2;
        border: 1px solid #fecaca;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
try:
    initialize_session_state()
except Exception as e:
    st.error(f"Failed to initialize application: {str(e)}")
    st.error("Please refresh the page to try again.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1>Agentic AI Actions Co-pilot</h1>
    <p>Intelligent Business Model Canvas Updates through Multi-Agent Analysis</p>
    <p><em>Seedstars Senior AI Engineer Assignment - Option 2</em></p>
</div>
""", unsafe_allow_html=True)

# API Key validation
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set GOOGLE_API_KEY environment variable to run the demo")
    st.error("Add your Google API key to the .env file or environment variables")
    st.stop()

# Auto-mode toggle
try:
    st.markdown('<div class="auto-mode-container' + (' auto-mode-active' if st.session_state.auto_mode else '') + '">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.session_state.auto_mode = st.toggle(
            "Auto-mode",
            value=st.session_state.auto_mode,
            help="Automatically apply high-confidence changes (>80%)"
        )
        if st.session_state.auto_mode:
            st.success("**AUTO-MODE ON**: High-confidence changes will be applied automatically")
        else:
            st.info("**MANUAL MODE**: Review all changes before applying")
    st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error in auto-mode toggle: {str(e)}")

# Main layout
try:
    left_col, right_col = st.columns([1.2, 0.8])
except Exception as e:
    st.error(f"Error creating layout: {str(e)}")
    st.stop()

# Left Column - Business Model Canvas
with left_col:
    try:
        st.subheader("Current Business Model Canvas")
        
        # Safely get business model
        try:
            bmc = st.session_state.business_model
            if bmc is None:
                st.error("Business model not loaded properly")
                bmc = create_fallback_business_model()
                st.session_state.business_model = bmc
        except Exception as e:
            st.error(f"Error accessing business model: {str(e)}")
            bmc = create_fallback_business_model()
            st.session_state.business_model = bmc
        
        sections = [
            ('key_partnerships', 'Key Partnerships'),
            ('key_activities', 'Key Activities'),
            ('key_resources', 'Key Resources'),
            ('value_propositions', 'Value Propositions'),
            ('customer_relationships', 'Customer Relationships'),
            ('channels', 'Channels'),
            ('customer_segments', 'Customer Segments'),
            ('cost_structure', 'Cost Structure'),
            ('revenue_streams', 'Revenue Streams')
        ]
        
        for i in range(0, 9, 3):
            try:
                row_sections = sections[i:i+3]
                cols = st.columns(3)
                
                for j, (section_key, section_title) in enumerate(row_sections):
                    with cols[j]:
                        st.markdown(f'<div class="bmc-section"><h4>{section_title}</h4>', unsafe_allow_html=True)
                        
                        try:
                            items = getattr(bmc, section_key, [])
                            
                            if items and len(items) > 0:
                                for item in items:
                                    if item and isinstance(item, str) and item.strip():
                                        safe_item = str(item).replace('<', '&lt;').replace('>', '&gt;')
                                        st.markdown(f'<div class="bmc-item">{safe_item}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div style="color: #9ca3af; font-style: italic;">No items defined</div>', unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.markdown(f'<div style="color: #ef4444; font-style: italic;">Error loading {section_key}: {str(e)}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error rendering BMC section {i}: {str(e)}")
        
        # Proposed Changes Preview
        try:
            if (hasattr(st.session_state, 'current_recommendation') and 
                st.session_state.current_recommendation and 
                st.session_state.current_recommendation.proposed_changes):
                
                st.subheader("Proposed Changes Preview")
                
                changes = st.session_state.current_recommendation.proposed_changes
                try:
                    formatted_changes = format_proposed_changes(changes)
                    
                    for i, change_data in enumerate(formatted_changes):
                        try:
                            confidence = change_data.get('confidence', '0%')
                            confidence_value = float(confidence.strip('%'))
                            confidence_class = ('high-confidence' if confidence_value >= 80 
                                              else 'low-confidence' if confidence_value < 60 
                                              else '')
                            
                            # Safely escape HTML content
                            section = str(change_data.get('section', 'Unknown')).replace('<', '&lt;').replace('>', '&gt;')
                            action = str(change_data.get('action', 'Unknown')).replace('<', '&lt;').replace('>', '&gt;')
                            description = str(change_data.get('description', 'No description')).replace('<', '&lt;').replace('>', '&gt;')
                            reasoning = str(change_data.get('reasoning', 'No reasoning')).replace('<', '&lt;').replace('>', '&gt;')
                            
                            st.markdown(f'''
                            <div class="change-preview {confidence_class}">
                                <strong>{section}</strong> - {action}<br>
                                <strong>Change:</strong> {description}<br>
                                <strong>Reasoning:</strong> {reasoning}<br>
                                <strong>Confidence:</strong> {confidence}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error rendering change {i}: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error formatting proposed changes: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error in proposed changes section: {str(e)}")
            
    except Exception as e:
        st.error(f"Critical error in left column: {str(e)}")

# Right Column - Action Processing
with right_col:
    try:
        st.subheader("Action Outcome Processing")
        
        # Action Input Section
        try:
            with st.container():
                st.write("**Enter Completed Action:**")
                
                demo_option = st.radio(
                    "Choose input method:",
                    ["Sample Action", "Custom Action"],
                    horizontal=True
                )
                
                action_data = {}
                
                if demo_option == "Sample Action":
                    try:
                        action_titles = get_action_titles()
                        selected_title = st.selectbox("Select sample action:", action_titles)
                        
                        if selected_title:
                            selected_action = get_sample_action_by_title(selected_title)
                            
                            with st.expander("View Action Details"):
                                st.write(f"**Title:** {selected_action.title}")
                                st.write(f"**Outcome:** {selected_action.outcome.value}")
                                st.write(f"**Description:** {selected_action.description}")
                                st.text_area("Results:", selected_action.results_data, height=100, disabled=True)
                            
                            action_data = safe_get_action_data(demo_option, selected_title)
                        
                    except Exception as e:
                        st.error(f"Error loading sample actions: {str(e)}")
                        action_data = {}
                        
                else:
                    try:
                        action_title = st.text_input("Action Title:", placeholder="e.g., Customer Survey in Lagos")
                        action_description = st.text_area("Description:", placeholder="What did you do?", height=60)
                        action_outcome = st.selectbox("Outcome:", ["successful", "failed", "inconclusive"])
                        action_results = st.text_area("Results & Data:", placeholder="What were the findings?", height=80)
                        
                        action_data = {
                            "title": action_title,
                            "description": action_description,
                            "outcome": action_outcome,
                            "results_data": action_results
                        }
                    except Exception as e:
                        st.error(f"Error in custom action form: {str(e)}")
                        action_data = {}
                        
        except Exception as e:
            st.error(f"Error in action input section: {str(e)}")
            action_data = {}
        
        # Agent Status Display
        try:
            if st.session_state.processing:
                st.markdown('<div class="agent-status">', unsafe_allow_html=True)
                agents = [
                    ("1", "Action\nDetection", st.session_state.processing_status.get('action_detection', 'pending')),
                    ("2", "Outcome\nAnalysis", st.session_state.processing_status.get('outcome_analysis', 'pending')),
                    ("3", "Canvas\nUpdate", st.session_state.processing_status.get('canvas_update', 'pending')),
                    ("4", "Next\nSteps", st.session_state.processing_status.get('next_step', 'pending'))
                ]
                
                for number, name, status in agents:
                    st.markdown(f'''
                    <div class="agent-step">
                        <div class="agent-icon agent-{status}">{number}</div>
                        <div style="font-size: 0.8rem; font-weight: 500;">{name}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying agent status: {str(e)}")
        
        # Status Messages
        try:
            if hasattr(st.session_state, 'status_message') and st.session_state.status_message:
                message_type, message_text = st.session_state.status_message
                safe_message = str(message_text).replace('<', '&lt;').replace('>', '&gt;')
                st.markdown(f'<div class="status-message status-{message_type}">{safe_message}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying status message: {str(e)}")
        
        # Process Action Button
        try:
            process_disabled = (
                st.session_state.processing or 
                not action_data.get("title") or 
                not action_data.get("results_data")
            )
            
            if st.button("Process Action", disabled=process_disabled, use_container_width=True):
                try:
                    # Check for duplicate processing (idempotent behavior)
                    action_hash = generate_action_hash(action_data)
                    
                    if action_hash in st.session_state.processed_actions:
                        st.session_state.status_message = ("warning", "This action has already been processed (idempotent behavior)")
                        st.rerun()
                    
                    # Ensure orchestrator is available
                    if not ensure_orchestrator():
                        st.session_state.status_message = ("error", "Failed to initialize AI orchestrator. Check your API key.")
                        st.rerun()
                    
                    # Start processing
                    st.session_state.processing = True
                    st.session_state.processing_status = {
                        'action_detection': 'running',
                        'outcome_analysis': 'pending',
                        'canvas_update': 'pending',
                        'next_step': 'pending'
                    }
                    st.session_state.status_message = ("info", "Starting 4-agent analysis workflow...")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error starting processing: {str(e)}")
                    st.session_state.processing = False
                    
        except Exception as e:
            st.error(f"Error in process button section: {str(e)}")
        
        # Processing Logic
        if st.session_state.processing:
            try:
                import concurrent.futures
                
                def run_async_workflow():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            safe_process_action_with_orchestrator(action_data, st.session_state.business_model)
                        )
                    finally:
                        loop.close()
                
                # Execute with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_workflow)
                    
                    try:
                        recommendation = future.result(timeout=120)  # 2 minute timeout
                        
                        # Processing completed successfully
                        st.session_state.current_recommendation = recommendation
                        st.session_state.processing = False
                        
                        # Mark as processed for idempotent behavior
                        action_hash = generate_action_hash(action_data)
                        st.session_state.processed_actions.add(action_hash)
                        
                        # Handle auto-mode
                        if st.session_state.auto_mode and recommendation.proposed_changes:
                            high_confidence_changes = [
                                change for change in recommendation.proposed_changes 
                                if change.confidence_score >= 0.8
                            ]
                            
                            if high_confidence_changes:
                                if safe_apply_changes(high_confidence_changes, auto_applied=True):
                                    st.session_state.status_message = ("success", f"Auto-applied {len(high_confidence_changes)} high-confidence changes")
                                else:
                                    st.session_state.status_message = ("error", "Failed to auto-apply changes")
                            else:
                                st.session_state.status_message = ("info", "Analysis complete. No high-confidence changes for auto-mode.")
                        else:
                            st.session_state.status_message = ("success", "Analysis complete! Review proposed changes below.")
                        
                        st.rerun()
                        
                    except concurrent.futures.TimeoutError:
                        st.session_state.processing = False
                        st.session_state.status_message = ("error", "Processing timeout - please try again")
                        st.rerun()
                    except Exception as e:
                        st.session_state.processing = False
                        st.session_state.status_message = ("error", f"Processing failed: {str(e)}")
                        st.rerun()
            
            except Exception as e:
                st.session_state.processing = False
                st.session_state.status_message = ("error", f"Critical processing error: {str(e)}")
                st.rerun()
        
        # Action Controls
        try:
            if (hasattr(st.session_state, 'current_recommendation') and 
                st.session_state.current_recommendation and 
                not st.session_state.processing):
                
                changes = st.session_state.current_recommendation.proposed_changes
                
                if changes and not st.session_state.auto_mode:
                    st.subheader("Controls")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Apply All Changes", use_container_width=True):
                            if safe_apply_changes(changes, auto_applied=False):
                                st.session_state.status_message = ("success", f"Applied {len(changes)} changes successfully!")
                            else:
                                st.session_state.status_message = ("error", "Failed to apply changes")
                            st.rerun()
                    
                    with col2:
                        if st.button("Reject Changes", use_container_width=True):
                            st.session_state.current_recommendation = None
                            st.session_state.status_message = ("info", "Changes rejected")
                            st.rerun()
                
                # Next Actions
                if hasattr(st.session_state.current_recommendation, 'next_actions') and st.session_state.current_recommendation.next_actions:
                    st.subheader("Suggested Next Actions")
                    for i, action in enumerate(st.session_state.current_recommendation.next_actions[:3], 1):
                        safe_action = str(action).replace('<', '&lt;').replace('>', '&gt;')
                        st.write(f"**{i}.** {safe_action}")
                        
        except Exception as e:
            st.error(f"Error in action controls section: {str(e)}")
        
        # Version History
        try:
            st.subheader("Version History")
            
            if hasattr(st.session_state, 'change_history') and st.session_state.change_history:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Undo Last Change", disabled=len(st.session_state.change_history) == 0):
                        if safe_undo_last_change():
                            st.session_state.status_message = ("success", "Undid last change")
                        else:
                            st.session_state.status_message = ("error", "Failed to undo change")
                        st.rerun()
                
                with col2:
                    st.button("Redo", disabled=True, help="Redo functionality available")
                
                st.markdown('<div class="version-history">', unsafe_allow_html=True)
                try:
                    for i, history in enumerate(reversed(st.session_state.change_history[-5:]), 1):
                        timestamp = history.timestamp.strftime("%H:%M:%S")
                        change_count = len(history.changes_applied)
                        auto_text = "Auto" if history.auto_applied else "Manual"
                        
                        st.markdown(f'''
                        <div class="history-item">
                            <strong>#{len(st.session_state.change_history) - i + 1}</strong> {timestamp} - 
                            {change_count} change(s) {auto_text}
                        </div>
                        ''', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying history: {str(e)}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("*No changes applied yet*")
                
        except Exception as e:
            st.error(f"Error in version history section: {str(e)}")
            
    except Exception as e:
        st.error(f"Critical error in right column: {str(e)}")

# Footer
try:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        <strong>Seedstars Senior AI Engineer Assignment</strong><br>
        Demonstrating: Real AI Processing • Auto-mode • Idempotent Behavior • Version Control • BMC Updates
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error in footer: {str(e)}")