"""
Enhanced Streamlit interface for the Agentic AI Actions Co-pilot system.
SIGMA-inspired design with chat integration and professional UI.

This application demonstrates a sophisticated multi-agent workflow for
automatically updating business model canvases based on completed action outcomes.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import time

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from business_models import (
    BusinessModelCanvas,
    CompletedAction,
    AgentRecommendation,
    ProposedChange,
    ProcessingStatus,
    AgentStatus,
    ActionOutcome
)
from mock_data import (
    get_sample_business_model_canvas,
    get_sample_completed_actions,
    get_action_titles,
    get_sample_action_by_title
)
from agentic_engine import AgenticOrchestrator, validate_safety
from utils import (
    load_business_model,
    save_business_model,
    format_proposed_changes,
    create_before_after_comparison,
    generate_change_summary,
    apply_changes_to_bmc,
    create_change_history,
    save_change_history,
    export_business_model_to_csv,
    export_changes_to_json
)


# SIGMA-Inspired Styling
def load_sigma_css():
    """Load SIGMA-inspired CSS styling"""
    st.markdown("""
    <style>
    /* SIGMA Color Palette */
    :root {
        --sigma-blue: #1E40AF;
        --sigma-light-blue: #3B82F6;
        --sigma-dark-blue: #1E3A8A;
        --sigma-bg-light: #F8FAFC;
        --sigma-card-bg: #FFFFFF;
        --sigma-text-primary: #1F2937;
        --sigma-text-secondary: #6B7280;
        --sigma-border: #E5E7EB;
        --sigma-success: #10B981;
        --sigma-warning: #F59E0B;
        --sigma-danger: #EF4444;
    }
    
    /* Global Styles */
    .stApp {
        background-color: var(--sigma-bg-light);
    }
    
    /* Header Styling */
    .sigma-header {
        background: linear-gradient(135deg, var(--sigma-dark-blue) 0%, var(--sigma-light-blue) 100%);
        color: white;
        padding: 2rem 1.5rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 1rem 1rem;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.15);
    }
    
    .sigma-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sigma-header .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sigma-header .features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .feature-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
    }
    
    /* Card Styling */
    .sigma-card {
        background: var(--sigma-card-bg);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--sigma-border);
        transition: all 0.3s ease;
    }
    
    .sigma-card:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .sigma-card h3 {
        color: var(--sigma-dark-blue);
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.25rem;
    }
    
    /* Agent Status Styling */
    .agent-pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--sigma-card-bg);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    .agent-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }
    
    .agent-step::after {
        content: '';
        position: absolute;
        top: 25px;
        right: -50%;
        width: 100%;
        height: 2px;
        background: var(--sigma-border);
        z-index: 1;
    }
    
    .agent-step:last-child::after {
        display: none;
    }
    
    .agent-icon {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
        z-index: 2;
    }
    
    .agent-completed { background: var(--sigma-success); }
    .agent-running { background: var(--sigma-light-blue); animation: pulse 2s infinite; }
    .agent-pending { background: var(--sigma-text-secondary); }
    .agent-failed { background: var(--sigma-danger); }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .agent-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--sigma-text-primary);
        text-align: center;
    }
    
    /* Chat Interface */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: var(--sigma-card-bg);
        border-radius: 1rem;
        border: 1px solid var(--sigma-border);
    }
    
    .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 1rem;
        max-width: 80%;
    }
    
    .chat-message.user {
        background: var(--sigma-light-blue);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
    }
    
    .chat-message.assistant {
        background: var(--sigma-bg-light);
        border: 1px solid var(--sigma-border);
        border-bottom-left-radius: 0.25rem;
    }
    
    .chat-message.system {
        background: var(--sigma-success);
        color: white;
        text-align: center;
        margin: 0 auto;
        max-width: 60%;
        font-size: 0.9rem;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        color: var(--sigma-text-secondary);
        font-style: italic;
    }
    
    .typing-dots {
        display: inline-flex;
        margin-left: 0.5rem;
    }
    
    .typing-dots span {
        height: 4px;
        width: 4px;
        background: var(--sigma-text-secondary);
        border-radius: 50%;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-10px); opacity: 1; }
    }
    
    /* Confidence Indicators */
    .confidence-high { 
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        border-left: 4px solid var(--sigma-success);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .confidence-medium { 
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        border-left: 4px solid var(--sigma-warning);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .confidence-low { 
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-left: 4px solid var(--sigma-danger);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Auto-mode Toggle */
    .auto-mode-container {
        background: var(--sigma-card-bg);
        border: 2px solid var(--sigma-light-blue);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .auto-mode-active {
        background: linear-gradient(135deg, #DBEAFE, #BFDBFE);
        border-color: var(--sigma-success);
    }
    
    /* Business Model Canvas Grid */
    .bmc-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .bmc-section {
        background: var(--sigma-card-bg);
        border: 1px solid var(--sigma-border);
        border-radius: 0.75rem;
        padding: 1rem;
        min-height: 150px;
    }
    
    .bmc-section h4 {
        color: var(--sigma-dark-blue);
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .bmc-item {
        background: var(--sigma-bg-light);
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        border-left: 3px solid var(--sigma-light-blue);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .sigma-header .features {
            gap: 1rem;
        }
        
        .bmc-grid {
            grid-template-columns: 1fr;
        }
        
        .agent-pipeline {
            flex-direction: column;
            gap: 1rem;
        }
        
        .agent-step::after {
            display: none;
        }
        
        .chat-message {
            max-width: 95%;
        }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--sigma-light-blue), var(--sigma-dark-blue));
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(30, 64, 175, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.3);
    }
    
    /* Metrics */
    .metric-card {
        background: var(--sigma-card-bg);
        border: 1px solid var(--sigma-border);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--sigma-dark-blue);
    }
    
    .metric-label {
        color: var(--sigma-text-secondary);
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Session State Initialization
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'business_model' not in st.session_state:
        st.session_state.business_model = load_business_model()

    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

    if 'current_recommendation' not in st.session_state:
        st.session_state.current_recommendation = None

    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False

    if 'change_history' not in st.session_state:
        st.session_state.change_history = []

    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = True


# Utility Functions
def check_api_key() -> bool:
    """Check if Google API key is configured."""
    return bool(os.getenv("GOOGLE_API_KEY"))


def display_header():
    """Display the SIGMA-inspired header."""
    st.markdown("""
    <div class="sigma-header">
        <h1>⚡ Agentic AI Actions Co-pilot</h1>
        <div class="subtitle">
            Intelligent Business Model Canvas Updates through Multi-Agent AI Analysis<br>
            <em>Specialized for Emerging Market Entrepreneurs</em>
        </div>
        <div class="features">
            <div class="feature-badge">🤖 4-Agent Workflow</div>
            <div class="feature-badge">🎯 Smart BMC Updates</div>
            <div class="feature-badge">🔄 Auto-mode Capable</div>
            <div class="feature-badge">📊 Change Tracking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_auto_mode_toggle():
    """Display the auto-mode toggle with enhanced styling."""
    container_class = "auto-mode-container auto-mode-active" if st.session_state.auto_mode else "auto-mode-container"
    
    st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.session_state.auto_mode = st.toggle(
            "🤖 Auto-mode",
            value=st.session_state.auto_mode,
            help="Automatically apply high-confidence changes without manual approval"
        )
        
        if st.session_state.auto_mode:
            st.success("Auto-mode: System will apply safe changes automatically")
        else:
            st.info("Manual mode: Review all changes before applying")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_agent_pipeline(status: Optional[ProcessingStatus] = None):
    """Display the agent processing pipeline with visual status."""
    if not status:
        # Show default pipeline
        agents = [
            ("1", "Action Detection", "pending"),
            ("2", "Outcome Analysis", "pending"), 
            ("3", "Canvas Updates", "pending"),
            ("4", "Next Steps", "pending")
        ]
    else:
        agents = [
            ("1", "Action Detection", status.action_detection_status.value),
            ("2", "Outcome Analysis", status.outcome_analysis_status.value),
            ("3", "Canvas Updates", status.canvas_update_status.value),
            ("4", "Next Steps", status.next_step_status.value)
        ]
    
    st.markdown('<div class="agent-pipeline">', unsafe_allow_html=True)
    
    for number, name, status_val in agents:
        st.markdown(f"""
        <div class="agent-step">
            <div class="agent-icon agent-{status_val}">{number}</div>
            <div class="agent-label">{name}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_chat_interface():
    """Display the chat interface for agentic interactions."""
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>💬 Horo Chat - Actions Co-pilot</h3>', unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        message_class = f"chat-message {message['role']}"
        st.markdown(f'<div class="{message_class}">{message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def add_chat_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })


def display_typing_indicator():
    """Display typing indicator for AI processing."""
    st.markdown("""
    <div class="typing-indicator">
        Horo is analyzing your action outcome
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_input_section():
    """Display the enhanced input section."""
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>📋 Action Outcome Input</h3>', unsafe_allow_html=True)
    
    # Action selection method
    input_method = st.radio(
        "Choose input method:",
        ["Select Sample Action", "Custom Action Input"],
        horizontal=True
    )

    if input_method == "Select Sample Action":
        # Sample action dropdown
        action_titles = get_action_titles()
        selected_title = st.selectbox(
            "Select a sample action:",
            action_titles,
            help="Choose from realistic emerging market scenarios"
        )

        selected_action = get_sample_action_by_title(selected_title)

        # Display action details in cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Action Title</div>
                <div style="font-weight: 600; color: var(--sigma-dark-blue); margin-top: 0.5rem;">
                    {selected_action.title}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            outcome_color = {
                "successful": "var(--sigma-success)",
                "failed": "var(--sigma-danger)", 
                "inconclusive": "var(--sigma-warning)"
            }
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Outcome</div>
                <div style="font-weight: 600; color: {outcome_color.get(selected_action.outcome.value, 'var(--sigma-text-primary)')}; margin-top: 0.5rem;">
                    {selected_action.outcome.value.title()}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📋 View Full Action Details", expanded=False):
            st.write("**Description:**")
            st.write(selected_action.description)
            
            st.write("**Results:**")
            st.text_area(
                "Results",
                selected_action.results_data,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )

        action_data = {
            "title": selected_action.title,
            "description": selected_action.description,
            "outcome": selected_action.outcome.value,
            "results_data": selected_action.results_data,
            "success_metrics": selected_action.success_metrics or {}
        }

    else:
        # Custom input with better styling
        col1, col2 = st.columns(2)
        with col1:
            custom_title = st.text_input("Action Title", placeholder="e.g., Market Validation Survey")
            custom_outcome = st.selectbox("Outcome", ["successful", "failed", "inconclusive"])

        with col2:
            custom_description = st.text_area(
                "Description",
                placeholder="Describe what was done and measured...",
                height=100
            )

        custom_results = st.text_area(
            "Results Data",
            placeholder="Detailed findings, metrics, and insights...",
            height=150
        )

        action_data = {
            "title": custom_title,
            "description": custom_description,
            "outcome": custom_outcome,
            "results_data": custom_results,
            "success_metrics": {}
        }

    st.markdown('</div>', unsafe_allow_html=True)
    return action_data


async def process_action_with_agents(action_data: Dict) -> Optional[AgentRecommendation]:
    """Process action through the multi-agent workflow with chat updates."""
    try:
        if not st.session_state.orchestrator:
            selected_model = "gemini-1.5-flash"
            st.session_state.orchestrator = AgenticOrchestrator(model_name=selected_model)

        # Add initial chat message
        add_chat_message("user", f"🎯 **Action Completed**: {action_data['title']}")
        add_chat_message("system", "🤖 Activating 4-agent analysis workflow...")
        
        recommendation = await st.session_state.orchestrator.process_action_outcome(
            action_data,
            st.session_state.business_model
        )

        # Add completion message
        if recommendation:
            add_chat_message("assistant", f"✅ **Analysis Complete**\n\n**Confidence Level**: {recommendation.confidence_level.value.title()}\n\n**Key Insight**: {recommendation.reasoning}")
            
            if recommendation.proposed_changes:
                changes_summary = f"💡 **Proposed {len(recommendation.proposed_changes)} changes** across your business model canvas"
                add_chat_message("assistant", changes_summary)

        return recommendation

    except Exception as e:
        add_chat_message("system", f"❌ **Error**: {str(e)}")
        st.error(f"Error in agent processing: {str(e)}")
        return None


def display_processing_section(action_data: Dict):
    """Display the processing section with enhanced visuals."""
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>🚀 AI Agent Processing</h3>', unsafe_allow_html=True)
    
    # Show default pipeline
    display_agent_pipeline()

    if st.button("🔄 Analyze Action Outcome", type="primary", use_container_width=True):
        if not check_api_key():
            st.error("Please configure Google API key first")
            return

        if not action_data.get("title") or not action_data.get("results_data"):
            st.warning("Please provide action title and results data")
            return

        # Show processing states
        progress_container = st.empty()
        status_container = st.empty()

        try:
            # Simulate processing steps with visual feedback
            with st.spinner("🤖 Initializing AI agents..."):
                # Show typing indicator
                with status_container.container():
                    display_typing_indicator()
                
                time.sleep(1)  # Demo delay
                
                # Run the async processing
                recommendation = asyncio.run(process_action_with_agents(action_data))

            if recommendation:
                st.session_state.current_recommendation = recommendation
                st.success("✅ Analysis completed successfully!")

                # Update chat with results
                st.rerun()

            else:
                st.error("❌ Analysis failed. Please check your inputs and try again.")

        except Exception as e:
            st.error(f"Processing error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)


def display_results_section():
    """Display the results section with enhanced styling."""
    if not st.session_state.current_recommendation:
        return

    recommendation = st.session_state.current_recommendation
    
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>📊 Analysis Results</h3>', unsafe_allow_html=True)

    # Overall confidence with styling
    confidence_class = f"confidence-{recommendation.confidence_level.value}"
    
    st.markdown(f"""
    <div class="{confidence_class}">
        <strong>🎯 Overall Confidence: {recommendation.confidence_level.value.title()}</strong><br><br>
        {recommendation.reasoning}
    </div>
    """, unsafe_allow_html=True)

    # Proposed Changes
    if recommendation.proposed_changes:
        st.subheader("💡 Proposed Changes")

        formatted_changes = format_proposed_changes(recommendation.proposed_changes)

        for i, change in enumerate(formatted_changes):
            confidence_class = f"confidence-{change['confidence_level'].lower()}"
            
            with st.expander(f"🔄 {change['section']}: {change['action']}", expanded=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Change:** {change['description']}")
                    st.write(f"**Reasoning:** {change['reasoning']}")

                with col2:
                    st.markdown(f"""
                    <div class="{confidence_class}" style="text-align: center; padding: 0.5rem;">
                        <strong>Confidence</strong><br>
                        {change['confidence']}
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.info("ℹ️ No changes recommended based on the analysis.")

    # Next Actions
    if recommendation.next_actions:
        st.subheader("🎯 Recommended Next Actions")
        for i, action in enumerate(recommendation.next_actions, 1):
            st.markdown(f"**{i}.** {action}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Before/After Comparison
    if recommendation.proposed_changes:
        display_before_after_comparison(recommendation.proposed_changes)


def display_before_after_comparison(changes: List[ProposedChange]):
    """Display before/after comparison with enhanced styling."""
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>⚖️ Before/After Comparison</h3>', unsafe_allow_html=True)

    # Apply changes temporarily for comparison
    updated_bmc = apply_changes_to_bmc(st.session_state.business_model, changes)
    comparison = create_before_after_comparison(st.session_state.business_model, updated_bmc)

    # Display only changed sections
    changed_sections = {k: v for k, v in comparison.items() if v["changed"]}

    if changed_sections:
        for section_name, section_data in changed_sections.items():
            with st.expander(f"📋 {section_name.replace('_', ' ').title()}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Before:**")
                    for item in section_data["before"]:
                        st.markdown(f'<div class="bmc-item">• {item}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown("**After:**")
                    for item in section_data["after"]:
                        st.markdown(f'<div class="bmc-item">• {item}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def display_controls_section():
    """Display the controls section with enhanced styling."""
    if not st.session_state.current_recommendation or not st.session_state.current_recommendation.proposed_changes:
        return

    recommendation = st.session_state.current_recommendation
    
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>⚙️ Controls</h3>', unsafe_allow_html=True)

    # Auto-mode handling
    if st.session_state.auto_mode:
        safe_changes = [change for change in recommendation.proposed_changes if validate_safety([change])]

        if safe_changes:
            st.success(f"🤖 Auto-mode: {len(safe_changes)} safe changes will be applied automatically")

            if st.button("✅ Apply Safe Changes Automatically", type="primary"):
                apply_changes(safe_changes, auto_applied=True)

        unsafe_changes = [change for change in recommendation.proposed_changes if not validate_safety([change])]
        if unsafe_changes:
            st.warning(f"⚠️ {len(unsafe_changes)} changes require manual review")

    else:
        # Manual mode
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Apply All Changes", type="primary", use_container_width=True):
                apply_changes(recommendation.proposed_changes, auto_applied=False)

        with col2:
            if st.button("🔍 Review Changes", use_container_width=True):
                display_change_review_modal()

        with col3:
            if st.button("❌ Reject Changes", use_container_width=True):
                st.session_state.current_recommendation = None
                add_chat_message("user", "❌ Changes rejected")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def apply_changes(changes: List[ProposedChange], auto_applied: bool = False):
    """Apply the proposed changes to the business model."""
    try:
        # Store old state for history
        old_bmc = st.session_state.business_model

        # Apply changes
        updated_bmc = apply_changes_to_bmc(old_bmc, changes)

        # Save updated business model
        st.session_state.business_model = updated_bmc
        save_business_model(updated_bmc)

        # Create and save change history
        history = create_change_history(
            old_bmc,
            updated_bmc,
            "current_action",
            changes,
            auto_applied
        )
        save_change_history(history)

        # Update session state
        st.session_state.change_history.append(history)

        # Display success message
        change_summary = generate_change_summary(changes)
        mode_text = "automatically" if auto_applied else "successfully"

        st.success(f"✅ Changes applied {mode_text}!")
        
        # Add to chat
        add_chat_message("system", f"✅ **Changes Applied {mode_text.title()}**\n\n{change_summary}")

        # Clear current recommendation
        st.session_state.current_recommendation = None

        # Export updated model
        export_business_model_to_csv(updated_bmc)

        st.rerun()

    except Exception as e:
        st.error(f"❌ Error applying changes: {str(e)}")


def display_change_review_modal():
    """Display detailed change review."""
    if not st.session_state.current_recommendation:
        return

    st.subheader("🔍 Detailed Change Review")

    changes = st.session_state.current_recommendation.proposed_changes
    selected_changes = []

    for i, change in enumerate(changes):
        col1, col2 = st.columns([1, 4])

        with col1:
            apply_change = st.checkbox(f"Apply", key=f"change_{i}", value=True)

        with col2:
            section_name = change.canvas_section.replace("_", " ").title()
            confidence_class = "confidence-high" if change.confidence_score >= 0.8 else ("confidence-medium" if change.confidence_score >= 0.6 else "confidence-low")
            
            st.markdown(f"""
            <div class="{confidence_class}">
                <strong>{section_name}</strong> - {change.change_type.value.title()}<br>
                <strong>Change:</strong> {change.proposed_value}<br>
                <strong>Confidence:</strong> {change.confidence_score:.0%}
            </div>
            """, unsafe_allow_html=True)

        if apply_change:
            selected_changes.append(change)

        st.divider()

    # Apply selected changes
    if st.button("✅ Apply Selected Changes", type="primary"):
        if selected_changes:
            apply_changes(selected_changes, auto_applied=False)
        else:
            st.warning("⚠️ No changes selected")


def display_current_bmc():
    """Display the current business model canvas with enhanced styling."""
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>🎯 Current Business Model Canvas</h3>', unsafe_allow_html=True)

    bmc = st.session_state.business_model

    # BMC Grid Layout
    st.markdown('<div class="bmc-grid">', unsafe_allow_html=True)
    
    sections = [
        ("key_partnerships", "🤝 Key Partnerships"),
        ("key_activities", "⚙️ Key Activities"), 
        ("key_resources", "💎 Key Resources"),
        ("value_propositions", "💡 Value Propositions"),
        ("customer_relationships", "🤝 Customer Relationships"),
        ("channels", "📡 Channels"),
        ("customer_segments", "👥 Customer Segments"),
        ("revenue_streams", "💰 Revenue Streams"),
        ("cost_structure", "💸 Cost Structure")
    ]

    for section_key, section_title in sections:
        st.markdown(f'<div class="bmc-section"><h4>{section_title}</h4>', unsafe_allow_html=True)
        
        items = getattr(bmc, section_key, [])
        if items:
            for item in items:
                st.markdown(f'<div class="bmc-item">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: var(--sigma-text-secondary); font-style: italic;">No items defined</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def display_sidebar():
    """Display the enhanced sidebar."""
    st.sidebar.markdown("### ⚙️ Configuration")

    # API Key status
    if check_api_key():
        st.sidebar.success("✅ Google API key configured")
    else:
        st.sidebar.error("❌ Google API key not found")
        st.sidebar.info("Set GOOGLE_API_KEY environment variable")

    # Model settings
    st.sidebar.markdown("### 🤖 AI Model Settings")
    model_name = st.sidebar.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0,
        help="Choose the Google Gemini model for analysis"
    )

    # Export options
    st.sidebar.markdown("### 📤 Export Options")
    if st.sidebar.button("💾 Export Business Model (CSV)"):
        if export_business_model_to_csv(st.session_state.business_model):
            st.sidebar.success("✅ Exported to business_model.csv")
        else:
            st.sidebar.error("❌ Export failed")

    # System information
    st.sidebar.markdown("### 📊 System Info")
    total_elements = sum(len(getattr(st.session_state.business_model, attr, []))
                        for attr in ['customer_segments', 'value_propositions', 'channels',
                                   'customer_relationships', 'revenue_streams', 'key_resources',
                                   'key_activities', 'key_partnerships', 'cost_structure'])
    
    st.sidebar.info(f"""
    **BMC Sections**: 9  
    **Total Elements**: {total_elements}  
    **Last Updated**: {st.session_state.business_model.last_updated.strftime('%Y-%m-%d %H:%M')}  
    **Auto-mode**: {'🟢 Enabled' if st.session_state.auto_mode else '🔴 Disabled'}
    """)


def display_metrics_dashboard():
    """Display key metrics dashboard."""
    st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
    st.markdown('<h3>📈 Canvas Metrics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_elements = sum(len(getattr(st.session_state.business_model, attr, []))
                            for attr in ['customer_segments', 'value_propositions', 'channels',
                                       'customer_relationships', 'revenue_streams', 'key_resources',
                                       'key_activities', 'key_partnerships', 'cost_structure'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_elements}</div>
            <div class="metric-label">Total Elements</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.business_model.customer_segments)}</div>
            <div class="metric-label">Customer Segments</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.business_model.revenue_streams)}</div>
            <div class="metric-label">Revenue Streams</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.change_history)}</div>
            <div class="metric-label">Changes Applied</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application function."""
    # Page config
    st.set_page_config(
        page_title="Agentic AI Actions Co-pilot",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load CSS and initialize
    load_sigma_css()
    initialize_session_state()

    # Header and auto-mode toggle
    display_header()
    display_auto_mode_toggle()
    
    # Sidebar
    display_sidebar()

    # Main content tabs with icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Action Analysis", 
        "💬 Chat Interface", 
        "🎯 Business Model", 
        "📊 Analytics"
    ])

    with tab1:
        # Action Analysis Tab
        action_data = display_input_section()
        display_processing_section(action_data)
        display_results_section()
        display_controls_section()

    with tab2:
        # Chat Interface Tab
        display_chat_interface()
        
        # Chat input
        if st.button("🎯 Simulate Action Completion", use_container_width=True):
            add_chat_message("user", "I just completed a market validation survey in Lagos")
            add_chat_message("system", "🤖 Detecting completed action... Analyzing outcomes...")
            st.rerun()

    with tab3:
        # Business Model Tab
        display_current_bmc()
        display_metrics_dashboard()

    with tab4:
        # Analytics Tab
        st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
        st.markdown('<h3>📈 Change History</h3>', unsafe_allow_html=True)

        if st.session_state.change_history:
            for i, history in enumerate(reversed(st.session_state.change_history[-10:])):
                with st.expander(f"Change #{len(st.session_state.change_history) - i}: {history.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Auto-applied:** {'✅ Yes' if history.auto_applied else '👤 Manual'}")
                        st.write(f"**Changes Applied:** {len(history.changes_applied)}")
                    
                    with col2:
                        for change in history.changes_applied:
                            st.write(f"• {change.canvas_section.replace('_', ' ').title()}: {change.change_type.value}")
        else:
            st.info("📝 No change history available yet. Complete some actions to see changes here!")

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Future analytics placeholder
        st.markdown('<div class="sigma-card">', unsafe_allow_html=True)
        st.markdown('<h3>🔮 Advanced Analytics</h3>', unsafe_allow_html=True)
        st.info("📊 Advanced insights, trend analysis, and predictive recommendations coming soon...")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()