"""
Streamlit interface for the Agentic AI Actions Co-pilot system.

This application demonstrates a sophisticated multi-agent workflow for
automatically updating business model canvases based on completed action outcomes.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

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


# Streamlit Configuration
st.set_page_config(
    page_title="Agentic AI Actions Co-pilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }
    .confidence-high { background-color: #d4edda; padding: 0.5rem; border-radius: 0.25rem; }
    .confidence-medium { background-color: #fff3cd; padding: 0.5rem; border-radius: 0.25rem; }
    .confidence-low { background-color: #f8d7da; padding: 0.5rem; border-radius: 0.25rem; }
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .status-pending { background-color: #e2e3e5; color: #6c757d; }
    .status-running { background-color: #b3d4fc; color: #0c5460; }
    .status-completed { background-color: #d4edda; color: #155724; }
    .status-failed { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


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


def check_api_key() -> bool:
    """Check if Google API key is configured."""
    return bool(os.getenv("GOOGLE_API_KEY"))


def display_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">Agentic AI Actions Co-pilot - Demo</h1>', unsafe_allow_html=True)
    st.markdown('''
    <div class="sub-header">
    Intelligent Business Model Canvas Updates through Multi-Agent AI Analysis<br>
    <em>Specialized for African Fintech Markets</em>
    </div>
    ''', unsafe_allow_html=True)

    # Auto-mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.session_state.auto_mode = st.toggle(
            "Auto-mode",
            value=st.session_state.auto_mode,
            help="Automatically apply high-confidence changes without manual approval"
        )


def display_sidebar():
    """Display the sidebar with settings and information."""
    st.sidebar.header("Configuration")

    # API Key status
    if check_api_key():
        st.sidebar.success("Google API key configured")
    else:
        st.sidebar.error("Google API key not found")
        st.sidebar.info("Set GOOGLE_API_KEY environment variable")

    # Model settings
    st.sidebar.subheader("AI Model Settings")
    model_name = st.sidebar.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0,
        help="Choose the Google Gemini model for analysis"
    )

    # Persist selected model for use when creating the orchestrator
    st.session_state.model_name = model_name

    # Export options
    st.sidebar.subheader("Export Options")
    if st.sidebar.button("Export Business Model (CSV)"):
        if export_business_model_to_csv(st.session_state.business_model):
            st.sidebar.success("Exported to business_model.csv")
        else:
            st.sidebar.error("Export failed")

    # System information
    st.sidebar.subheader("System Info")
    st.sidebar.info(f"""
    **Current BMC Sections:** 9
    **Last Updated:** {st.session_state.business_model.last_updated.strftime('%Y-%m-%d %H:%M')}
    **Auto-mode:** {'Enabled' if st.session_state.auto_mode else 'Disabled'}
    """)


def display_input_section():
    """Display the input section for action selection."""
    st.header("Action Outcome Input")

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
            help="Choose from realistic Ghanaian fintech scenarios"
        )

        selected_action = get_sample_action_by_title(selected_title)

        # Display action details
        with st.expander("Action Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Title:** {selected_action.title}")
                st.write(f"**Outcome:** {selected_action.outcome.value.title()}")
            with col2:
                st.write(f"**Completed:** {selected_action.completion_date.strftime('%Y-%m-%d')}")

            st.write("**Description:**")
            st.write(selected_action.description)

            st.write("**Results:**")
            st.text_area(
                "Results",
                selected_action.results_data,
                height=150,
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
        # Custom input
        st.write("**Enter custom action outcome:**")

        col1, col2 = st.columns(2)
        with col1:
            custom_title = st.text_input("Action Title", placeholder="e.g., Customer Survey Results")
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

    return action_data


async def process_action_with_agents(action_data: Dict) -> Optional[AgentRecommendation]:
    """Process action through the multi-agent workflow."""
    try:
        if not st.session_state.orchestrator:
            selected_model = st.session_state.get("model_name", "gemini-1.5-flash")
            st.session_state.orchestrator = AgenticOrchestrator(model_name=selected_model)

        recommendation = await st.session_state.orchestrator.process_action_outcome(
            action_data,
            st.session_state.business_model
        )

        return recommendation

    except Exception as e:
        st.error(f"Error in agent processing: {str(e)}")
        return None


def display_processing_section(action_data: Dict):
    """Display the processing section with agent status."""
    st.header("AI Agent Processing")

    if st.button("Analyze Action Outcome", type="primary", use_container_width=True):
        if not check_api_key():
            st.error("Please configure Google API key first")
            return

        if not action_data.get("title") or not action_data.get("results_data"):
            st.warning("Please provide action title and results data")
            return

        # Create processing placeholders
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        try:
            # Run the async processing
            with st.spinner("Initializing AI agents..."):
                recommendation = asyncio.run(process_action_with_agents(action_data))

            if recommendation:
                st.session_state.current_recommendation = recommendation
                st.success("Analysis completed successfully!")

                # Display agent workflow summary
                with st.expander("Agent Workflow Summary", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown("**Action Detection**")
                        st.markdown('<div class="status-completed agent-status">Completed</div>', unsafe_allow_html=True)
                        st.caption("Validated and structured action data")

                    with col2:
                        st.markdown("**Outcome Analysis**")
                        st.markdown('<div class="status-completed agent-status">Completed</div>', unsafe_allow_html=True)
                        st.caption("Analyzed business implications")

                    with col3:
                        st.markdown("**Canvas Updates**")
                        st.markdown('<div class="status-completed agent-status">Completed</div>', unsafe_allow_html=True)
                        st.caption("Generated specific BMC updates")

                    with col4:
                        st.markdown("**Next Steps**")
                        st.markdown('<div class="status-completed agent-status">Completed</div>', unsafe_allow_html=True)
                        st.caption("Suggested follow-up actions")

            else:
                st.error("Analysis failed. Please check your inputs and try again.")

        except Exception as e:
            st.error(f"Processing error: {str(e)}")


def display_results_section():
    """Display the results section with recommendations."""
    if not st.session_state.current_recommendation:
        return

    recommendation = st.session_state.current_recommendation
    st.header("Analysis Results")

    # Overall confidence and summary
    confidence_colors = {
        "high": "confidence-high",
        "medium": "confidence-medium",
        "low": "confidence-low"
    }

    confidence_class = confidence_colors.get(recommendation.confidence_level.value, "confidence-low")

    st.markdown(f"""
    <div class="{confidence_class}">
    <strong>Overall Confidence: {recommendation.confidence_level.value.title()}</strong><br>
    {recommendation.reasoning}
    </div>
    """, unsafe_allow_html=True)

    # Proposed Changes
    if recommendation.proposed_changes:
        st.subheader("Proposed Changes")

        formatted_changes = format_proposed_changes(recommendation.proposed_changes)

        for i, change in enumerate(formatted_changes):
            with st.expander(f"{change['section']}: {change['action']}", expanded=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Change:** {change['description']}")
                    st.write(f"**Reasoning:** {change['reasoning']}")

                with col2:
                    confidence_class = f"confidence-{change['confidence_level'].lower()}"
                    st.markdown(f'<div class="{confidence_class}">Confidence: {change["confidence"]}</div>',
                                unsafe_allow_html=True)

    else:
        st.info("No changes recommended based on the analysis.")

    # Next Actions
    if recommendation.next_actions:
        st.subheader("Recommended Next Actions")
        for i, action in enumerate(recommendation.next_actions, 1):
            st.write(f"{i}. {action}")

    # Before/After Comparison
    if recommendation.proposed_changes:
        st.subheader("Before/After Comparison")

        # Apply changes temporarily for comparison
        updated_bmc = apply_changes_to_bmc(st.session_state.business_model, recommendation.proposed_changes)
        comparison = create_before_after_comparison(st.session_state.business_model, updated_bmc)

        # Display only changed sections
        changed_sections = {k: v for k, v in comparison.items() if v["changed"]}

        if changed_sections:
            for section_name, section_data in changed_sections.items():
                with st.expander(f"{section_name.replace('_', ' ').title()}", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Before:**")
                        for item in section_data["before"]:
                            st.write(f"• {item}")

                    with col2:
                        st.write("**After:**")
                        for item in section_data["after"]:
                            st.write(f"• {item}")


def display_controls_section():
    """Display the controls section for applying changes."""
    if not st.session_state.current_recommendation or not st.session_state.current_recommendation.proposed_changes:
        return

    st.header("Controls")

    recommendation = st.session_state.current_recommendation

    # Auto-mode handling
    if st.session_state.auto_mode:
        safe_changes = [change for change in recommendation.proposed_changes if validate_safety([change])]

        if safe_changes:
            st.success(f"Auto-mode: {len(safe_changes)} safe changes will be applied automatically")

            if st.button("Apply Safe Changes Automatically", type="primary"):
                apply_changes(safe_changes, auto_applied=True)

        unsafe_changes = [change for change in recommendation.proposed_changes if not validate_safety([change])]
        if unsafe_changes:
            st.warning(f"{len(unsafe_changes)} changes require manual review")

    else:
        # Manual mode
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Apply All Changes", type="primary", use_container_width=True):
                apply_changes(recommendation.proposed_changes, auto_applied=False)

        with col2:
            if st.button("Review Changes", use_container_width=True):
                display_change_review_modal()

        with col3:
            if st.button("Reject Changes", use_container_width=True):
                st.session_state.current_recommendation = None
                st.rerun()


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
            "current_action",  # In a real system, use actual action ID
            changes,
            auto_applied
        )
        save_change_history(history)

        # Update session state
        st.session_state.change_history.append(history)

        # Display success message
        change_summary = generate_change_summary(changes)
        mode_text = "automatically" if auto_applied else "successfully"

        st.success(f"Changes applied {mode_text}!")
        st.info(change_summary)

        # Clear current recommendation
        st.session_state.current_recommendation = None

        # Export updated model
        export_business_model_to_csv(updated_bmc)

        st.rerun()

    except Exception as e:
        st.error(f"Error applying changes: {str(e)}")


def display_change_review_modal():
    """Display detailed change review in a modal-style expansion."""
    if not st.session_state.current_recommendation:
        return

    st.subheader("Detailed Change Review")

    changes = st.session_state.current_recommendation.proposed_changes

    # Allow users to select which changes to apply
    selected_changes = []

    for i, change in enumerate(changes):
        col1, col2 = st.columns([1, 4])

        with col1:
            apply_change = st.checkbox(f"Apply", key=f"change_{i}", value=True)

        with col2:
            section_name = change.canvas_section.replace("_", " ").title()
            st.write(f"**{section_name}** - {change.change_type.value.title()}")
            st.write(f"Change: {change.proposed_value}")
            st.write(f"Confidence: {change.confidence_score:.0%}")

        if apply_change:
            selected_changes.append(change)

        st.divider()

    # Apply selected changes
    if st.button("Apply Selected Changes", type="primary"):
        if selected_changes:
            apply_changes(selected_changes, auto_applied=False)
        else:
            st.warning("No changes selected")


def display_current_bmc():
    """Display the current business model canvas."""
    st.header("Current Business Model Canvas")

    bmc = st.session_state.business_model

    # Create BMC layout
    col1, col2, col3 = st.columns([2, 3, 2])

    # Left column
    with col1:
        st.subheader("Key Partnerships")
        for partnership in bmc.key_partnerships:
            st.write(f"• {partnership}")

        st.subheader("Key Activities")
        for activity in bmc.key_activities:
            st.write(f"• {activity}")

        st.subheader("Key Resources")
        for resource in bmc.key_resources:
            st.write(f"• {resource}")

    # Middle column
    with col2:
        st.subheader("Value Propositions")
        for vp in bmc.value_propositions:
            st.write(f"• {vp}")

        st.subheader("Customer Relationships")
        for relationship in bmc.customer_relationships:
            st.write(f"• {relationship}")

        st.subheader("Channels")
        for channel in bmc.channels:
            st.write(f"• {channel}")

    # Right column
    with col3:
        st.subheader("Customer Segments")
        for segment in bmc.customer_segments:
            st.write(f"• {segment}")

        st.subheader("Revenue Streams")
        for revenue in bmc.revenue_streams:
            st.write(f"• {revenue}")

        st.subheader("Cost Structure")
        for cost in bmc.cost_structure:
            st.write(f"• {cost}")


def main():
    """Main application function."""
    initialize_session_state()

    display_header()
    display_sidebar()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Action Analysis", "Business Model", "History & Analytics"])

    with tab1:
        action_data = display_input_section()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        display_processing_section(action_data)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        display_results_section()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        display_controls_section()

    with tab2:
        display_current_bmc()

        # BMC Statistics
        st.subheader("Canvas Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_elements = sum(len(getattr(st.session_state.business_model, attr, []))
                                for attr in ['customer_segments', 'value_propositions', 'channels',
                                           'customer_relationships', 'revenue_streams', 'key_resources',
                                           'key_activities', 'key_partnerships', 'cost_structure'])
            st.metric("Total Elements", total_elements)

        with col2:
            st.metric("Customer Segments", len(st.session_state.business_model.customer_segments))

        with col3:
            st.metric("Revenue Streams", len(st.session_state.business_model.revenue_streams))

        with col4:
            st.metric("Key Partnerships", len(st.session_state.business_model.key_partnerships))

    with tab3:
        st.subheader("Change History")

        if st.session_state.change_history:
            for i, history in enumerate(reversed(st.session_state.change_history[-10:])):  # Last 10 changes
                with st.expander(f"Change #{len(st.session_state.change_history) - i}: {history.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Auto-applied:** {'Yes' if history.auto_applied else 'No'}")
                    st.write(f"**Changes Applied:** {len(history.changes_applied)}")

                    for change in history.changes_applied:
                        st.write(f"• {change.canvas_section.replace('_', ' ').title()}: {change.change_type.value}")
        else:
            st.info("No change history available yet. Make some changes to see them here!")

        # Analytics placeholder
        st.subheader("Analytics")
        st.info("Advanced analytics and insights coming soon...")


if __name__ == "__main__":
    main()