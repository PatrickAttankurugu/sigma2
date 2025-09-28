"""
SIGMA Agentic AI Actions Co-pilot - Complete Application
Demonstrates: Business Design → Actions → AI Analysis → BMC Updates → Strategic Next Steps

Seedstars Senior AI Engineer Assignment - Option 2
Enhanced with: Strategic Next Steps Generation, Business Intelligence, Quality Validation
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv

from modules.business_design import BusinessDesignManager
from modules.bmc_canvas import BusinessModelCanvas
from modules.ai_engine import QualityEnhancedAI
from modules.ui_components import *
from modules.utils import setup_logging, SessionMetrics

# Load environment variables
load_dotenv()

# Initialize loggers
app_logger, ai_logger, bmc_logger, metrics_logger, quality_logger = setup_logging()

# Streamlit App Configuration
st.set_page_config(
    page_title="SIGMA Agentic AI Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def main():
    """Main Streamlit application with business design phase"""
    
    # Initialize session metrics
    if 'session_metrics' not in st.session_state:
        st.session_state.session_metrics = SessionMetrics()
    
    # Render header
    render_header()
    
    # Check API key
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
    if 'business_design_manager' not in st.session_state:
        st.session_state.business_design_manager = BusinessDesignManager()
    
    # Initialize persistent next steps storage
    if 'latest_next_steps' not in st.session_state:
        st.session_state.latest_next_steps = None
    if 'latest_next_steps_context' not in st.session_state:
        st.session_state.latest_next_steps_context = None

    # Main Application Flow
    if not st.session_state.bmc.is_complete():
        # Phase 1: Business Design
        render_business_design_phase()
    else:
        # Phase 2: Actions & Experiments
        render_actions_phase()

    # Sidebar with session info
    metrics = st.session_state.session_metrics.get_session_summary()
    render_sidebar_info(metrics, st.session_state.bmc)
    
    # Footer
    render_footer()


def render_business_design_phase():
    """Render the business design phase"""
    
    # Business design form
    completed = st.session_state.business_design_manager.render_business_design_form(
        st.session_state.bmc
    )
    
    if completed:
        # Record completion in metrics
        sections_count = sum(
            len(st.session_state.bmc.get_section(section)) 
            for section in st.session_state.bmc.get_section_names()
        )
        st.session_state.session_metrics.record_business_design_completed(sections_count)
        
        # Auto-refresh to move to next phase
        time.sleep(2)
        st.rerun()


def render_actions_phase():
    """Render the actions and experiments phase"""
    
    # Auto-mode toggle
    auto_mode_changed = st.toggle(
        "**Auto-mode**: Apply high-confidence changes (>80%) automatically", 
        value=st.session_state.auto_mode,
        help="When enabled, changes with >80% confidence will be applied automatically to your business model"
    )
    
    if auto_mode_changed != st.session_state.auto_mode:
        st.session_state.auto_mode = auto_mode_changed
        app_logger.info(f"Auto-mode toggled: {auto_mode_changed}")

    # Main layout
    col1, col2 = st.columns([1.5, 1.2])
    
    # Left Column: Business Model Canvas Display
    with col1:
        render_business_canvas(st.session_state.bmc)
    
    # Right Column: Action Input & Analysis
    with col2:
        # Get sample actions based on user's business
        sample_actions = st.session_state.business_design_manager.get_sample_actions(
            st.session_state.bmc
        )
        
        # Render action form (this handles state persistence)
        form_action_data, form_action_type = render_action_form(sample_actions)
        
        # Get current action data (persistent across reruns)
        current_action_data, current_action_type = get_current_action()
        
        # Use current persistent data if available, otherwise use form data
        action_data = current_action_data or form_action_data
        action_type = current_action_type or form_action_type
        
        # Show analyze button only if we have action data
        if action_data:
            analyze_clicked = st.button(
                "Analyze Action & Update Business Model", 
                use_container_width=True, 
                type="primary",
                key="analyze_button"
            )
            
            if analyze_clicked:
                # Clear previous next steps when starting new analysis
                st.session_state.latest_next_steps = None
                st.session_state.latest_next_steps_context = None
                
                with st.spinner("AI analyzing your action and generating strategic next steps..."):
                    try:
                        # Get AI recommendation with quality control and enhanced next steps
                        recommendation, quality = st.session_state.ai.analyze_action_with_quality_control(
                            action_data, st.session_state.bmc
                        )
                        
                        # Record metrics including quality data
                        retries_used = 1 if quality.overall_score < 0.6 else 0
                        st.session_state.session_metrics.record_action_analyzed(
                            action_data.get('outcome', 'Unknown'), 
                            quality.overall_score, 
                            retries_used
                        )
                        
                        # Record changes proposed
                        if recommendation["changes"]:
                            avg_confidence = sum(c.get('confidence', 0) for c in recommendation["changes"]) / len(recommendation["changes"])
                            st.session_state.session_metrics.record_changes_proposed(len(recommendation["changes"]), avg_confidence)
                        
                        # Store results in session state for display
                        st.session_state.latest_recommendation = recommendation
                        st.session_state.latest_quality = quality
                        st.session_state.latest_action_data = action_data
                        
                        # IMPORTANT: Store next steps separately before any clearing happens
                        if recommendation.get("next_steps"):
                            st.session_state.latest_next_steps = recommendation["next_steps"]
                            st.session_state.latest_next_steps_context = {
                                "action_title": action_data.get('title', 'Unknown'),
                                "action_outcome": action_data.get('outcome', 'Unknown'),
                                "timestamp": time.time()
                            }
                        
                        # Clear the current action after successful analysis
                        clear_current_action()
                        
                        # Display results
                        display_analysis_results(recommendation, quality, action_data)
                        
                    except Exception as e:
                        app_logger.error(f"Error during AI analysis: {e}")
                        st.error(f"Error during analysis: {str(e)}")
                        st.error("Please check your API key and try again.")
        
        # Display previous results if they exist (without next steps)
        elif hasattr(st.session_state, 'latest_recommendation') and st.session_state.latest_recommendation:
            st.info("Previous analysis results:")
            display_analysis_results(
                st.session_state.latest_recommendation, 
                st.session_state.latest_quality, 
                st.session_state.latest_action_data
            )
    
    # Display persistent next steps below the main interface
    display_persistent_next_steps()

    # Footer with session metrics
    metrics = st.session_state.session_metrics.get_session_summary()
    quality_data = st.session_state.ai.get_quality_dashboard_data()
    render_session_metrics(metrics, quality_data)


def display_analysis_results(recommendation: dict, quality, action_data: dict):
    """Display the AI analysis results with enhanced next steps"""
    
    # Display AI Analysis with Quality Indicator
    st.success("✅ Analysis Complete!")
    
    # Show quality indicator
    display_quality_indicator(quality)
    
    with st.expander("🧠 AI Analysis", expanded=True):
        st.write(recommendation["analysis"])
    
    # Show proposed changes with enhanced visuals
    if recommendation["changes"]:
        st.subheader("🔄 Proposed Business Model Updates")
        
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
            section_display = st.session_state.bmc.get_section_display_name(change['section'])
            
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
                changes_applied += apply_change_to_bmc(change)
            
            if changes_applied > 0:
                st.session_state.session_metrics.record_changes_applied(changes_applied, auto_applied=True)
                st.success(f"Auto-applied {changes_applied} high-confidence changes!")
                
                # FIXED: Only clear BMC changes, preserve next steps
                if 'latest_recommendation' in st.session_state:
                    # Next steps are already stored separately above
                    del st.session_state.latest_recommendation
                    
                time.sleep(1)
                st.rerun()
        
        # Manual controls (if auto-mode is off or there are non-auto changes)
        elif not st.session_state.auto_mode:
            col_apply, col_reject = st.columns(2)
            
            with col_apply:
                if st.button("Apply All Changes", use_container_width=True, key="apply_changes"):
                    for change in recommendation["changes"]:
                        if change.get("confidence", 0) >= 0.6:  # Apply medium+ confidence
                            changes_applied += apply_change_to_bmc(change)
                    
                    if changes_applied > 0:
                        st.session_state.session_metrics.record_changes_applied(changes_applied, auto_applied=False)
                    st.success(f"Applied {changes_applied} changes to your business model!")
                    
                    # Clear BMC changes but preserve next steps
                    if 'latest_recommendation' in st.session_state:
                        del st.session_state.latest_recommendation
                        
                    time.sleep(1)
                    st.rerun()
            
            with col_reject:
                if st.button("Reject Changes", use_container_width=True, key="reject_changes"):
                    app_logger.info("User rejected all proposed changes")
                    st.info("Changes rejected. Business model remains unchanged.")
                    
                    # Clear BMC changes but preserve next steps
                    if 'latest_recommendation' in st.session_state:
                        del st.session_state.latest_recommendation
    
    else:
        st.info("No business model changes suggested based on this action.")
    
    # Show next steps inline if we have them and no auto-mode clearing happened
    next_steps = recommendation.get("next_steps", [])
    if next_steps and 'latest_recommendation' in st.session_state:
        display_enhanced_next_steps(next_steps)


def display_persistent_next_steps():
    """Display next steps that persist even after BMC changes are applied"""
    if (hasattr(st.session_state, 'latest_next_steps') and 
        st.session_state.latest_next_steps and 
        not hasattr(st.session_state, 'latest_recommendation')):
        
        # Show persistent next steps with context
        st.markdown("---")
        st.subheader("🎯 Your Strategic Action Plan")
        
        if st.session_state.latest_next_steps_context:
            context = st.session_state.latest_next_steps_context
            st.caption(f"Based on: **{context['action_title']}** ({context['action_outcome']})")
        
        display_enhanced_next_steps(st.session_state.latest_next_steps)
        
        # Option to clear next steps
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Clear Action Plan", use_container_width=True):
                st.session_state.latest_next_steps = None
                st.session_state.latest_next_steps_context = None
                st.rerun()


def apply_change_to_bmc(change: dict) -> int:
    """Apply a single change to the business model canvas"""
    section = change["section"]
    current_items = st.session_state.bmc.get_section(section)
    
    if change["type"] == "add":
        if change["new"] not in current_items:
            current_items.append(change["new"])
            st.session_state.bmc.update_section(section, current_items)
            return 1
            
    elif change["type"] == "modify" and change.get("current"):
        try:
            idx = current_items.index(change["current"])
            current_items[idx] = change["new"]
            st.session_state.bmc.update_section(section, current_items)
            return 1
        except ValueError:
            current_items.append(change["new"])
            st.session_state.bmc.update_section(section, current_items)
            return 1
            
    elif change["type"] == "remove" and change.get("current"):
        try:
            current_items.remove(change["current"])
            st.session_state.bmc.update_section(section, current_items)
            return 1
        except ValueError:
            pass
    
    return 0


if __name__ == "__main__":
    main()