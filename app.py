"""
SIGMA Agentic AI Actions Co-pilot - Complete Application with Multi-Agent System
Demonstrates: Business Design → Actions → Multi-Agent AI Analysis → BMC Updates → Strategic Next Steps

Features:
- 4 specialized AI agents (Strategy, Market Research, Product, Execution)
- Real-time streaming via WebSocket
- Collaborative agent workflow
- Enhanced business insights
"""

import streamlit as st
import os
import time
import uuid
from dotenv import load_dotenv

from modules.business_design import BusinessDesignManager
from modules.bmc_canvas import BusinessModelCanvas
from modules.ai_engine import QualityEnhancedAI
from modules.multi_agent_system import MultiAgentSystem
from modules.websocket_client import get_websocket_client
from modules.ui_components import *
from modules.utils import setup_logging, SessionMetrics

load_dotenv()

app_logger, ai_logger, bmc_logger, metrics_logger, quality_logger = setup_logging()

st.set_page_config(
    page_title="SIGMA Agentic AI Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def main():
    """Main Streamlit application with multi-agent system and streaming"""

    if 'session_metrics' not in st.session_state:
        st.session_state.session_metrics = SessionMetrics()

    render_header()

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

    # Initialize BMC
    if 'bmc' not in st.session_state:
        st.session_state.bmc = BusinessModelCanvas()

    # Initialize Multi-Agent System (preferred) or fallback to single AI
    if 'multi_agent_system' not in st.session_state:
        try:
            st.session_state.multi_agent_system = MultiAgentSystem(st.session_state.bmc, api_key)
            app_logger.info("Multi-agent system initialized successfully")
        except Exception as e:
            app_logger.warning(f"Failed to initialize multi-agent system: {e}")
            st.session_state.multi_agent_system = None

    # Fallback to single AI if multi-agent fails
    if 'ai' not in st.session_state:
        try:
            st.session_state.ai = QualityEnhancedAI(api_key)
        except Exception as e:
            app_logger.error(f"Failed to initialize AI: {e}")
            st.error(f"Failed to initialize AI: {e}")
            st.stop()

    # Initialize WebSocket client (optional - for streaming)
    if 'use_streaming' not in st.session_state:
        st.session_state.use_streaming = False
    if 'ws_client' not in st.session_state:
        st.session_state.ws_client = None
    if 'ws_session_id' not in st.session_state:
        st.session_state.ws_session_id = str(uuid.uuid4())

    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False
    if 'business_design_manager' not in st.session_state:
        st.session_state.business_design_manager = BusinessDesignManager()

    if 'latest_next_steps' not in st.session_state:
        st.session_state.latest_next_steps = None
    if 'latest_next_steps_context' not in st.session_state:
        st.session_state.latest_next_steps_context = None

    if not st.session_state.bmc.is_complete():
        render_business_design_phase()
    else:
        render_actions_phase()

    metrics = st.session_state.session_metrics.get_session_summary()
    render_sidebar_info(metrics, st.session_state.bmc)
    
    render_footer()


def render_business_design_phase():
    """Render the business design phase"""
    
    completed = st.session_state.business_design_manager.render_business_design_form(
        st.session_state.bmc
    )
    
    if completed:
        sections_count = sum(
            len(st.session_state.bmc.get_section(section)) 
            for section in st.session_state.bmc.get_section_names()
        )
        st.session_state.session_metrics.record_business_design_completed(sections_count)
        
        time.sleep(2)
        st.rerun()


def render_actions_phase():
    """Render the actions and experiments phase with multi-agent support"""

    col_auto, col_streaming = st.columns(2)

    with col_auto:
        auto_mode_changed = st.toggle(
            "**Auto-mode**: Apply high-confidence changes (>80%) automatically",
            value=st.session_state.auto_mode,
            help="When enabled, changes with >80% confidence will be applied automatically to your business model"
        )

        if auto_mode_changed != st.session_state.auto_mode:
            st.session_state.auto_mode = auto_mode_changed
            app_logger.info(f"Auto-mode toggled: {auto_mode_changed}")

    with col_streaming:
        streaming_mode = st.toggle(
            "**Multi-Agent Streaming**: Real-time agent collaboration",
            value=st.session_state.use_streaming,
            help="Enable real-time streaming to see all 4 agents working together (requires WebSocket server)"
        )

        if streaming_mode != st.session_state.use_streaming:
            st.session_state.use_streaming = streaming_mode
            app_logger.info(f"Streaming mode toggled: {streaming_mode}")

            if streaming_mode and not st.session_state.ws_client:
                try:
                    st.session_state.ws_client = get_websocket_client()
                    if st.session_state.ws_client:
                        st.success("✅ Connected to multi-agent streaming server")
                        app_logger.info("WebSocket client connected")
                    else:
                        st.session_state.use_streaming = False
                        st.error("❌ Could not connect to streaming server. Using non-streaming mode.")
                except Exception as e:
                    st.session_state.use_streaming = False
                    st.error(f"❌ Streaming not available: {e}")

    # Show multi-agent system status
    if st.session_state.multi_agent_system:
        st.info("🤖 Multi-Agent System Active: Strategy • Market Research • Product • Execution")

    col1, col2 = st.columns([1.5, 1.2])
    
    with col1:
        render_business_canvas(st.session_state.bmc)
    
    with col2:
        sample_actions = st.session_state.business_design_manager.get_sample_actions(
            st.session_state.bmc
        )
        
        form_action_data, form_action_type = render_action_form(sample_actions)
        
        current_action_data, current_action_type = get_current_action()
        
        action_data = current_action_data or form_action_data
        action_type = current_action_type or form_action_type
        
        if action_data:
            analyze_clicked = st.button(
                "Analyze Action & Update Business Model", 
                use_container_width=True, 
                type="primary",
                key="analyze_button"
            )
            
            if analyze_clicked:
                st.session_state.latest_next_steps = None
                st.session_state.latest_next_steps_context = None

                # Use multi-agent system if available, otherwise fallback to single AI
                use_multi_agent = st.session_state.multi_agent_system is not None

                if use_multi_agent:
                    analyze_with_multi_agent_system(action_data)
                else:
                    with st.spinner("AI analyzing your action and generating strategic next steps..."):
                        try:
                            recommendation, quality = st.session_state.ai.analyze_action_with_quality_control(
                                action_data, st.session_state.bmc
                            )

                            retries_used = 1 if quality.overall_score < 0.6 else 0
                            st.session_state.session_metrics.record_action_analyzed(
                                action_data.get('outcome', 'Unknown'),
                                quality.overall_score,
                                retries_used
                            )

                            if recommendation["changes"]:
                                avg_confidence = sum(c.get('confidence', 0) for c in recommendation["changes"]) / len(recommendation["changes"])
                                st.session_state.session_metrics.record_changes_proposed(len(recommendation["changes"]), avg_confidence)

                            st.session_state.latest_recommendation = recommendation
                            st.session_state.latest_quality = quality
                            st.session_state.latest_action_data = action_data

                            if recommendation.get("next_steps"):
                                st.session_state.latest_next_steps = recommendation["next_steps"]
                                st.session_state.latest_next_steps_context = {
                                    "action_title": action_data.get('title', 'Unknown'),
                                    "action_outcome": action_data.get('outcome', 'Unknown'),
                                    "timestamp": time.time()
                                }

                            clear_current_action()

                            display_analysis_results(recommendation, quality, action_data)

                        except Exception as e:
                            app_logger.error(f"Error during AI analysis: {e}")
                            st.error(f"Error during analysis: {str(e)}")
                            st.error("Please check your API key and try again.")
        
        elif hasattr(st.session_state, 'latest_recommendation') and st.session_state.latest_recommendation:
            st.info("Previous analysis results:")
            display_analysis_results(
                st.session_state.latest_recommendation, 
                st.session_state.latest_quality, 
                st.session_state.latest_action_data
            )
    
    display_persistent_next_steps()

    metrics = st.session_state.session_metrics.get_session_summary()
    quality_data = st.session_state.ai.get_quality_dashboard_data()
    render_session_metrics(metrics, quality_data)


def analyze_with_multi_agent_system(action_data: dict):
    """Analyze action using the multi-agent system with optional streaming"""
    try:
        # Use streaming mode if enabled and client is available
        if st.session_state.use_streaming and st.session_state.ws_client:
            analyze_with_streaming(action_data)
        else:
            analyze_without_streaming(action_data)

    except Exception as e:
        app_logger.error(f"Error in multi-agent analysis: {e}")
        st.error(f"Error during multi-agent analysis: {str(e)}")


def analyze_with_streaming(action_data: dict):
    """Analyze action with real-time streaming"""
    st.markdown("### 🤖 Multi-Agent Analysis in Progress")

    # Create placeholder for agent updates
    agent_status_placeholder = st.empty()
    progress_placeholder = st.empty()

    try:
        # Initialize session if not already done
        if not st.session_state.ws_client.session_id:
            bmc_data = {
                'customer_segments': st.session_state.bmc.get_section('customer_segments'),
                'value_propositions': st.session_state.bmc.get_section('value_propositions'),
                'business_models': st.session_state.bmc.get_section('business_models'),
                'market_opportunities': st.session_state.bmc.get_section('market_opportunities')
            }
            st.session_state.ws_client.init_session(st.session_state.ws_session_id, bmc_data)

        # Start analysis
        with st.spinner("Connecting to multi-agent system..."):
            # Send analysis request
            result = st.session_state.ws_client.analyze_action(action_data)

            # Show streaming updates while waiting
            for i in range(100):
                updates = st.session_state.ws_client.get_stream_updates()
                if updates:
                    for update in updates:
                        agent = update.get('agent', 'system')
                        message = update.get('message', '')
                        agent_status_placeholder.info(f"**{agent.upper()}**: {message}")

                time.sleep(0.1)

                if result:
                    break

        # Process result
        recommendation = {
            'analysis': result.get('analysis', ''),
            'changes': result.get('changes', []),
            'next_steps': result.get('next_steps', [])
        }

        # Record metrics
        st.session_state.session_metrics.record_action_analyzed(
            action_data.get('outcome', 'Unknown'),
            0.9,  # Multi-agent system is high quality
            0
        )

        if recommendation["changes"]:
            avg_confidence = sum(c.get('confidence', 0) for c in recommendation["changes"]) / len(recommendation["changes"])
            st.session_state.session_metrics.record_changes_proposed(len(recommendation["changes"]), avg_confidence)

        # Store results
        st.session_state.latest_recommendation = recommendation
        st.session_state.latest_quality = None  # No quality object for multi-agent
        st.session_state.latest_action_data = action_data

        if recommendation.get("next_steps"):
            st.session_state.latest_next_steps = recommendation["next_steps"]
            st.session_state.latest_next_steps_context = {
                "action_title": action_data.get('title', 'Unknown'),
                "action_outcome": action_data.get('outcome', 'Unknown'),
                "timestamp": time.time()
            }

        clear_current_action()
        display_analysis_results(recommendation, None, action_data)

    except Exception as e:
        app_logger.error(f"Error in streaming analysis: {e}")
        st.error(f"Streaming analysis failed: {e}. Falling back to non-streaming mode.")
        analyze_without_streaming(action_data)


def analyze_without_streaming(action_data: dict):
    """Analyze action without streaming (direct multi-agent call)"""
    with st.spinner("🤖 Multi-agent system analyzing your action..."):
        try:
            # Use multi-agent system directly
            result = st.session_state.multi_agent_system.analyze_action(action_data)

            recommendation = {
                'analysis': result.get('analysis', ''),
                'changes': result.get('changes', []),
                'next_steps': result.get('next_steps', [])
            }

            # Show agent insights if available
            if result.get('agent_insights'):
                with st.expander("🔍 Agent Insights", expanded=False):
                    insights = result['agent_insights']
                    st.markdown(f"**Strategy Agent**: {insights.get('strategy', 'N/A')}")
                    st.markdown(f"**Market Research Agent**: {insights.get('market', 'N/A')}")
                    st.markdown(f"**Product Agent**: {insights.get('product', 'N/A')}")

            # Record metrics
            st.session_state.session_metrics.record_action_analyzed(
                action_data.get('outcome', 'Unknown'),
                0.9,  # Multi-agent system is high quality
                0
            )

            if recommendation["changes"]:
                avg_confidence = sum(c.get('confidence', 0) for c in recommendation["changes"]) / len(recommendation["changes"])
                st.session_state.session_metrics.record_changes_proposed(len(recommendation["changes"]), avg_confidence)

            # Store results
            st.session_state.latest_recommendation = recommendation
            st.session_state.latest_quality = None  # No quality object for multi-agent
            st.session_state.latest_action_data = action_data

            if recommendation.get("next_steps"):
                st.session_state.latest_next_steps = recommendation["next_steps"]
                st.session_state.latest_next_steps_context = {
                    "action_title": action_data.get('title', 'Unknown'),
                    "action_outcome": action_data.get('outcome', 'Unknown'),
                    "timestamp": time.time()
                }

            clear_current_action()
            display_analysis_results(recommendation, None, action_data)

        except Exception as e:
            app_logger.error(f"Error in multi-agent analysis: {e}")
            raise


def display_analysis_results(recommendation: dict, quality, action_data: dict):
    """Display the AI analysis results"""
    
    st.success("Analysis Complete!")

    # Display quality indicator only if quality object exists (single AI mode)
    if quality:
        display_quality_indicator(quality)
    else:
        st.info("🤖 Analysis completed by Multi-Agent System")
    
    with st.expander("AI Analysis", expanded=True):
        st.write(recommendation["analysis"])
    
    if recommendation["changes"]:
        st.subheader("Proposed Business Model Updates")
        
        display_change_preview(recommendation["changes"], st.session_state.bmc)
        
        high_confidence_changes = []
        changes_applied = 0
        
        st.subheader("Individual Changes")
        for i, change in enumerate(recommendation["changes"]):
            confidence = change.get("confidence", 0)
            
            if confidence >= 0.8:
                high_confidence_changes.append(change)
            
            section_display = st.session_state.bmc.get_section_display_name(change['section'])
            
            with st.container():
                st.markdown(f"**{section_display}** - {change['type'].title()}")
                
                display_confidence_indicator(confidence)
                
                st.write(f"**Change:** {change['new']}")
                st.write(f"**Reasoning:** {change['reason']}")
                st.markdown("---")
        
        if st.session_state.auto_mode and high_confidence_changes:
            st.info(f"**Auto-mode Active**: Applying {len(high_confidence_changes)} high-confidence changes...")
            
            for change in high_confidence_changes:
                changes_applied += apply_change_to_bmc(change)
            
            if changes_applied > 0:
                st.session_state.session_metrics.record_changes_applied(changes_applied, auto_applied=True)
                st.success(f"Auto-applied {changes_applied} high-confidence changes!")
                
                if 'latest_recommendation' in st.session_state:
                    del st.session_state.latest_recommendation
                    
                time.sleep(1)
                st.rerun()
        
        elif not st.session_state.auto_mode:
            col_apply, col_reject = st.columns(2)
            
            with col_apply:
                if st.button("Apply All Changes", use_container_width=True, key="apply_changes"):
                    for change in recommendation["changes"]:
                        if change.get("confidence", 0) >= 0.6:
                            changes_applied += apply_change_to_bmc(change)
                    
                    if changes_applied > 0:
                        st.session_state.session_metrics.record_changes_applied(changes_applied, auto_applied=False)
                    st.success(f"Applied {changes_applied} changes to your business model!")
                    
                    if 'latest_recommendation' in st.session_state:
                        del st.session_state.latest_recommendation
                        
                    time.sleep(1)
                    st.rerun()
            
            with col_reject:
                if st.button("Reject Changes", use_container_width=True, key="reject_changes"):
                    app_logger.info("User rejected all proposed changes")
                    st.info("Changes rejected. Business model remains unchanged.")
                    
                    if 'latest_recommendation' in st.session_state:
                        del st.session_state.latest_recommendation
    
    else:
        st.info("No business model changes suggested based on this action.")
    
    next_steps = recommendation.get("next_steps", [])
    if next_steps and 'latest_recommendation' in st.session_state:
        display_enhanced_next_steps(next_steps)


def display_persistent_next_steps():
    """Display next steps that persist even after BMC changes are applied"""
    if (hasattr(st.session_state, 'latest_next_steps') and 
        st.session_state.latest_next_steps and 
        not hasattr(st.session_state, 'latest_recommendation')):
        
        st.markdown("---")
        st.subheader("Your Strategic Action Plan")
        
        if st.session_state.latest_next_steps_context:
            context = st.session_state.latest_next_steps_context
            st.caption(f"Based on: **{context['action_title']}** ({context['action_outcome']})")
        
        display_enhanced_next_steps(st.session_state.latest_next_steps)
        
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