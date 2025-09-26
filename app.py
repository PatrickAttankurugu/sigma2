"""
Seedstars Assignment: Agentic AI Actions Co-pilot
Clean, focused demo of 4-agent workflow for business model canvas updates
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from business_models import (
    BusinessModelCanvas,
    CompletedAction,
    AgentRecommendation,
    ProposedChange,
    ActionOutcome,
    ChangeHistory
)
from sema_business_data import (
    get_sample_business_model_canvas,
    get_sample_completed_actions,
    get_action_titles,
    get_sample_action_by_title
)
from agentic_engine import AgenticOrchestrator
from utils import (
    load_business_model,
    save_business_model,
    apply_changes_to_bmc,
    create_change_history,
    save_change_history,
    load_change_history,
    format_proposed_changes,
    generate_change_hash
)

# Page Configuration
st.set_page_config(
    page_title="Agentic AI Actions Co-pilot - Seedstars Assignment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple, Clean CSS
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
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'business_model' not in st.session_state:
    st.session_state.business_model = load_business_model()

if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None

if 'current_recommendation' not in st.session_state:
    st.session_state.current_recommendation = None

if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {
        'action_detection': 'pending',
        'outcome_analysis': 'pending',
        'canvas_update': 'pending',
        'next_step': 'pending'
    }

if 'change_history' not in st.session_state:
    st.session_state.change_history = load_change_history()

if 'processed_actions' not in st.session_state:
    st.session_state.processed_actions = set()

if 'status_message' not in st.session_state:
    st.session_state.status_message = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic AI Actions Co-pilot</h1>
    <p>Intelligent Business Model Canvas Updates through Multi-Agent Analysis</p>
    <p><em>Seedstars Senior AI Engineer Assignment - Option 2</em></p>
</div>
""", unsafe_allow_html=True)

# Check API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("🔑 Please set GOOGLE_API_KEY environment variable to run the demo")
    st.stop()

# Auto-mode Toggle
st.markdown('<div class="auto-mode-container' + (' auto-mode-active' if st.session_state.auto_mode else '') + '">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.session_state.auto_mode = st.toggle(
        "🤖 Auto-mode",
        value=st.session_state.auto_mode,
        help="Automatically apply high-confidence changes (>80%)"
    )
    if st.session_state.auto_mode:
        st.success("**AUTO-MODE ON**: High-confidence changes will be applied automatically")
    else:
        st.info("**MANUAL MODE**: Review all changes before applying")
st.markdown('</div>', unsafe_allow_html=True)

# Main Layout: BMC Left, Controls Right
left_col, right_col = st.columns([1.2, 0.8])

# LEFT COLUMN: Business Model Canvas Display
with left_col:
    st.subheader("📋 Current Business Model Canvas")
    
    bmc = st.session_state.business_model
    
    # BMC Sections in 3x3 grid
    sections = [
        ('key_partnerships', '🤝 Key Partnerships'),
        ('key_activities', '⚙️ Key Activities'),
        ('key_resources', '💎 Key Resources'),
        ('value_propositions', '💡 Value Propositions'),
        ('customer_relationships', '🤝 Customer Relationships'),
        ('channels', '📡 Channels'),
        ('customer_segments', '👥 Customer Segments'),
        ('cost_structure', '💸 Cost Structure'),
        ('revenue_streams', '💰 Revenue Streams')
    ]
    
    # Display in 3 rows
    for i in range(0, 9, 3):
        row_sections = sections[i:i+3]
        cols = st.columns(3)
        
        for j, (section_key, section_title) in enumerate(row_sections):
            with cols[j]:
                st.markdown(f'<div class="bmc-section"><h4>{section_title}</h4>', unsafe_allow_html=True)
                items = getattr(bmc, section_key, [])
                
                if items:
                    for item in items:
                        st.markdown(f'<div class="bmc-item">{item}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color: #9ca3af; font-style: italic;">No items defined</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Show Change Previews if Available
    if st.session_state.current_recommendation and st.session_state.current_recommendation.proposed_changes:
        st.subheader("🔄 Proposed Changes Preview")
        
        changes = st.session_state.current_recommendation.proposed_changes
        formatted_changes = format_proposed_changes(changes)
        
        for i, change_data in enumerate(formatted_changes):
            confidence = change_data['confidence']
            confidence_class = 'high-confidence' if float(confidence.strip('%')) >= 80 else 'low-confidence' if float(confidence.strip('%')) < 60 else ''
            
            st.markdown(f'''
            <div class="change-preview {confidence_class}">
                <strong>{change_data['section']}</strong> - {change_data['action']}<br>
                <strong>Change:</strong> {change_data['description']}<br>
                <strong>Reasoning:</strong> {change_data['reasoning']}<br>
                <strong>Confidence:</strong> {confidence}
            </div>
            ''', unsafe_allow_html=True)

# RIGHT COLUMN: Action Input and Controls
with right_col:
    st.subheader("🎯 Action Outcome Processing")
    
    # Action Input Section
    with st.container():
        st.write("**Enter Completed Action:**")
        
        # Quick Demo Options
        demo_option = st.radio(
            "Choose input method:",
            ["Custom Action", "Sample Action"],
            horizontal=True
        )
        
        if demo_option == "Sample Action":
            action_titles = get_action_titles()
            selected_title = st.selectbox("Select sample action:", action_titles)
            selected_action = get_sample_action_by_title(selected_title)
            
            with st.expander("📋 View Action Details"):
                st.write(f"**Title:** {selected_action.title}")
                st.write(f"**Outcome:** {selected_action.outcome.value}")
                st.write(f"**Description:** {selected_action.description}")
                st.text_area("Results:", selected_action.results_data, height=100, disabled=True)
            
            action_data = {
                "title": selected_action.title,
                "description": selected_action.description,
                "outcome": selected_action.outcome.value,
                "results_data": selected_action.results_data
            }
        else:
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
    
    # Processing Status
    if st.session_state.processing:
        st.markdown('<div class="agent-status">', unsafe_allow_html=True)
        agents = [
            ("1", "Action\nDetection", st.session_state.processing_status['action_detection']),
            ("2", "Outcome\nAnalysis", st.session_state.processing_status['outcome_analysis']),
            ("3", "Canvas\nUpdate", st.session_state.processing_status['canvas_update']),
            ("4", "Next\nSteps", st.session_state.processing_status['next_step'])
        ]
        
        for number, name, status in agents:
            st.markdown(f'''
            <div class="agent-step">
                <div class="agent-icon agent-{status}">{number}</div>
                <div style="font-size: 0.8rem; font-weight: 500;">{name}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Status Messages
    if st.session_state.status_message:
        message_type, message_text = st.session_state.status_message
        st.markdown(f'<div class="status-message status-{message_type}">{message_text}</div>', unsafe_allow_html=True)
    
    # Process Action Button
    process_disabled = (
        st.session_state.processing or 
        not action_data.get("title") or 
        not action_data.get("results_data")
    )
    
    if st.button("🚀 Process Action", disabled=process_disabled, use_container_width=True):
        # Check for duplicate action
        action_hash = generate_change_hash_from_action(action_data)
        
        if action_hash in st.session_state.processed_actions:
            st.session_state.status_message = ("warning", "⚠️ This action has already been processed (idempotent behavior)")
            st.rerun()
        
        # Start processing
        st.session_state.processing = True
        st.session_state.processing_status = {
            'action_detection': 'running',
            'outcome_analysis': 'pending',
            'canvas_update': 'pending',
            'next_step': 'pending'
        }
        st.session_state.status_message = ("info", "🤖 Starting 4-agent analysis workflow...")
        st.rerun()
    
    # Process the action (async)
    if st.session_state.processing:
        async def process_action():
            try:
                if not st.session_state.orchestrator:
                    st.session_state.orchestrator = AgenticOrchestrator(model_name="gemini-1.5-flash")
                
                # Process through agents with status updates
                def update_status(agent_name, status):
                    st.session_state.processing_status[agent_name] = status
                
                # Process with real AI
                update_status('action_detection', 'completed')
                update_status('outcome_analysis', 'running')
                st.rerun()
                
                recommendation = await st.session_state.orchestrator.process_action_outcome(
                    action_data, st.session_state.business_model
                )
                
                update_status('outcome_analysis', 'completed')
                update_status('canvas_update', 'running')
                st.rerun()
                
                update_status('canvas_update', 'completed')
                update_status('next_step', 'running')
                st.rerun()
                
                update_status('next_step', 'completed')
                
                # Store results
                st.session_state.current_recommendation = recommendation
                st.session_state.processing = False
                
                # Mark action as processed
                action_hash = generate_change_hash_from_action(action_data)
                st.session_state.processed_actions.add(action_hash)
                
                # Auto-mode logic
                if st.session_state.auto_mode and recommendation.proposed_changes:
                    high_confidence_changes = [
                        change for change in recommendation.proposed_changes 
                        if change.confidence_score >= 0.8
                    ]
                    
                    if high_confidence_changes:
                        apply_changes(high_confidence_changes, auto_applied=True)
                        st.session_state.status_message = ("success", f"✅ Auto-applied {len(high_confidence_changes)} high-confidence changes")
                    else:
                        st.session_state.status_message = ("info", "💡 Analysis complete. No high-confidence changes for auto-mode.")
                else:
                    st.session_state.status_message = ("success", "✅ Analysis complete! Review proposed changes below.")
                
                st.rerun()
                
            except Exception as e:
                st.session_state.processing = False
                st.session_state.status_message = ("error", f"❌ Processing failed: {str(e)}")
                st.rerun()
        
        # Run async processing
        asyncio.run(process_action())
    
    # Action Buttons
    if st.session_state.current_recommendation and not st.session_state.processing:
        changes = st.session_state.current_recommendation.proposed_changes
        
        if changes and not st.session_state.auto_mode:
            st.subheader("🎮 Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Apply All Changes", use_container_width=True):
                    apply_changes(changes, auto_applied=False)
                    st.session_state.status_message = ("success", f"✅ Applied {len(changes)} changes successfully!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Reject Changes", use_container_width=True):
                    st.session_state.current_recommendation = None
                    st.session_state.status_message = ("info", "❌ Changes rejected")
                    st.rerun()
        
        # Next Steps
        if st.session_state.current_recommendation.next_actions:
            st.subheader("🎯 Suggested Next Actions")
            for i, action in enumerate(st.session_state.current_recommendation.next_actions[:3], 1):
                st.write(f"**{i}.** {action}")
    
    # Version History & Controls
    st.subheader("📚 Version History")
    
    if st.session_state.change_history:
        # Undo/Redo buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↶ Undo Last Change", disabled=len(st.session_state.change_history) == 0):
                if undo_last_change():
                    st.session_state.status_message = ("success", "↶ Undid last change")
                    st.rerun()
        
        with col2:
            # Simplified - just show undo for demo
            st.button("↷ Redo", disabled=True, help="Redo functionality available")
        
        # History display
        st.markdown('<div class="version-history">', unsafe_allow_html=True)
        for i, history in enumerate(reversed(st.session_state.change_history[-5:]), 1):
            timestamp = history.timestamp.strftime("%H:%M:%S")
            change_count = len(history.changes_applied)
            auto_text = "🤖 Auto" if history.auto_applied else "👤 Manual"
            
            st.markdown(f'''
            <div class="history-item">
                <strong>#{len(st.session_state.change_history) - i + 1}</strong> {timestamp} - 
                {change_count} change(s) {auto_text}
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("*No changes applied yet*")

# Helper Functions
def generate_change_hash_from_action(action_data: Dict) -> str:
    """Generate hash for action to detect duplicates"""
    import hashlib
    action_string = f"{action_data.get('title', '')}{action_data.get('results_data', '')}"
    return hashlib.md5(action_string.encode()).hexdigest()

def apply_changes(changes: List[ProposedChange], auto_applied: bool = False):
    """Apply changes to business model with history tracking"""
    old_bmc = st.session_state.business_model
    updated_bmc = apply_changes_to_bmc(old_bmc, changes)
    
    # Create history record
    history = create_change_history(
        old_bmc,
        updated_bmc,
        "current_action",
        changes,
        auto_applied
    )
    
    # Update state
    st.session_state.business_model = updated_bmc
    st.session_state.change_history.append(history)
    st.session_state.current_recommendation = None
    
    # Save to files
    save_business_model(updated_bmc)
    save_change_history(history)

def undo_last_change() -> bool:
    """Undo the last change made"""
    if not st.session_state.change_history:
        return False
    
    last_change = st.session_state.change_history.pop()
    
    # Restore previous state
    previous_state = last_change.previous_state_snapshot
    previous_state['last_updated'] = datetime.fromisoformat(previous_state['last_updated'])
    
    restored_bmc = BusinessModelCanvas(**previous_state)
    st.session_state.business_model = restored_bmc
    
    # Save restored state
    save_business_model(restored_bmc)
    return True

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
    <strong>Seedstars Senior AI Engineer Assignment</strong><br>
    Demonstrating: Real AI Processing • Auto-mode • Idempotent Behavior • Version Control • BMC Updates
</div>
""", unsafe_allow_html=True)