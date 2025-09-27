"""
Reusable UI Components for SIGMA Agentic AI Actions Co-pilot
Enhanced with Strategic Next Steps Display
"""

import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from .bmc_canvas import BusinessModelCanvas


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


def display_quality_indicator(quality):
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


def display_priority_badge(priority: str):
    """Display priority badge with appropriate color"""
    if priority.lower() == "high":
        st.markdown(
            '<span style="background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">🔥 HIGH</span>',
            unsafe_allow_html=True
        )
    elif priority.lower() == "medium":
        st.markdown(
            '<span style="background-color: #ffa500; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">⚡ MEDIUM</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span style="background-color: #00cc00; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">✅ LOW</span>',
            unsafe_allow_html=True
        )


def display_difficulty_indicator(difficulty: str):
    """Display difficulty indicator"""
    if difficulty.lower() == "hard":
        st.markdown("🔴 Hard")
    elif difficulty.lower() == "medium":
        st.markdown("🟡 Medium")
    else:
        st.markdown("🟢 Easy")


def display_enhanced_next_steps(next_steps: List[Dict[str, Any]]):
    """Display enhanced next steps with rich formatting and implementation guidance"""
    if not next_steps:
        st.info("No specific next steps provided.")
        return
    
    st.subheader("🎯 Strategic Next Steps")
    st.caption("AI-generated action plan based on your experiment outcome")
    
    # Sort by priority (high first)
    priority_order = {"high": 1, "medium": 2, "low": 3}
    sorted_steps = sorted(next_steps, key=lambda x: priority_order.get(x.get('priority', 'medium').lower(), 2))
    
    for i, step in enumerate(sorted_steps, 1):
        # Handle both dict and string formats
        if isinstance(step, str):
            st.write(f"**{i}.** {step}")
            continue
        
        # Rich card layout for detailed next steps
        with st.container():
            # Create card-like appearance
            st.markdown(f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 8px 0; background-color: #fafafa;">
            </div>
            """, unsafe_allow_html=True)
            
            # Header with title and badges
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {i}. {step.get('title', 'Next Step')}")
            
            with col2:
                display_priority_badge(step.get('priority', 'medium'))
            
            with col3:
                display_difficulty_indicator(step.get('difficulty', 'medium'))
            
            # Description
            description = step.get('description', '')
            if description:
                st.write(description)
            
            # Key details in columns
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                timeline = step.get('timeline', 'TBD')
                st.markdown(f"**⏱️ Timeline:** {timeline}")
                
                stage = step.get('stage', 'validation')
                st.markdown(f"**🎭 Stage:** {stage.title()}")
            
            with detail_col2:
                resources = step.get('resources_needed', [])
                if resources:
                    st.markdown("**💰 Resources:**")
                    for resource in resources[:3]:  # Show max 3
                        st.markdown(f"• {resource}")
                    if len(resources) > 3:
                        st.markdown(f"• +{len(resources) - 3} more...")
            
            with detail_col3:
                metrics = step.get('success_metrics', [])
                if metrics:
                    st.markdown("**📊 Success Metrics:**")
                    for metric in metrics[:3]:  # Show max 3
                        st.markdown(f"• {metric}")
                    if len(metrics) > 3:
                        st.markdown(f"• +{len(metrics) - 3} more...")
            
            # Implementation steps in expandable section
            implementation_steps = step.get('implementation_steps', [])
            if implementation_steps:
                with st.expander(f"📋 Implementation Guide for Step {i}", expanded=False):
                    st.markdown("**Step-by-step implementation:**")
                    for j, impl_step in enumerate(implementation_steps, 1):
                        st.markdown(f"**{j}.** {impl_step}")
                    
                    # Additional resources if available
                    if len(resources) > 3:
                        st.markdown("**Additional Resources Needed:**")
                        for resource in resources[3:]:
                            st.markdown(f"• {resource}")
                    
                    if len(metrics) > 3:
                        st.markdown("**Additional Success Metrics:**")
                        for metric in metrics[3:]:
                            st.markdown(f"• {metric}")
            
            st.markdown("---")


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
    
    st.subheader("🔄 Preview Changes")
    st.caption("See how your Business Model will look after applying these changes")
    
    for i, change in enumerate(changes):
        section_display = bmc.get_section_display_name(change['section'])
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


def render_business_canvas(bmc: BusinessModelCanvas):
    """Render the business model canvas in a clean layout"""
    st.subheader("Your Current Business Model")
    
    # Show business stage and completion
    stage = bmc.get_business_stage()
    completion = bmc.get_completeness_score()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"Stage: **{stage.title()}** | Completion: **{completion:.0%}**")
    with col2:
        st.progress(completion)
    
    sections = bmc.get_all_sections()
    
    # Display in 2x2 grid
    col1, col2 = st.columns(2)
    
    section_items = list(sections.items())
    
    for i in range(0, len(section_items), 2):
        with col1 if i % 4 == 0 else col2:
            # First section
            section_key, items = section_items[i]
            display_name = bmc.get_section_display_name(section_key)
            
            st.markdown(f"**{display_name}**")
            if items:
                for j, item in enumerate(items, 1):
                    st.markdown(f"{j}. {item}")
            else:
                st.markdown("*No items defined*")
            st.markdown("")
            
            # Second section (if exists)
            if i + 1 < len(section_items):
                section_key, items = section_items[i + 1]
                display_name = bmc.get_section_display_name(section_key)
                
                st.markdown(f"**{display_name}**")
                if items:
                    for j, item in enumerate(items, 1):
                        st.markdown(f"{j}. {item}")
                else:
                    st.markdown("*No items defined*")
                st.markdown("")


def render_session_metrics(metrics: Dict[str, Any], quality_data: Dict[str, Any]):
    """Render session metrics in footer"""
    st.markdown("---")
    
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


def render_sidebar_info(metrics, bmc: BusinessModelCanvas):
    """Render sidebar with session info and business summary"""
    with st.sidebar:
        st.subheader("Session Info")
        
        # Session metrics
        st.write(f"Session ID: `{metrics['session_id']}`")
        st.write(f"Actions Analyzed: {metrics['actions_analyzed']}")
        st.write(f"Changes Applied: {metrics['changes_applied']}")
        st.write(f"Duration: {metrics['duration_seconds']:.0f}s")
        
        # Business summary if complete
        if bmc.is_complete():
            st.markdown("---")
            st.subheader("Your Business")
            
            # Business stage indicator
            stage = bmc.get_business_stage()
            risk_assessment = bmc.get_risk_assessment()
            
            if stage == "validation":
                st.markdown("🔬 **Stage:** Discovery/Validation")
            elif stage == "growth":
                st.markdown("📈 **Stage:** Growth")
            else:
                st.markdown("🚀 **Stage:** Scale")
            
            st.caption(bmc.get_business_summary())
            
            # Risk indicator
            if "high" in risk_assessment.lower():
                st.warning(f"⚠️ {risk_assessment}")
            else:
                st.info(f"✅ {risk_assessment}")
            
            # Completion score
            completion = bmc.get_completeness_score()
            st.progress(completion)
            st.caption(f"Business Design: {completion:.0%} complete")


def render_action_form(sample_actions: Dict[str, Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Render the action input form with persistent state management"""
    st.subheader("Log Completed Action")
    
    # Initialize session state for action management
    if 'current_action_data' not in st.session_state:
        st.session_state.current_action_data = None
    if 'current_action_type' not in st.session_state:
        st.session_state.current_action_type = None
    if 'action_selection_changed' not in st.session_state:
        st.session_state.action_selection_changed = False
    
    # Sample action selector
    use_sample = st.selectbox(
        "Choose sample action or create custom:",
        ["Custom Action"] + list(sample_actions.keys()),
        key="action_selector"
    )
    
    # Check if selection changed - clear stored data if so
    if 'previous_selection' not in st.session_state:
        st.session_state.previous_selection = use_sample
    elif st.session_state.previous_selection != use_sample:
        st.session_state.current_action_data = None
        st.session_state.current_action_type = None
        st.session_state.previous_selection = use_sample
        st.session_state.action_selection_changed = True
    
    if use_sample != "Custom Action":
        # Use selected sample action
        action_data = sample_actions[use_sample]
        
        st.info(f"**Sample Action Selected:** {action_data['title']}")
        
        with st.expander("View Action Details", expanded=False):
            st.write(f"**Outcome:** {action_data['outcome']}")
            st.write(f"**Description:** {action_data['description']}")
            st.write("**Results:**")
            st.code(action_data['results'])
        
        # Store in session state
        st.session_state.current_action_data = action_data
        st.session_state.current_action_type = "sample"
        
        return action_data, "sample"
    
    else:
        # Custom action form
        st.write("**Create Custom Action:**")
        
        # Check if we have stored custom action data
        if (st.session_state.current_action_data is not None and 
            st.session_state.current_action_type == "custom" and 
            not st.session_state.action_selection_changed):
            
            # Display stored custom action
            stored_data = st.session_state.current_action_data
            st.success("✅ **Custom Action Ready for Analysis:**")
            st.write(f"**Title:** {stored_data['title']}")
            st.write(f"**Outcome:** {stored_data['outcome']}")
            st.write(f"**Description:** {stored_data['description']}")
            st.write(f"**Results:** {stored_data['results'][:100]}...")
            
            # Option to edit or use current
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit Custom Action", use_container_width=True):
                    st.session_state.current_action_data = None
                    st.session_state.current_action_type = None
                    st.rerun()
            with col2:
                st.write("*Ready for analysis below* ⬇️")
            
            return stored_data, "custom"
        
        else:
            # Reset the changed flag
            st.session_state.action_selection_changed = False
            
            # Show form for new custom action
            with st.form("custom_action_form", clear_on_submit=False):
                
                action_data = {
                    "title": st.text_input(
                        "Action/Experiment Title", 
                        placeholder="e.g., Customer interviews with target segment"
                    ),
                    "outcome": st.selectbox("Outcome", ["Successful", "Failed", "Inconclusive"]),
                    "description": st.text_area(
                        "What did you do?", 
                        placeholder="Describe the action/experiment you completed",
                        height=100
                    ),
                    "results": st.text_area(
                        "Results & Key Learnings", 
                        placeholder="What did you learn? Include metrics, feedback, insights...",
                        height=150
                    )
                }
                
                form_submitted = st.form_submit_button("Save Custom Action", use_container_width=True)
                
                if form_submitted:
                    if not all([action_data["title"], action_data["description"], action_data["results"]]):
                        st.error("Please fill in all fields for custom action")
                        return None, None
                    else:
                        # Store in session state
                        st.session_state.current_action_data = action_data
                        st.session_state.current_action_type = "custom"
                        st.success("✅ Custom action saved! You can now analyze it below.")
                        st.rerun()
            
            return None, None


def get_current_action() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Get the currently selected/stored action data"""
    if hasattr(st.session_state, 'current_action_data') and st.session_state.current_action_data:
        return st.session_state.current_action_data, st.session_state.current_action_type
    return None, None


def clear_current_action():
    """Clear the currently stored action data"""
    if 'current_action_data' in st.session_state:
        st.session_state.current_action_data = None
    if 'current_action_type' in st.session_state:
        st.session_state.current_action_type = None


def render_header():
    """Render the main application header"""
    st.title("SIGMA Agentic AI Actions Co-pilot")
    st.markdown("**Seedstars Assignment**: Complete experiments → AI updates your business model → Get next steps")


def render_footer():
    """Render the application footer"""
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 40px;">
        <strong>SIGMA Agentic AI Actions Co-pilot</strong> | 
        Seedstars Senior AI Engineer Assignment | 
        Enhanced: Strategic Next Steps, Business Intelligence, Quality Validation
    </div>
    """, unsafe_allow_html=True)