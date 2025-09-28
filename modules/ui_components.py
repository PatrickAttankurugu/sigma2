"""
Reusable UI Components for SIGMA Actions Co-pilot
"""

import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from .bmc_canvas import BusinessModelCanvas


def display_confidence_indicator(confidence: float, label: str = "Confidence"):
    """Display visual confidence indicator with progress bar"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
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
            '<span style="background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">HIGH</span>',
            unsafe_allow_html=True
        )
    elif priority.lower() == "medium":
        st.markdown(
            '<span style="background-color: #ffa500; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">MEDIUM</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span style="background-color: #00cc00; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">LOW</span>',
            unsafe_allow_html=True
        )


def display_difficulty_indicator(difficulty: str):
    """Display difficulty indicator"""
    if difficulty.lower() == "hard":
        st.markdown("Hard")
    elif difficulty.lower() == "medium":
        st.markdown("Medium")
    else:
        st.markdown("Easy")


def display_enhanced_next_steps(next_steps: List[Dict[str, Any]]):
    """Display enhanced next steps with rich formatting"""
    if not next_steps:
        st.info("No specific next steps provided.")
        return
    
    if isinstance(next_steps[0], str):
        st.subheader("Suggested Next Experiments")
        for i, step in enumerate(next_steps, 1):
            st.write(f"**{i}.** {step}")
        return
    
    st.subheader("Strategic Next Steps")
    st.caption("AI-generated action plan based on your experiment outcome")
    
    priority_order = {"high": 1, "medium": 2, "low": 3}
    try:
        sorted_steps = sorted(next_steps, key=lambda x: priority_order.get(x.get('priority', 'medium').lower(), 2))
    except (KeyError, TypeError):
        sorted_steps = next_steps
    
    for i, step in enumerate(sorted_steps, 1):
        if isinstance(step, str):
            st.write(f"**{i}.** {step}")
            continue
        
        with st.container():
            st.markdown("""
            <div style="border-left: 4px solid #1f77b4; padding-left: 12px; margin: 16px 0;">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {i}. {step.get('title', 'Next Step')}")
            
            with col2:
                if step.get('priority'):
                    display_priority_badge(step.get('priority', 'medium'))
            
            with col3:
                if step.get('difficulty'):
                    display_difficulty_indicator(step.get('difficulty', 'medium'))
            
            description = step.get('description', '')
            if description:
                st.write(description)
            
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                timeline = step.get('timeline', 'TBD')
                st.markdown(f"**Timeline:** {timeline}")
                
                stage = step.get('stage', 'validation')
                st.markdown(f"**Stage:** {stage.title()}")
            
            with detail_col2:
                resources = step.get('resources_needed', [])
                if resources:
                    st.markdown("**Resources:**")
                    for resource in resources[:3]:
                        st.markdown(f"• {resource}")
                    if len(resources) > 3:
                        st.markdown(f"• +{len(resources) - 3} more...")
            
            with detail_col3:
                metrics = step.get('success_metrics', [])
                if metrics:
                    st.markdown("**Success Metrics:**")
                    for metric in metrics[:3]:
                        st.markdown(f"• {metric}")
                    if len(metrics) > 3:
                        st.markdown(f"• +{len(metrics) - 3} more...")
            
            implementation_steps = step.get('implementation_steps', [])
            if implementation_steps:
                with st.expander(f"Implementation Guide for Step {i}", expanded=False):
                    st.markdown("**Step-by-step implementation:**")
                    for j, impl_step in enumerate(implementation_steps, 1):
                        st.markdown(f"**{j}.** {impl_step}")
                    
                    if len(resources) > 3:
                        st.markdown("**Additional Resources Needed:**")
                        for resource in resources[3:]:
                            st.markdown(f"• {resource}")
                    
                    if len(metrics) > 3:
                        st.markdown("**Additional Success Metrics:**")
                        for metric in metrics[3:]:
                            st.markdown(f"• {metric}")
            
            st.markdown("</div>", unsafe_allow_html=True)
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
    
    st.subheader("Preview Changes")
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
    
    stage = bmc.get_business_stage()
    completion = bmc.get_completeness_score()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"Stage: **{stage.title()}** | Completion: **{completion:.0%}**")
    with col2:
        st.progress(completion)
    
    sections = bmc.get_all_sections()
    
    col1, col2 = st.columns(2)
    
    section_items = list(sections.items())
    
    for i in range(0, len(section_items), 2):
        with col1 if i % 4 == 0 else col2:
            section_key, items = section_items[i]
            display_name = bmc.get_section_display_name(section_key)
            
            st.markdown(f"**{display_name}**")
            if items:
                for j, item in enumerate(items, 1):
                    st.markdown(f"{j}. {item}")
            else:
                st.markdown("*No items defined*")
            st.markdown("")
            
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
        
        st.write(f"Session ID: `{metrics['session_id']}`")
        st.write(f"Actions Analyzed: {metrics['actions_analyzed']}")
        st.write(f"Changes Applied: {metrics['changes_applied']}")
        st.write(f"Duration: {metrics['duration_seconds']:.0f}s")
        
        if bmc.is_complete():
            st.markdown("---")
            st.subheader("Your Business")
            
            stage = bmc.get_business_stage()
            risk_assessment = bmc.get_risk_assessment()
            
            if stage == "validation":
                st.markdown("**Stage:** Discovery/Validation")
            elif stage == "growth":
                st.markdown("**Stage:** Growth")
            else:
                st.markdown("**Stage:** Scale")
            
            st.caption(bmc.get_business_summary())
            
            if "high" in risk_assessment.lower():
                st.warning(f"{risk_assessment}")
            else:
                st.info(f"{risk_assessment}")
            
            completion = bmc.get_completeness_score()
            st.progress(completion)
            st.caption(f"Business Design: {completion:.0%} complete")


def render_action_form(sample_actions: Dict[str, Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Render the action input form with persistent state management"""
    st.subheader("Log Completed Action")
    
    if 'current_action_data' not in st.session_state:
        st.session_state.current_action_data = None
    if 'current_action_type' not in st.session_state:
        st.session_state.current_action_type = None
    if 'action_selection_changed' not in st.session_state:
        st.session_state.action_selection_changed = False
    if 'action_edit_mode' not in st.session_state:
        st.session_state.action_edit_mode = False
    
    use_sample = st.selectbox(
        "Choose sample action or create custom:",
        ["Custom Action"] + list(sample_actions.keys()),
        key="action_selector"
    )
    
    if 'previous_selection' not in st.session_state:
        st.session_state.previous_selection = use_sample
    elif st.session_state.previous_selection != use_sample:
        st.session_state.current_action_data = None
        st.session_state.current_action_type = None
        st.session_state.previous_selection = use_sample
        st.session_state.action_selection_changed = True
        st.session_state.action_edit_mode = False
    
    if use_sample != "Custom Action":
        action_data = sample_actions[use_sample]
        
        st.info(f"**Sample Action Selected:** {action_data['title']}")
        
        with st.expander("View Action Details", expanded=False):
            st.write(f"**Outcome:** {action_data['outcome']}")
            st.write(f"**Description:** {action_data['description']}")
            st.write("**Results:**")
            st.code(action_data['results'])
        
        st.session_state.current_action_data = action_data
        st.session_state.current_action_type = "sample"
        st.session_state.action_edit_mode = False
        
        return action_data, "sample"
    
    else:
        st.write("**Create Custom Action:**")
        
        if (st.session_state.current_action_data is not None and 
            st.session_state.current_action_type == "custom" and 
            not st.session_state.action_selection_changed and
            not st.session_state.action_edit_mode):
            
            stored_data = st.session_state.current_action_data
            st.success("Custom Action Ready for Analysis:")
            st.write(f"**Title:** {stored_data['title']}")
            st.write(f"**Outcome:** {stored_data['outcome']}")
            st.write(f"**Description:** {stored_data['description']}")
            st.write(f"**Results:** {stored_data['results'][:100]}...")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit Custom Action", use_container_width=True):
                    st.session_state.action_edit_mode = True
                    st.rerun()
            with col2:
                st.write("*Ready for analysis below*")
            
            return stored_data, "custom"
        
        else:
            st.session_state.action_selection_changed = False
            
            existing_data = None
            if st.session_state.action_edit_mode and st.session_state.current_action_data:
                existing_data = st.session_state.current_action_data
            
            form_title = "Edit Custom Action:" if st.session_state.action_edit_mode else "Create Custom Action:"
            st.write(f"**{form_title}**")
            
            with st.form("custom_action_form", clear_on_submit=False):
                
                action_data = {
                    "title": st.text_input(
                        "Action/Experiment Title", 
                        value=existing_data.get('title', '') if existing_data else '',
                        placeholder="e.g., Customer interviews with target segment"
                    ),
                    "outcome": st.selectbox(
                        "Outcome", 
                        ["Successful", "Failed", "Inconclusive"],
                        index=["Successful", "Failed", "Inconclusive"].index(existing_data.get('outcome', 'Successful')) if existing_data else 0
                    ),
                    "description": st.text_area(
                        "What did you do?", 
                        value=existing_data.get('description', '') if existing_data else '',
                        placeholder="Describe the action/experiment you completed",
                        height=100
                    ),
                    "results": st.text_area(
                        "Results & Key Learnings", 
                        value=existing_data.get('results', '') if existing_data else '',
                        placeholder="What did you learn? Include metrics, feedback, insights...",
                        height=150
                    )
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    save_button_text = "Update Custom Action" if st.session_state.action_edit_mode else "Save Custom Action"
                    form_submitted = st.form_submit_button(save_button_text, use_container_width=True)
                
                with col2:
                    cancel_clicked = False
                    if st.session_state.action_edit_mode:
                        cancel_clicked = st.form_submit_button("Cancel Edit", use_container_width=True)
                
                if cancel_clicked:
                    st.session_state.action_edit_mode = False
                    st.rerun()
                
                if form_submitted:
                    if not all([action_data["title"], action_data["description"], action_data["results"]]):
                        st.error("Please fill in all fields for custom action")
                        return None, None
                    else:
                        st.session_state.current_action_data = action_data
                        st.session_state.current_action_type = "custom"
                        st.session_state.action_edit_mode = False
                        
                        success_message = "Custom action updated! You can now analyze it below." if existing_data else "Custom action saved! You can now analyze it below."
                        st.success(success_message)
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
    if 'action_edit_mode' in st.session_state:
        st.session_state.action_edit_mode = False


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