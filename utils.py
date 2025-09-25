"""
Enhanced utility functions for the Agentic AI Actions Co-pilot system.

This module provides comprehensive data management, UI helpers, business logic utilities,
and professional formatting functions for the business model canvas management system.
"""

import json
import os
import pickle
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import re

import pandas as pd
import streamlit as st

from business_models import (
    BusinessModelCanvas,
    ProposedChange,
    ChangeHistory,
    ChangeType,
    ConfidenceLevel,
    AgentRecommendation,
    CompletedAction,
    ActionOutcome
)
from mock_data import get_sample_business_model_canvas


# Data Management Functions

def load_business_model(file_path: str = "business_model.json") -> BusinessModelCanvas:
    """
    Load current business model canvas state from file.

    Args:
        file_path: Path to the BMC JSON file

    Returns:
        BusinessModelCanvas object
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle datetime parsing for loaded data
                if 'last_updated' in data and isinstance(data['last_updated'], str):
                    data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                return BusinessModelCanvas(**data)
        else:
            # Return sample BMC if no file exists
            return get_sample_business_model_canvas()
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Warning: Could not load business model from {file_path}: {e}")
        return get_sample_business_model_canvas()


def save_business_model(bmc: BusinessModelCanvas, file_path: str = "business_model.json") -> bool:
    """
    Save business model canvas to file.

    Args:
        bmc: BusinessModelCanvas to save
        file_path: Path where to save the BMC

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to dict and handle datetime serialization
        bmc_dict = bmc.dict()
        bmc_dict['last_updated'] = bmc.last_updated.isoformat()

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(bmc_dict, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving business model to {file_path}: {e}")
        return False


def create_change_history(
    old_state: BusinessModelCanvas,
    new_state: BusinessModelCanvas,
    trigger_action_id: str,
    applied_changes: List[ProposedChange],
    auto_applied: bool = False
) -> ChangeHistory:
    """
    Create a change history record for version tracking.

    Args:
        old_state: BMC state before changes
        new_state: BMC state after changes
        trigger_action_id: ID of action that triggered changes
        applied_changes: List of changes that were applied
        auto_applied: Whether changes were applied automatically

    Returns:
        ChangeHistory object
    """
    # Create snapshots with datetime handling
    old_snapshot = old_state.dict()
    old_snapshot['last_updated'] = old_state.last_updated.isoformat()
    
    new_snapshot = new_state.dict()
    new_snapshot['last_updated'] = new_state.last_updated.isoformat()

    return ChangeHistory(
        trigger_action_id=trigger_action_id,
        changes_applied=applied_changes,
        previous_state_snapshot=old_snapshot,
        new_state_snapshot=new_snapshot,
        auto_applied=auto_applied
    )


def save_change_history(history: ChangeHistory, history_file: str = "change_history.json") -> bool:
    """
    Save change history to file for audit trail.

    Args:
        history: ChangeHistory to save
        history_file: File to save history records

    Returns:
        True if successful
    """
    try:
        # Load existing history
        existing_history = []
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                existing_history = json.load(f)

        # Add new history record
        history_dict = history.dict()
        history_dict['timestamp'] = history.timestamp.isoformat()
        
        # Handle ProposedChange serialization
        changes_list = []
        for change in history.changes_applied:
            change_dict = change.dict()
            changes_list.append(change_dict)
        history_dict['changes_applied'] = changes_list
        
        existing_history.append(history_dict)

        # Keep only last 100 records to prevent file from growing too large
        if len(existing_history) > 100:
            existing_history = existing_history[-100:]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error saving change history: {e}")
        return False


def load_change_history(history_file: str = "change_history.json") -> List[ChangeHistory]:
    """
    Load change history from file.

    Args:
        history_file: File containing history records

    Returns:
        List of ChangeHistory objects
    """
    try:
        if not os.path.exists(history_file):
            return []

        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)

        # Convert back to ChangeHistory objects
        history_objects = []
        for record in history_data:
            try:
                # Convert ISO timestamp back to datetime
                record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                
                # Convert changes back to ProposedChange objects
                changes = []
                for change_data in record.get('changes_applied', []):
                    change = ProposedChange(**change_data)
                    changes.append(change)
                record['changes_applied'] = changes
                
                history_objects.append(ChangeHistory(**record))
            except (ValueError, KeyError, TypeError) as e:
                print(f"Skipping invalid history record: {e}")
                continue

        return history_objects
    except Exception as e:
        print(f"Error loading change history: {e}")
        return []


def validate_change_safety(change: ProposedChange) -> bool:
    """
    Determine if a change is safe for automatic application.

    Args:
        change: ProposedChange to validate

    Returns:
        True if safe for auto-application
    """
    # Safety criteria
    if change.confidence_score < 0.7:
        return False

    if change.change_type == ChangeType.REMOVE:
        return False  # Removals should always require manual approval

    # Check for critical sections that need manual review
    critical_sections = {'revenue_streams', 'cost_structure', 'key_partnerships'}
    if change.canvas_section in critical_sections and change.confidence_score < 0.85:
        return False

    # Validate proposed value is not empty or too generic
    if not change.proposed_value.strip():
        return False
        
    generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test'}
    if change.proposed_value.lower().strip() in generic_terms:
        return False

    return True


# UI Helper Functions

def format_proposed_changes(changes: List[ProposedChange]) -> List[Dict[str, Any]]:
    """
    Format proposed changes for human-readable display.

    Args:
        changes: List of ProposedChange objects

    Returns:
        List of formatted change dictionaries
    """
    formatted_changes = []

    for change in changes:
        formatted_change = {
            "section": change.canvas_section.replace("_", " ").title(),
            "action": change.change_type.value.title(),
            "description": _format_change_description(change),
            "reasoning": change.reasoning,
            "confidence": f"{change.confidence_score:.0%}",
            "confidence_level": _get_confidence_category(change.confidence_score),
            "safety_level": "Safe" if validate_change_safety(change) else "Review Required",
            "impact_assessment": determine_change_impact(change, None)
        }
        formatted_changes.append(formatted_change)

    return formatted_changes


def _format_change_description(change: ProposedChange) -> str:
    """Format individual change description."""
    if change.change_type == ChangeType.ADD:
        return f"Add: {change.proposed_value}"
    elif change.change_type == ChangeType.MODIFY and change.current_value:
        return f"Change '{_truncate_text(change.current_value, 30)}' to '{_truncate_text(change.proposed_value, 30)}'"
    elif change.change_type == ChangeType.REMOVE and change.current_value:
        return f"Remove: {_truncate_text(change.current_value, 50)}"
    else:
        return _truncate_text(change.proposed_value, 50)


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def _get_confidence_category(score: float) -> str:
    """Get confidence category label."""
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    else:
        return "Low"


def create_before_after_comparison(
    old_bmc: BusinessModelCanvas,
    new_bmc: BusinessModelCanvas
) -> Dict[str, Dict[str, Any]]:
    """
    Create side-by-side comparison of BMC states.

    Args:
        old_bmc: Previous BMC state
        new_bmc: Updated BMC state

    Returns:
        Dictionary with before/after comparison for each section
    """
    comparison = {}

    sections = [
        'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
        'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
    ]

    for section in sections:
        old_values = getattr(old_bmc, section, [])
        new_values = getattr(new_bmc, section, [])

        # Calculate change metrics
        added_items = set(new_values) - set(old_values)
        removed_items = set(old_values) - set(new_values)
        unchanged_items = set(old_values) & set(new_values)

        comparison[section] = {
            "before": old_values,
            "after": new_values,
            "changed": old_values != new_values,
            "added_count": len(added_items),
            "removed_count": len(removed_items),
            "unchanged_count": len(unchanged_items),
            "change_summary": _generate_section_change_summary(added_items, removed_items)
        }

    return comparison


def _generate_section_change_summary(added: set, removed: set) -> str:
    """Generate a summary of changes for a section."""
    summary_parts = []
    
    if added:
        summary_parts.append(f"Added {len(added)} item(s)")
    
    if removed:
        summary_parts.append(f"Removed {len(removed)} item(s)")
    
    if not summary_parts:
        return "No changes"
    
    return ", ".join(summary_parts)


def generate_change_summary(applied_changes: List[ProposedChange]) -> str:
    """
    Generate a human-readable summary of applied changes.

    Args:
        applied_changes: List of changes that were applied

    Returns:
        Summary text for notifications
    """
    if not applied_changes:
        return "No changes were applied."

    change_count = len(applied_changes)
    sections_affected = len(set(change.canvas_section for change in applied_changes))

    summary = f"Applied {change_count} change(s) to {sections_affected} section(s) of your business model:\n"

    # Group changes by section
    by_section = {}
    for change in applied_changes:
        section = change.canvas_section.replace("_", " ").title()
        if section not in by_section:
            by_section[section] = []
        by_section[section].append(change.change_type.value.title())

    for section, actions in by_section.items():
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        action_summary = ", ".join(f"{count} {action.lower()}" if count > 1 else action.lower() 
                                  for action, count in action_counts.items())
        summary += f"• {section}: {action_summary}\n"

    return summary.strip()


# Business Logic Functions

def calculate_confidence_score(
    reasoning: str,
    data_quality: float = 0.5,
    sample_size: int = 0,
    validation_sources: int = 1,
    historical_data: Optional[List[ChangeHistory]] = None
) -> float:
    """
    Calculate confidence score based on multiple factors.

    Args:
        reasoning: AI reasoning for the change
        data_quality: Quality of underlying data (0.0-1.0)
        sample_size: Size of data sample
        validation_sources: Number of validation sources
        historical_data: Past change history for context

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Base score from reasoning quality
    reasoning_score = _assess_reasoning_quality(reasoning)
    
    # Data quality factor
    data_score = min(data_quality, 1.0)
    
    # Sample size factor (logarithmic scaling)
    if sample_size > 0:
        import math
        sample_score = min(math.log(sample_size + 1) / 10, 0.3)  # Cap at 0.3 contribution
    else:
        sample_score = 0.0
    
    # Validation sources factor
    validation_score = min(validation_sources / 5.0, 0.2)  # Cap at 0.2 contribution
    
    # Historical performance adjustment
    historical_score = 0.0
    if historical_data:
        historical_score = _assess_historical_performance(historical_data)
    
    # Weighted combination
    final_score = (
        reasoning_score * 0.4 +
        data_score * 0.3 +
        sample_score * 0.15 +
        validation_score * 0.1 +
        historical_score * 0.05
    )

    return max(0.0, min(1.0, final_score))


def _assess_reasoning_quality(reasoning: str) -> float:
    """Assess quality of AI reasoning text."""
    if not reasoning or len(reasoning) < 50:
        return 0.3

    score = 0.5  # Base score

    # Check for specific elements that indicate good reasoning
    quality_indicators = [
        'data shows', 'evidence suggests', 'analysis indicates',
        'market research', 'customer feedback', 'metrics demonstrate',
        'specifically', 'therefore', 'because', 'resulted in',
        'correlation', 'trend', 'pattern', 'significant'
    ]

    reasoning_lower = reasoning.lower()
    matched_indicators = sum(1 for indicator in quality_indicators if indicator in reasoning_lower)
    score += min(matched_indicators * 0.05, 0.3)  # Cap bonus at 0.3

    # Check for quantitative references
    quantitative_patterns = [
        r'\d+%', r'\d+\.\d+%', r'\d+ users?', r'\d+ customers?',
        r'\d+ responses?', r'n=\d+', r'\d+ months?', r'\d+ weeks?'
    ]
    
    quantitative_matches = sum(1 for pattern in quantitative_patterns if re.search(pattern, reasoning))
    score += min(quantitative_matches * 0.03, 0.15)  # Cap bonus at 0.15

    # Penalize vague language
    vague_indicators = [
        'might', 'could be', 'possibly', 'perhaps', 'maybe',
        'unclear', 'uncertain', 'ambiguous', 'seems like'
    ]

    vague_matches = sum(1 for indicator in vague_indicators if indicator in reasoning_lower)
    score -= min(vague_matches * 0.05, 0.2)  # Cap penalty at 0.2

    return max(0.1, min(1.0, score))


def _assess_historical_performance(historical_data: List[ChangeHistory]) -> float:
    """Assess performance based on historical change success."""
    if not historical_data:
        return 0.5

    # This is simplified - in practice you'd want to track actual outcomes
    recent_changes = historical_data[-10:]  # Last 10 changes
    
    # For now, return neutral score
    # In a real system, you'd track whether changes led to positive outcomes
    return 0.5


def determine_change_impact(change: ProposedChange, bmc: Optional[BusinessModelCanvas]) -> str:
    """
    Assess the significance of a proposed change.

    Args:
        change: The proposed change
        bmc: Current business model canvas (optional)

    Returns:
        Impact level: "low", "medium", or "high"
    """
    # High impact sections
    high_impact_sections = {'value_propositions', 'revenue_streams', 'customer_segments'}

    # High impact change types
    if change.change_type == ChangeType.REMOVE:
        return "high"

    # Section-based impact
    if change.canvas_section in high_impact_sections:
        if change.confidence_score < 0.7:
            return "high"
        else:
            return "medium"

    # Low confidence changes are always higher impact
    if change.confidence_score < 0.6:
        return "high"
    elif change.confidence_score < 0.8:
        return "medium"
    else:
        return "low"


def suggest_validation_experiments(
    changes: List[ProposedChange], 
    current_bmc: BusinessModelCanvas
) -> List[str]:
    """
    Suggest validation experiments based on proposed changes.

    Args:
        changes: List of proposed changes
        current_bmc: The current business model canvas

    Returns:
        List of suggested validation experiments
    """
    suggestions = []
    sections_changed = set(change.canvas_section for change in changes)

    # Customer segment validation
    if 'customer_segments' in sections_changed:
        suggestions.append(
            "Conduct customer interviews with new target segments to validate assumptions"
        )
        suggestions.append(
            "Run demographic analysis to confirm segment characteristics and size"
        )

    # Value proposition testing
    if 'value_propositions' in sections_changed:
        suggestions.append(
            "Design A/B tests for updated value propositions with existing customers"
        )
        suggestions.append(
            "Create landing page tests to measure value proposition resonance"
        )

    # Channel effectiveness
    if 'channels' in sections_changed:
        suggestions.append(
            "Test new channels with small pilot groups to measure effectiveness and cost"
        )
        suggestions.append(
            "Analyze channel attribution to understand customer acquisition paths"
        )

    # Revenue stream validation
    if 'revenue_streams' in sections_changed:
        suggestions.append(
            "Model financial projections for new revenue streams and test willingness to pay"
        )
        suggestions.append(
            "Conduct pricing sensitivity analysis with current customer base"
        )

    # Partnership validation
    if 'key_partnerships' in sections_changed:
        suggestions.append(
            "Reach out to potential partners to validate partnership opportunities and terms"
        )
        suggestions.append(
            "Assess partnership integration complexity and resource requirements"
        )

    # Generic suggestions if no specific matches
    if not suggestions:
        suggestions = [
            "Survey current customers to validate proposed business model changes",
            "Analyze competitor responses to similar strategic moves",
            "Create financial models to assess impact of proposed changes"
        ]

    return suggestions


def apply_changes_to_bmc(
    bmc: BusinessModelCanvas,
    changes: List[ProposedChange]
) -> BusinessModelCanvas:
    """
    Apply a list of proposed changes to a business model canvas.

    Args:
        bmc: Current business model canvas
        changes: List of changes to apply

    Returns:
        Updated business model canvas
    """
    # Create a copy to avoid modifying the original
    updated_bmc = BusinessModelCanvas(**bmc.dict())

    for change in changes:
        section_values = list(getattr(updated_bmc, change.canvas_section, []))

        if change.change_type == ChangeType.ADD:
            # Avoid duplicates
            if change.proposed_value.strip() and change.proposed_value not in section_values:
                section_values.append(change.proposed_value.strip())

        elif change.change_type == ChangeType.MODIFY and change.current_value:
            try:
                index = section_values.index(change.current_value)
                if change.proposed_value.strip():
                    section_values[index] = change.proposed_value.strip()
            except ValueError:
                # If current value not found, add the new value if it's valid
                if change.proposed_value.strip():
                    section_values.append(change.proposed_value.strip())

        elif change.change_type == ChangeType.REMOVE and change.current_value:
            try:
                section_values.remove(change.current_value)
            except ValueError:
                pass  # Value not found, ignore

        # Update the section with modified values
        setattr(updated_bmc, change.canvas_section, section_values)

    # Update timestamp
    updated_bmc.last_updated = datetime.now()
    return updated_bmc


# Analytics and Reporting Functions

def generate_change_analytics(history: List[ChangeHistory]) -> Dict[str, Any]:
    """
    Generate analytics from change history.

    Args:
        history: List of change history records

    Returns:
        Dictionary with analytics data
    """
    if not history:
        return {
            "total_changes": 0,
            "auto_applied_ratio": 0.0,
            "sections_most_changed": [],
            "change_frequency": {},
            "average_confidence": 0.0
        }

    total_changes = sum(len(record.changes_applied) for record in history)
    auto_applied = sum(1 for record in history if record.auto_applied)
    
    # Section analysis
    section_changes = {}
    all_confidences = []
    
    for record in history:
        for change in record.changes_applied:
            section = change.canvas_section
            section_changes[section] = section_changes.get(section, 0) + 1
            all_confidences.append(change.confidence_score)

    # Sort sections by change frequency
    sections_most_changed = sorted(
        section_changes.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]

    # Change frequency over time
    change_frequency = {}
    for record in history:
        date_key = record.timestamp.strftime("%Y-%m")
        change_frequency[date_key] = change_frequency.get(date_key, 0) + len(record.changes_applied)

    return {
        "total_changes": total_changes,
        "total_change_events": len(history),
        "auto_applied_ratio": auto_applied / len(history) if history else 0.0,
        "sections_most_changed": sections_most_changed,
        "change_frequency": change_frequency,
        "average_confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
        "confidence_distribution": _calculate_confidence_distribution(all_confidences)
    }


def _calculate_confidence_distribution(confidences: List[float]) -> Dict[str, int]:
    """Calculate distribution of confidence scores."""
    if not confidences:
        return {"high": 0, "medium": 0, "low": 0}
    
    high = sum(1 for c in confidences if c >= 0.8)
    medium = sum(1 for c in confidences if 0.6 <= c < 0.8)
    low = sum(1 for c in confidences if c < 0.6)
    
    return {"high": high, "medium": medium, "low": low}


def generate_business_model_health_score(bmc: BusinessModelCanvas) -> Dict[str, Any]:
    """
    Generate a health score for the business model canvas.

    Args:
        bmc: Business model canvas to evaluate

    Returns:
        Dictionary with health metrics
    """
    sections = [
        'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
        'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
    ]
    
    # Calculate completeness
    total_elements = 0
    empty_sections = 0
    section_scores = {}
    
    for section in sections:
        values = getattr(bmc, section, [])
        element_count = len(values)
        total_elements += element_count
        
        if element_count == 0:
            empty_sections += 1
            section_scores[section] = 0.0
        else:
            # Score based on number of elements and content quality
            content_quality = _assess_content_quality(values)
            section_scores[section] = min(element_count / 3.0, 1.0) * content_quality
    
    completeness_score = (9 - empty_sections) / 9.0
    detail_score = sum(section_scores.values()) / len(section_scores)
    
    # Overall health score
    health_score = (completeness_score * 0.6 + detail_score * 0.4)
    
    return {
        "overall_health_score": health_score,
        "completeness_score": completeness_score,
        "detail_score": detail_score,
        "total_elements": total_elements,
        "empty_sections": empty_sections,
        "section_scores": section_scores,
        "recommendations": _generate_health_recommendations(section_scores, empty_sections)
    }


def _assess_content_quality(values: List[str]) -> float:
    """Assess quality of content in a BMC section."""
    if not values:
        return 0.0
    
    quality_score = 0.0
    
    for value in values:
        value_score = 0.5  # Base score
        
        # Length check
        if len(value) > 20:
            value_score += 0.2
        if len(value) > 50:
            value_score += 0.1
            
        # Avoid generic terms
        generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test'}
        if value.lower().strip() not in generic_terms:
            value_score += 0.2
            
        quality_score += min(value_score, 1.0)
    
    return quality_score / len(values)


def _generate_health_recommendations(section_scores: Dict[str, float], empty_sections: int) -> List[str]:
    """Generate recommendations for improving BMC health."""
    recommendations = []
    
    if empty_sections > 0:
        recommendations.append(f"Complete {empty_sections} empty sections to improve business model clarity")
    
    # Find lowest scoring sections
    low_scoring_sections = [
        section.replace('_', ' ').title() 
        for section, score in section_scores.items() 
        if score < 0.5
    ]
    
    if low_scoring_sections:
        recommendations.append(f"Add more detail to: {', '.join(low_scoring_sections[:3])}")
    
    if not recommendations:
        recommendations.append("Business model canvas shows good completeness and detail")
    
    return recommendations


# Export Functions

def export_business_model_to_csv(bmc: BusinessModelCanvas, file_path: str = "business_model.csv") -> bool:
    """
    Export business model canvas to CSV format.

    Args:
        bmc: BusinessModelCanvas to export
        file_path: Path for CSV export

    Returns:
        True if successful
    """
    try:
        sections = [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]

        # Find the maximum number of items in any section
        max_items = max(len(getattr(bmc, section, [])) for section in sections)
        
        # Create data dictionary
        data = {}
        for section in sections:
            section_items = getattr(bmc, section, [])
            column_name = section.replace('_', ' ').title()
            
            # Pad with empty strings to match max_items length
            padded_items = section_items + [''] * (max_items - len(section_items))
            data[column_name] = padded_items

        # Add metadata
        data['Last Updated'] = [bmc.last_updated.strftime('%Y-%m-%d %H:%M')] + [''] * (max_items - 1)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False


def export_changes_to_json(
    changes: List[ProposedChange],
    file_path: str = "proposed_changes.json"
) -> bool:
    """
    Export proposed changes to JSON format.

    Args:
        changes: List of ProposedChange objects
        file_path: Path for JSON export

    Returns:
        True if successful
    """
    try:
        changes_data = []
        for change in changes:
            change_dict = change.dict()
            changes_data.append(change_dict)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_changes": len(changes),
            "proposed_changes": changes_data
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error exporting changes to JSON: {e}")
        return False


def export_analytics_report(
    bmc: BusinessModelCanvas,
    history: List[ChangeHistory],
    file_path: str = "analytics_report.json"
) -> bool:
    """
    Export comprehensive analytics report.

    Args:
        bmc: Current business model canvas
        history: Change history
        file_path: Path for export

    Returns:
        True if successful
    """
    try:
        analytics = generate_change_analytics(history)
        health_score = generate_business_model_health_score(bmc)
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "business_model_health": health_score,
            "change_analytics": analytics,
            "summary": {
                "overall_health": health_score["overall_health_score"],
                "total_changes": analytics["total_changes"],
                "auto_applied_ratio": analytics["auto_applied_ratio"],
                "average_confidence": analytics["average_confidence"]
            }
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error exporting analytics report: {e}")
        return False


# Utility Functions for UI Components

def create_section_summary(bmc: BusinessModelCanvas, section_name: str) -> Dict[str, Any]:
    """Create a summary for a specific BMC section."""
    values = getattr(bmc, section_name, [])
    
    return {
        "section_name": section_name.replace('_', ' ').title(),
        "element_count": len(values),
        "elements": values,
        "is_empty": len(values) == 0,
        "quality_score": _assess_content_quality(values) if values else 0.0
    }


def format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as 'time ago' string."""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "Just now"


def generate_section_icon(section_name: str) -> str:
    """Generate appropriate icon for BMC section."""
    icons = {
        'customer_segments': '👥',
        'value_propositions': '💡',
        'channels': '📡',
        'customer_relationships': '🤝',
        'revenue_streams': '💰',
        'key_resources': '💎',
        'key_activities': '⚙️',
        'key_partnerships': '🤝',
        'cost_structure': '💸'
    }
    return icons.get(section_name, '📋')


# Hash and Validation Functions

def generate_change_hash(change: ProposedChange) -> str:
    """Generate a hash for a change to detect duplicates."""
    change_string = f"{change.canvas_section}_{change.change_type.value}_{change.proposed_value}"
    return hashlib.md5(change_string.encode()).hexdigest()


def deduplicate_changes(changes: List[ProposedChange]) -> List[ProposedChange]:
    """Remove duplicate changes based on content hash."""
    seen_hashes = set()
    unique_changes = []
    
    for change in changes:
        change_hash = generate_change_hash(change)
        if change_hash not in seen_hashes:
            seen_hashes.add(change_hash)
            unique_changes.append(change)
    
    return unique_changes


def validate_bmc_integrity(bmc: BusinessModelCanvas) -> List[str]:
    """Validate business model canvas integrity and return issues."""
    issues = []
    
    # Check for empty sections
    sections = [
        'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
        'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
    ]
    
    for section in sections:
        values = getattr(bmc, section, [])
        if not values:
            issues.append(f"{section.replace('_', ' ').title()} section is empty")
    
    # Check for generic content
    all_values = []
    for section in sections:
        all_values.extend(getattr(bmc, section, []))
    
    generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test'}
    generic_count = sum(1 for value in all_values if value.lower().strip() in generic_terms)
    
    if generic_count > 0:
        issues.append(f"{generic_count} placeholder values found - replace with specific content")
    
    return issues


# Export all utility functions
__all__ = [
    # Data management
    'load_business_model', 'save_business_model', 'create_change_history', 
    'save_change_history', 'load_change_history',
    
    # Validation and safety
    'validate_change_safety', 'validate_bmc_integrity', 'deduplicate_changes',
    
    # Formatting and display
    'format_proposed_changes', 'create_before_after_comparison', 'generate_change_summary',
    'create_section_summary', 'format_time_ago', 'generate_section_icon',
    
    # Business logic
    'calculate_confidence_score', 'determine_change_impact', 'suggest_validation_experiments',
    'apply_changes_to_bmc',
    
    # Analytics
    'generate_change_analytics', 'generate_business_model_health_score',
    
    # Export functions
    'export_business_model_to_csv', 'export_changes_to_json', 'export_analytics_report',
    
    # Utility functions
    'generate_change_hash'
]