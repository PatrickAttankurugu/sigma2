"""
Utility functions for the Agentic AI Actions Co-pilot system.

This module provides data management, UI helpers, and business logic utilities
for the business model canvas management system.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

from business_models import (
    BusinessModelCanvas,
    ProposedChange,
    ChangeHistory,
    ChangeType,
    ConfidenceLevel
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
                return BusinessModelCanvas(**data)
        else:
            # Return sample BMC if no file exists
            return get_sample_business_model_canvas()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
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
    return ChangeHistory(
        trigger_action_id=trigger_action_id,
        changes_applied=applied_changes,
        previous_state_snapshot=old_state.dict(),
        new_state_snapshot=new_state.dict(),
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
            # Convert ISO timestamp back to datetime
            record['timestamp'] = datetime.fromisoformat(record['timestamp'])
            history_objects.append(ChangeHistory(**record))

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

    return True


# UI Helper Functions

def format_proposed_changes(changes: List[ProposedChange]) -> List[Dict[str, str]]:
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
            "confidence_level": _get_confidence_category(change.confidence_score)
        }
        formatted_changes.append(formatted_change)

    return formatted_changes


def _format_change_description(change: ProposedChange) -> str:
    """Format individual change description."""
    if change.change_type == ChangeType.ADD:
        return f"Add: {change.proposed_value}"
    elif change.change_type == ChangeType.MODIFY and change.current_value:
        return f"Change '{change.current_value}' to '{change.proposed_value}'"
    elif change.change_type == ChangeType.REMOVE and change.current_value:
        return f"Remove: {change.current_value}"
    else:
        return change.proposed_value


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
) -> Dict[str, Dict[str, List[str]]]:
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

        comparison[section] = {
            "before": old_values,
            "after": new_values,
            "changed": old_values != new_values
        }

    return comparison


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
        action_summary = ", ".join(set(actions))  # Remove duplicates
        summary += f"• {section}: {action_summary}\n"

    return summary.strip()


# Business Logic Functions

def calculate_confidence_score(
    reasoning: str,
    historical_data: Optional[List[ChangeHistory]] = None
) -> float:
    """
    Calculate confidence score based on reasoning quality and historical data.

    Args:
        reasoning: AI reasoning for the change
        historical_data: Past change history for context

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Base score from reasoning quality
    base_score = _assess_reasoning_quality(reasoning)

    # Adjust based on historical success if available
    if historical_data:
        historical_adjustment = _assess_historical_performance(historical_data)
        base_score = (base_score + historical_adjustment) / 2

    return max(0.0, min(1.0, base_score))


def _assess_reasoning_quality(reasoning: str) -> float:
    """Assess quality of AI reasoning text."""
    if not reasoning or len(reasoning) < 50:
        return 0.3

    score = 0.5  # Base score

    # Check for specific elements that indicate good reasoning
    quality_indicators = [
        'data shows', 'evidence suggests', 'analysis indicates',
        'market research', 'customer feedback', 'metrics demonstrate',
        'specifically', 'therefore', 'because', 'resulted in'
    ]

    reasoning_lower = reasoning.lower()
    for indicator in quality_indicators:
        if indicator in reasoning_lower:
            score += 0.1

    # Penalize vague language
    vague_indicators = [
        'might', 'could be', 'possibly', 'perhaps', 'maybe',
        'unclear', 'uncertain', 'ambiguous'
    ]

    for indicator in vague_indicators:
        if indicator in reasoning_lower:
            score -= 0.1

    return max(0.1, min(1.0, score))


def _assess_historical_performance(historical_data: List[ChangeHistory]) -> float:
    """Assess performance based on historical change success."""
    if not historical_data:
        return 0.5

    # This is a simplified version - in practice you'd want to track
    # actual outcomes of applied changes
    recent_changes = historical_data[-10:]  # Last 10 changes

    # For now, return neutral score
    # In a real system, you'd track whether changes led to positive outcomes
    return 0.5


def determine_change_impact(change: ProposedChange, bmc: BusinessModelCanvas) -> str:
    """
    Assess the significance of a proposed change.

    Args:
        change: The proposed change
        bmc: Current business model canvas

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


def suggest_validation_experiments(updated_bmc: BusinessModelCanvas) -> List[str]:
    """
    Suggest validation experiments based on updated business model.

    Args:
        updated_bmc: The updated business model canvas

    Returns:
        List of suggested validation experiments
    """
    suggestions = []

    # Customer segment validation
    if updated_bmc.customer_segments:
        suggestions.append(
            "Conduct customer interviews with new target segments to validate assumptions"
        )

    # Value proposition testing
    if updated_bmc.value_propositions:
        suggestions.append(
            "Run A/B tests on updated value propositions with existing customers"
        )

    # Channel effectiveness
    if updated_bmc.channels:
        suggestions.append(
            "Test new channels with small pilot groups to measure effectiveness"
        )

    # Revenue stream validation
    if updated_bmc.revenue_streams:
        suggestions.append(
            "Model financial projections for new revenue streams and test willingness to pay"
        )

    # Partnership validation
    if updated_bmc.key_partnerships:
        suggestions.append(
            "Reach out to potential partners to validate partnership opportunities"
        )

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
            if change.proposed_value not in section_values:
                section_values.append(change.proposed_value)

        elif change.change_type == ChangeType.MODIFY and change.current_value:
            try:
                index = section_values.index(change.current_value)
                section_values[index] = change.proposed_value
            except ValueError:
                # If current value not found, add the new value
                section_values.append(change.proposed_value)

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
        data = []
        sections = [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]

        max_items = max(len(getattr(bmc, section, [])) for section in sections)

        for i in range(max_items):
            row = {}
            for section in sections:
                section_items = getattr(bmc, section, [])
                row[section.replace('_', ' ').title()] = section_items[i] if i < len(section_items) else ""
            data.append(row)

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

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(changes_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error exporting changes to JSON: {e}")
        return False