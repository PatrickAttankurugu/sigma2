"""
Enhanced utility functions for the Agentic AI Actions Co-pilot system.
Streamlined for Seedstars assignment focusing on core agentic workflow,
version control, and idempotent behavior.
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

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

# ==================== DATA MANAGEMENT FUNCTIONS ====================

def load_business_model(file_path: str = "business_model.json") -> BusinessModelCanvas:
    """
    Load current business model canvas state from file.
    Falls back to sample data for clean demo experience.

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
            # Return sample BMC for demo
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
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
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
    Create a change history record for version tracking and undo functionality.

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
        auto_applied=auto_applied,
        applied_by="user"  # Could be enhanced to track actual user
    )


def save_change_history(history: ChangeHistory, history_file: str = "change_history.json") -> bool:
    """
    Save change history to file for audit trail and undo functionality.

    Args:
        history: ChangeHistory to save
        history_file: File to save history records

    Returns:
        True if successful
    """
    try:
        # Ensure directory exists
        Path(history_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        existing_history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    existing_history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_history = []

        # Add new history record
        history_dict = history.dict()
        history_dict['timestamp'] = history.timestamp.isoformat()
        
        # Handle ProposedChange serialization
        changes_list = []
        for change in history.changes_applied:
            change_dict = change.dict()
            # Handle datetime fields if any
            if 'created_at' in change_dict and hasattr(change_dict['created_at'], 'isoformat'):
                change_dict['created_at'] = change_dict['created_at'].isoformat()
            changes_list.append(change_dict)
        history_dict['changes_applied'] = changes_list
        
        existing_history.append(history_dict)

        # Keep only last 50 records to prevent file from growing too large
        if len(existing_history) > 50:
            existing_history = existing_history[-50:]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error saving change history: {e}")
        return False


def load_change_history(history_file: str = "change_history.json") -> List[ChangeHistory]:
    """
    Load change history from file for version control functionality.

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
                if isinstance(record['timestamp'], str):
                    record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                
                # Convert changes back to ProposedChange objects
                changes = []
                for change_data in record.get('changes_applied', []):
                    # Handle datetime conversion if needed
                    if 'created_at' in change_data and isinstance(change_data['created_at'], str):
                        change_data['created_at'] = datetime.fromisoformat(change_data['created_at'])
                    
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


# ==================== CHANGE APPLICATION AND VALIDATION ====================

def apply_changes_to_bmc(
    bmc: BusinessModelCanvas,
    changes: List[ProposedChange]
) -> BusinessModelCanvas:
    """
    Apply a list of proposed changes to a business model canvas.
    Implements idempotent behavior and conflict resolution.

    Args:
        bmc: Current business model canvas
        changes: List of changes to apply

    Returns:
        Updated business model canvas
    """
    # Create a copy to avoid modifying the original
    updated_bmc = BusinessModelCanvas(**bmc.dict())

    # Apply changes with conflict resolution
    for change in changes:
        try:
            section_values = list(getattr(updated_bmc, change.canvas_section, []))

            if change.change_type == ChangeType.ADD:
                # Avoid duplicates (idempotent behavior)
                if (change.proposed_value.strip() and 
                    change.proposed_value not in section_values):
                    section_values.append(change.proposed_value.strip())

            elif change.change_type == ChangeType.MODIFY and change.current_value:
                try:
                    index = section_values.index(change.current_value)
                    if change.proposed_value.strip():
                        section_values[index] = change.proposed_value.strip()
                except ValueError:
                    # If current value not found, add the new value if it's valid
                    # This provides graceful handling for concurrent modifications
                    if change.proposed_value.strip():
                        section_values.append(change.proposed_value.strip())

            elif change.change_type == ChangeType.REMOVE and change.current_value:
                try:
                    section_values.remove(change.current_value)
                except ValueError:
                    # Value not found, ignore (idempotent behavior)
                    pass

            # Update the section with modified values
            setattr(updated_bmc, change.canvas_section, section_values)
            
        except Exception as e:
            print(f"Warning: Could not apply change to {change.canvas_section}: {e}")
            continue

    # Update timestamp and increment version
    updated_bmc.last_updated = datetime.now()
    updated_bmc._increment_version()
    
    return updated_bmc


def validate_change_safety(change: ProposedChange) -> bool:
    """
    Determine if a change is safe for automatic application in auto-mode.
    Enhanced safety criteria for production-ready auto-mode.

    Args:
        change: ProposedChange to validate

    Returns:
        True if safe for auto-application
    """
    # Basic confidence threshold
    if change.confidence_score < 0.8:  # Higher threshold for auto-mode
        return False

    # No removals in auto-mode (too risky)
    if change.change_type == ChangeType.REMOVE:
        return False

    # Check for critical sections that need manual review
    critical_sections = {'revenue_streams', 'cost_structure'}
    if change.canvas_section in critical_sections:
        return False  # Always require manual approval for financial sections

    # Validate proposed value is substantial and not generic
    if not change.proposed_value or len(change.proposed_value.strip()) < 10:
        return False
        
    # Check for placeholder/generic content
    generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test', 
                    'lorem ipsum', 'sample', 'demo', 'todo'}
    if change.proposed_value.lower().strip() in generic_terms:
        return False
    
    # Check reasoning quality for auto-mode
    if not change.reasoning or len(change.reasoning.strip()) < 30:
        return False

    return True


def detect_change_conflicts(changes: List[ProposedChange]) -> List[str]:
    """
    Detect potential conflicts between proposed changes.

    Args:
        changes: List of proposed changes

    Returns:
        List of conflict descriptions
    """
    conflicts = []
    
    # Group changes by section
    section_changes = {}
    for change in changes:
        section = change.canvas_section
        if section not in section_changes:
            section_changes[section] = []
        section_changes[section].append(change)
    
    # Check for conflicts within each section
    for section, section_change_list in section_changes.items():
        if len(section_change_list) > 1:
            # Multiple changes to same section - check for conflicts
            for i, change1 in enumerate(section_change_list):
                for change2 in section_change_list[i+1:]:
                    if _changes_conflict(change1, change2):
                        conflicts.append(
                            f"Conflict in {section}: {change1.change_type.value} vs {change2.change_type.value}"
                        )
    
    return conflicts


def _changes_conflict(change1: ProposedChange, change2: ProposedChange) -> bool:
    """Check if two changes conflict with each other."""
    # Same value being modified differently
    if (change1.current_value and change2.current_value and 
        change1.current_value == change2.current_value and
        change1.proposed_value != change2.proposed_value):
        return True
    
    # One removes what the other modifies
    if (change1.change_type == ChangeType.REMOVE and 
        change2.change_type == ChangeType.MODIFY and
        change1.current_value == change2.current_value):
        return True
    
    return False


# ==================== DUPLICATE DETECTION (IDEMPOTENT BEHAVIOR) ====================

def generate_change_hash(change: ProposedChange) -> str:
    """
    Generate a hash for a change to detect duplicates.
    Essential for idempotent behavior.

    Args:
        change: ProposedChange to hash

    Returns:
        MD5 hash string
    """
    change_string = f"{change.canvas_section}_{change.change_type.value}_{change.proposed_value}"
    return hashlib.md5(change_string.encode()).hexdigest()


def generate_action_hash(action_data: Dict[str, Any]) -> str:
    """
    Generate a hash for an action to detect duplicate processing.

    Args:
        action_data: Dictionary containing action information

    Returns:
        MD5 hash string
    """
    # Use title and results_data as key components
    action_string = f"{action_data.get('title', '')}{action_data.get('results_data', '')}{action_data.get('outcome', '')}"
    return hashlib.md5(action_string.encode()).hexdigest()


def deduplicate_changes(changes: List[ProposedChange]) -> List[ProposedChange]:
    """
    Remove duplicate changes based on content hash.
    Ensures idempotent behavior when applying changes.

    Args:
        changes: List of proposed changes

    Returns:
        List of unique changes
    """
    seen_hashes = set()
    unique_changes = []
    
    for change in changes:
        change_hash = generate_change_hash(change)
        if change_hash not in seen_hashes:
            seen_hashes.add(change_hash)
            unique_changes.append(change)
    
    return unique_changes


# ==================== UI HELPER FUNCTIONS ====================

def format_proposed_changes(changes: List[ProposedChange]) -> List[Dict[str, Any]]:
    """
    Format proposed changes for human-readable display in the UI.

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
            "safety_level": "Safe for Auto-mode" if validate_change_safety(change) else "Manual Review Required",
            "impact_assessment": change.impact_assessment or "medium"
        }
        formatted_changes.append(formatted_change)

    return formatted_changes


def _format_change_description(change: ProposedChange) -> str:
    """Format individual change description for display."""
    if change.change_type == ChangeType.ADD:
        return f"Add: {_truncate_text(change.proposed_value, 80)}"
    elif change.change_type == ChangeType.MODIFY and change.current_value:
        return f"Change '{_truncate_text(change.current_value, 30)}' to '{_truncate_text(change.proposed_value, 30)}'"
    elif change.change_type == ChangeType.REMOVE and change.current_value:
        return f"Remove: {_truncate_text(change.current_value, 60)}"
    else:
        return _truncate_text(change.proposed_value, 60)


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length with ellipsis."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length-3] + "..."


def _get_confidence_category(score: float) -> str:
    """Get confidence category label for UI display."""
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
    Create side-by-side comparison of BMC states for version control display.

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
    Generate a human-readable summary of applied changes for notifications.

    Args:
        applied_changes: List of changes that were applied

    Returns:
        Summary text for UI notifications
    """
    if not applied_changes:
        return "No changes were applied."

    change_count = len(applied_changes)
    sections_affected = len(set(change.canvas_section for change in applied_changes))

    summary = f"Applied {change_count} change(s) to {sections_affected} section(s):\n"

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


# ==================== VERSION CONTROL FUNCTIONS ====================

def can_undo_change(history: List[ChangeHistory]) -> bool:
    """
    Check if there are changes that can be undone.

    Args:
        history: List of change history records

    Returns:
        True if undo is possible
    """
    return len(history) > 0


def get_last_change_summary(history: List[ChangeHistory]) -> Optional[str]:
    """
    Get a summary of the last change for undo confirmation.

    Args:
        history: List of change history records

    Returns:
        Summary string or None if no history
    """
    if not history:
        return None
    
    last_change = history[-1]
    change_count = len(last_change.changes_applied)
    timestamp = last_change.timestamp.strftime("%H:%M:%S")
    
    return f"Undo {change_count} change(s) from {timestamp}?"


def validate_bmc_integrity(bmc: BusinessModelCanvas) -> List[str]:
    """
    Validate business model canvas integrity and return any issues.
    Used for quality assurance after changes are applied.

    Args:
        bmc: BusinessModelCanvas to validate

    Returns:
        List of validation issues (empty if no issues)
    """
    issues = []
    
    sections = [
        'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
        'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
    ]
    
    # Check for completely empty sections
    empty_sections = []
    for section in sections:
        values = getattr(bmc, section, [])
        if not values:
            empty_sections.append(section.replace('_', ' ').title())
    
    if empty_sections:
        issues.append(f"Empty sections: {', '.join(empty_sections)}")
    
    # Check for placeholder content
    all_values = []
    for section in sections:
        all_values.extend(getattr(bmc, section, []))
    
    placeholder_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo'}
    placeholder_count = sum(1 for value in all_values if value.lower().strip() in placeholder_terms)
    
    if placeholder_count > 0:
        issues.append(f"{placeholder_count} placeholder values found - replace with specific content")
    
    # Check for very short descriptions
    short_items = [item for item in all_values if len(item.strip()) < 10]
    if len(short_items) > 0:
        issues.append(f"{len(short_items)} items are very short (< 10 characters)")
    
    return issues


# ==================== EXPORT FUNCTIONS (SIMPLIFIED FOR ASSIGNMENT) ====================

def export_business_model_to_csv(bmc: BusinessModelCanvas, file_path: str = "business_model.csv") -> bool:
    """
    Export business model canvas to CSV format for external use.

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
        max_items = max(len(getattr(bmc, section, [])) for section in sections) or 1
        
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
        data['Version'] = [bmc.version] + [''] * (max_items - 1)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False


# ==================== UTILITY FUNCTIONS FOR UI COMPONENTS ====================

def format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as 'time ago' string for UI display."""
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


def get_bmc_section_icon(section_name: str) -> str:
    """Generate appropriate emoji icon for BMC section."""
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


def create_section_summary(bmc: BusinessModelCanvas, section_name: str) -> Dict[str, Any]:
    """Create a summary for a specific BMC section for UI display."""
    values = getattr(bmc, section_name, [])
    
    return {
        "section_name": section_name.replace('_', ' ').title(),
        "icon": get_bmc_section_icon(section_name),
        "element_count": len(values),
        "elements": values,
        "is_empty": len(values) == 0,
        "quality_score": _assess_content_quality(values) if values else 0.0
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
            value_score += 0.3
            
        quality_score += min(value_score, 1.0)
    
    return min(quality_score / len(values), 1.0)


# Export key functions for clean imports
__all__ = [
    # Data management
    'load_business_model', 'save_business_model', 
    'create_change_history', 'save_change_history', 'load_change_history',
    
    # Change application and validation
    'apply_changes_to_bmc', 'validate_change_safety', 'detect_change_conflicts',
    
    # Duplicate detection (idempotent behavior)
    'generate_change_hash', 'generate_action_hash', 'deduplicate_changes',
    
    # UI helpers and formatting
    'format_proposed_changes', 'create_before_after_comparison', 'generate_change_summary',
    'create_section_summary', 'format_time_ago', 'get_bmc_section_icon',
    
    # Version control
    'can_undo_change', 'get_last_change_summary', 'validate_bmc_integrity',
    
    # Export
    'export_business_model_to_csv'
]