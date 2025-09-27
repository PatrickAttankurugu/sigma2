"""
Enhanced utility functions for the Agentic AI Actions Co-pilot system.
Comprehensive error handling, recovery mechanisms, and production-ready reliability.
"""

import json
import os
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONSTANTS AND DEFAULTS ====================

DEFAULT_BMC_DATA = {
    "customer_segments": ["Early adopters and tech-savvy users"],
    "value_propositions": ["Innovative solution addressing key market needs"],
    "channels": ["Digital platforms and direct sales"],
    "customer_relationships": ["Personal assistance and self-service"],
    "revenue_streams": ["Subscription fees and service charges"],
    "key_resources": ["Technology platform and skilled team"],
    "key_activities": ["Product development and customer support"],
    "key_partnerships": ["Strategic technology and business partners"],
    "cost_structure": ["Development costs and operational expenses"],
    "last_updated": None,  # Will be set to current time
    "version": "1.0.0",
    "tags": ["default", "fallback"]
}

# ==================== SAFE DATA MANAGEMENT FUNCTIONS ====================

def create_fallback_bmc() -> BusinessModelCanvas:
    """Create a fallback business model canvas when other loading methods fail"""
    try:
        bmc_data = DEFAULT_BMC_DATA.copy()
        bmc_data["last_updated"] = datetime.now()
        return BusinessModelCanvas(**bmc_data)
    except Exception as e:
        logger.error(f"Failed to create fallback BMC: {str(e)}")
        raise Exception(f"Critical error: Cannot create business model canvas: {str(e)}")


def safe_file_operation(operation: str, file_path: str, data: Any = None) -> Tuple[bool, Any, str]:
    """
    Safely perform file operations with comprehensive error handling
    
    Args:
        operation: 'read' or 'write'
        file_path: Path to the file
        data: Data to write (for write operations)
    
    Returns:
        Tuple of (success: bool, result: Any, error_message: str)
    """
    try:
        # Ensure directory exists for write operations
        if operation == 'write':
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if operation == 'read':
            if not os.path.exists(file_path):
                return False, None, f"File not found: {file_path}"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                return True, result, ""
            except json.JSONDecodeError as e:
                return False, None, f"Invalid JSON in {file_path}: {str(e)}"
            except UnicodeDecodeError as e:
                return False, None, f"Encoding error in {file_path}: {str(e)}"
            
        elif operation == 'write':
            if data is None:
                return False, None, "No data provided for write operation"
            
            try:
                # Create backup if file exists
                if os.path.exists(file_path):
                    backup_path = f"{file_path}.backup"
                    try:
                        os.rename(file_path, backup_path)
                    except Exception:
                        pass  # Continue without backup
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                return True, None, ""
                
            except TypeError as e:
                return False, None, f"Data serialization error: {str(e)}"
            except OSError as e:
                return False, None, f"File system error: {str(e)}"
        
        else:
            return False, None, f"Unknown operation: {operation}"
            
    except Exception as e:
        return False, None, f"Unexpected error in file operation: {str(e)}"


def load_business_model(file_path: str = "business_model.json") -> BusinessModelCanvas:
    """
    Load current business model canvas state from file with comprehensive fallbacks
    
    Args:
        file_path: Path to the BMC JSON file
    
    Returns:
        BusinessModelCanvas object
    """
    try:
        # Try to load from specified file
        success, data, error = safe_file_operation('read', file_path)
        
        if success and data:
            try:
                # Handle datetime parsing for loaded data
                if 'last_updated' in data and isinstance(data['last_updated'], str):
                    try:
                        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                    except ValueError:
                        data['last_updated'] = datetime.now()
                elif 'last_updated' not in data:
                    data['last_updated'] = datetime.now()
                
                # Validate required fields
                required_fields = ['customer_segments', 'value_propositions', 'channels', 
                                 'customer_relationships', 'revenue_streams', 'key_resources',
                                 'key_activities', 'key_partnerships', 'cost_structure']
                
                for field in required_fields:
                    if field not in data:
                        data[field] = []
                    elif not isinstance(data[field], list):
                        data[field] = [str(data[field])] if data[field] else []
                
                # Ensure version exists
                if 'version' not in data:
                    data['version'] = "1.0.0"
                
                return BusinessModelCanvas(**data)
                
            except Exception as e:
                logger.warning(f"Failed to parse BMC data from {file_path}: {str(e)}")
        
        # Try alternative file locations
        alternative_paths = [
            "data/business_model.json",
            "backup/business_model.json",
            f"{file_path}.backup"
        ]
        
        for alt_path in alternative_paths:
            try:
                success, data, error = safe_file_operation('read', alt_path)
                if success and data:
                    logger.info(f"Loaded BMC from alternative path: {alt_path}")
                    # Apply same processing as above
                    if 'last_updated' in data and isinstance(data['last_updated'], str):
                        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                    return BusinessModelCanvas(**data)
            except Exception:
                continue
        
        # If all file loading fails, try to get sample data
        try:
            from sema_business_data import get_sample_business_model_canvas
            logger.info("Loading sample business model canvas")
            return get_sample_business_model_canvas()
        except Exception as e:
            logger.warning(f"Failed to load sample BMC: {str(e)}")
        
        # Final fallback: create default BMC
        logger.info("Creating fallback business model canvas")
        return create_fallback_bmc()
        
    except Exception as e:
        logger.error(f"Critical error in load_business_model: {str(e)}")
        return create_fallback_bmc()


def save_business_model(bmc: BusinessModelCanvas, file_path: str = "business_model.json") -> bool:
    """
    Save business model canvas to file with comprehensive error handling
    
    Args:
        bmc: BusinessModelCanvas to save
        file_path: Path where to save the BMC
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if bmc is None:
            logger.error("Cannot save None business model")
            return False
        
        # Convert to dict and handle datetime serialization
        try:
            bmc_dict = bmc.dict()
            
            # Handle datetime serialization
            if 'last_updated' in bmc_dict and hasattr(bmc_dict['last_updated'], 'isoformat'):
                bmc_dict['last_updated'] = bmc_dict['last_updated'].isoformat()
            
            # Validate data before saving
            if not isinstance(bmc_dict, dict):
                logger.error("BMC data is not a dictionary")
                return False
            
            # Ensure all required fields are present
            required_fields = ['customer_segments', 'value_propositions', 'channels', 
                             'customer_relationships', 'revenue_streams', 'key_resources',
                             'key_activities', 'key_partnerships', 'cost_structure']
            
            for field in required_fields:
                if field not in bmc_dict:
                    bmc_dict[field] = []
                elif not isinstance(bmc_dict[field], list):
                    bmc_dict[field] = []
                    
        except Exception as e:
            logger.error(f"Failed to serialize BMC: {str(e)}")
            return False
        
        # Save to primary location
        success, _, error = safe_file_operation('write', file_path, bmc_dict)
        
        if success:
            # Also save backup copy
            try:
                backup_path = f"backup/{file_path}"
                safe_file_operation('write', backup_path, bmc_dict)
            except Exception:
                pass  # Backup failure is not critical
            
            logger.info(f"Successfully saved BMC to {file_path}")
            return True
        else:
            logger.error(f"Failed to save BMC to {file_path}: {error}")
            
            # Try alternative location
            try:
                alt_path = f"data/{file_path}"
                success, _, error = safe_file_operation('write', alt_path, bmc_dict)
                if success:
                    logger.info(f"Saved BMC to alternative location: {alt_path}")
                    return True
            except Exception:
                pass
            
            return False
        
    except Exception as e:
        logger.error(f"Critical error in save_business_model: {str(e)}")
        return False


def create_change_history(
    old_state: BusinessModelCanvas,
    new_state: BusinessModelCanvas,
    trigger_action_id: str,
    applied_changes: List[ProposedChange],
    auto_applied: bool = False
) -> Optional[ChangeHistory]:
    """
    Create a change history record with comprehensive error handling
    
    Args:
        old_state: BMC state before changes
        new_state: BMC state after changes
        trigger_action_id: ID of action that triggered changes
        applied_changes: List of changes that were applied
        auto_applied: Whether changes were applied automatically
    
    Returns:
        ChangeHistory object or None if creation fails
    """
    try:
        if old_state is None or new_state is None:
            logger.error("Cannot create change history with None states")
            return None
        
        if not applied_changes:
            logger.warning("Creating change history with no applied changes")
            applied_changes = []
        
        # Create snapshots with datetime handling
        try:
            old_snapshot = old_state.dict()
            if 'last_updated' in old_snapshot and hasattr(old_snapshot['last_updated'], 'isoformat'):
                old_snapshot['last_updated'] = old_snapshot['last_updated'].isoformat()
        except Exception as e:
            logger.error(f"Failed to create old state snapshot: {str(e)}")
            old_snapshot = {}
        
        try:
            new_snapshot = new_state.dict()
            if 'last_updated' in new_snapshot and hasattr(new_snapshot['last_updated'], 'isoformat'):
                new_snapshot['last_updated'] = new_snapshot['last_updated'].isoformat()
        except Exception as e:
            logger.error(f"Failed to create new state snapshot: {str(e)}")
            new_snapshot = {}
        
        return ChangeHistory(
            trigger_action_id=str(trigger_action_id) if trigger_action_id else "unknown",
            changes_applied=applied_changes,
            previous_state_snapshot=old_snapshot,
            new_state_snapshot=new_snapshot,
            auto_applied=bool(auto_applied),
            applied_by="user"  # Could be enhanced to track actual user
        )
        
    except Exception as e:
        logger.error(f"Failed to create change history: {str(e)}")
        return None


def save_change_history(history: ChangeHistory, history_file: str = "change_history.json") -> bool:
    """
    Save change history to file with comprehensive error handling
    
    Args:
        history: ChangeHistory to save
        history_file: File to save history records
    
    Returns:
        True if successful
    """
    try:
        if history is None:
            logger.error("Cannot save None change history")
            return False
        
        # Load existing history
        existing_history = []
        success, data, error = safe_file_operation('read', history_file)
        
        if success and data:
            if isinstance(data, list):
                existing_history = data
            else:
                logger.warning(f"Invalid history file format: {history_file}")
                existing_history = []
        
        # Prepare new history record
        try:
            history_dict = history.dict()
            
            # Handle datetime serialization
            if 'timestamp' in history_dict and hasattr(history_dict['timestamp'], 'isoformat'):
                history_dict['timestamp'] = history_dict['timestamp'].isoformat()
            
            # Handle ProposedChange serialization
            changes_list = []
            for change in history.changes_applied:
                try:
                    change_dict = change.dict()
                    # Handle datetime fields if any
                    if 'created_at' in change_dict and hasattr(change_dict['created_at'], 'isoformat'):
                        change_dict['created_at'] = change_dict['created_at'].isoformat()
                    changes_list.append(change_dict)
                except Exception as e:
                    logger.warning(f"Failed to serialize change: {str(e)}")
                    continue
            
            history_dict['changes_applied'] = changes_list
            
        except Exception as e:
            logger.error(f"Failed to serialize change history: {str(e)}")
            return False
        
        # Add new history record
        existing_history.append(history_dict)
        
        # Keep only last 100 records to prevent file from growing too large
        if len(existing_history) > 100:
            existing_history = existing_history[-100:]
        
        # Save updated history
        success, _, error = safe_file_operation('write', history_file, existing_history)
        
        if success:
            logger.info(f"Successfully saved change history to {history_file}")
            return True
        else:
            logger.error(f"Failed to save change history: {error}")
            return False
        
    except Exception as e:
        logger.error(f"Critical error in save_change_history: {str(e)}")
        return False


def load_change_history(history_file: str = "change_history.json") -> List[ChangeHistory]:
    """
    Load change history from file with comprehensive error handling
    
    Args:
        history_file: File containing history records
    
    Returns:
        List of ChangeHistory objects (empty list if loading fails)
    """
    try:
        success, data, error = safe_file_operation('read', history_file)
        
        if not success:
            logger.info(f"No change history file found: {history_file}")
            return []
        
        if not data or not isinstance(data, list):
            logger.warning(f"Invalid change history data in {history_file}")
            return []
        
        # Convert back to ChangeHistory objects
        history_objects = []
        for i, record in enumerate(data):
            try:
                if not isinstance(record, dict):
                    logger.warning(f"Skipping invalid history record {i}: not a dictionary")
                    continue
                
                # Convert ISO timestamp back to datetime
                if 'timestamp' in record and isinstance(record['timestamp'], str):
                    try:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    except ValueError:
                        record['timestamp'] = datetime.now()
                
                # Convert changes back to ProposedChange objects
                changes = []
                for change_data in record.get('changes_applied', []):
                    try:
                        if not isinstance(change_data, dict):
                            continue
                        
                        # Handle datetime conversion if needed
                        if 'created_at' in change_data and isinstance(change_data['created_at'], str):
                            try:
                                change_data['created_at'] = datetime.fromisoformat(change_data['created_at'])
                            except ValueError:
                                change_data['created_at'] = datetime.now()
                        
                        change = ProposedChange(**change_data)
                        changes.append(change)
                        
                    except Exception as e:
                        logger.warning(f"Skipping invalid change in record {i}: {str(e)}")
                        continue
                
                record['changes_applied'] = changes
                
                history_objects.append(ChangeHistory(**record))
                
            except Exception as e:
                logger.warning(f"Skipping invalid history record {i}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(history_objects)} change history records")
        return history_objects
        
    except Exception as e:
        logger.error(f"Critical error loading change history: {str(e)}")
        return []


# ==================== SAFE CHANGE APPLICATION AND VALIDATION ====================

def apply_changes_to_bmc(
    bmc: BusinessModelCanvas,
    changes: List[ProposedChange]
) -> BusinessModelCanvas:
    """
    Apply a list of proposed changes to a business model canvas with comprehensive error handling
    
    Args:
        bmc: Current business model canvas
        changes: List of changes to apply
    
    Returns:
        Updated business model canvas
    """
    try:
        if bmc is None:
            raise ValueError("Cannot apply changes to None business model")
        
        if not changes:
            logger.info("No changes to apply")
            return bmc
        
        # Create a copy to avoid modifying the original
        try:
            updated_bmc = BusinessModelCanvas(**bmc.dict())
        except Exception as e:
            logger.error(f"Failed to create BMC copy: {str(e)}")
            raise ValueError(f"Cannot create business model copy: {str(e)}")
        
        successful_changes = 0
        failed_changes = 0
        
        # Apply changes with conflict resolution
        for i, change in enumerate(changes):
            try:
                if not isinstance(change, ProposedChange):
                    logger.warning(f"Skipping invalid change {i}: not a ProposedChange object")
                    failed_changes += 1
                    continue
                
                # Validate canvas section
                if not hasattr(updated_bmc, change.canvas_section):
                    logger.warning(f"Skipping change {i}: invalid canvas section '{change.canvas_section}'")
                    failed_changes += 1
                    continue
                
                section_values = list(getattr(updated_bmc, change.canvas_section, []))
                
                if change.change_type == ChangeType.ADD:
                    # Avoid duplicates (idempotent behavior)
                    if (change.proposed_value and 
                        change.proposed_value.strip() and 
                        change.proposed_value not in section_values):
                        section_values.append(change.proposed_value.strip())
                        successful_changes += 1
                    else:
                        logger.info(f"Skipping duplicate ADD change {i}")
                
                elif change.change_type == ChangeType.MODIFY and change.current_value:
                    try:
                        index = section_values.index(change.current_value)
                        if change.proposed_value and change.proposed_value.strip():
                            section_values[index] = change.proposed_value.strip()
                            successful_changes += 1
                    except ValueError:
                        # If current value not found, add the new value if it's valid
                        if change.proposed_value and change.proposed_value.strip():
                            section_values.append(change.proposed_value.strip())
                            successful_changes += 1
                            logger.info(f"MODIFY change {i}: current value not found, added new value")
                
                elif change.change_type == ChangeType.REMOVE and change.current_value:
                    try:
                        section_values.remove(change.current_value)
                        successful_changes += 1
                    except ValueError:
                        # Value not found, ignore (idempotent behavior)
                        logger.info(f"REMOVE change {i}: value not found (idempotent)")
                
                # Update the section with modified values
                setattr(updated_bmc, change.canvas_section, section_values)
                
            except Exception as e:
                logger.warning(f"Failed to apply change {i} to {change.canvas_section}: {str(e)}")
                failed_changes += 1
                continue
        
        # Update timestamp and increment version
        try:
            updated_bmc.last_updated = datetime.now()
            updated_bmc._increment_version()
        except Exception as e:
            logger.warning(f"Failed to update BMC metadata: {str(e)}")
        
        logger.info(f"Applied {successful_changes} changes successfully, {failed_changes} failed")
        return updated_bmc
        
    except Exception as e:
        logger.error(f"Critical error in apply_changes_to_bmc: {str(e)}")
        raise Exception(f"Failed to apply changes: {str(e)}")


def validate_change_safety(change: ProposedChange) -> Tuple[bool, List[str]]:
    """
    Determine if a change is safe for automatic application with detailed reasoning
    
    Args:
        change: ProposedChange to validate
    
    Returns:
        Tuple of (is_safe: bool, reasons: List[str])
    """
    try:
        if change is None:
            return False, ["Change is None"]
        
        safety_issues = []
        
        # Basic confidence threshold
        if change.confidence_score < 0.8:
            safety_issues.append(f"Confidence score too low: {change.confidence_score:.2f} < 0.8")
        
        # No removals in auto-mode (too risky)
        if change.change_type == ChangeType.REMOVE:
            safety_issues.append("Removal operations not allowed in auto-mode")
        
        # Check for critical sections that need manual review
        critical_sections = {'revenue_streams', 'cost_structure'}
        if change.canvas_section in critical_sections:
            safety_issues.append(f"Critical section '{change.canvas_section}' requires manual approval")
        
        # Validate proposed value is substantial and not generic
        if not change.proposed_value or len(change.proposed_value.strip()) < 10:
            safety_issues.append("Proposed value too short or empty")
        
        # Check for placeholder/generic content
        generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test', 
                        'lorem ipsum', 'sample', 'demo', 'todo'}
        if change.proposed_value.lower().strip() in generic_terms:
            safety_issues.append("Proposed value contains generic/placeholder content")
        
        # Check reasoning quality for auto-mode
        if not change.reasoning or len(change.reasoning.strip()) < 30:
            safety_issues.append("Reasoning too short or missing")
        
        # Check for high-risk indicators in reasoning
        high_risk_indicators = ['major', 'significant', 'fundamental', 'critical', 'breaking', 'removing']
        if any(indicator in change.reasoning.lower() for indicator in high_risk_indicators):
            safety_issues.append("High-risk indicators found in reasoning")
        
        is_safe = len(safety_issues) == 0
        return is_safe, safety_issues
        
    except Exception as e:
        logger.error(f"Error validating change safety: {str(e)}")
        return False, [f"Validation error: {str(e)}"]


def detect_change_conflicts(changes: List[ProposedChange]) -> List[str]:
    """
    Detect potential conflicts between proposed changes with enhanced analysis
    
    Args:
        changes: List of proposed changes
    
    Returns:
        List of conflict descriptions
    """
    try:
        if not changes:
            return []
        
        conflicts = []
        
        # Group changes by section
        section_changes = {}
        for i, change in enumerate(changes):
            try:
                if not isinstance(change, ProposedChange):
                    conflicts.append(f"Change {i}: Invalid change type")
                    continue
                
                section = change.canvas_section
                if section not in section_changes:
                    section_changes[section] = []
                section_changes[section].append((i, change))
                
            except Exception as e:
                conflicts.append(f"Change {i}: Error processing - {str(e)}")
        
        # Check for conflicts within each section
        for section, section_change_list in section_changes.items():
            if len(section_change_list) > 1:
                # Multiple changes to same section - check for conflicts
                for i, (idx1, change1) in enumerate(section_change_list):
                    for idx2, change2 in section_change_list[i+1:]:
                        try:
                            if _changes_conflict(change1, change2):
                                conflicts.append(
                                    f"Conflict in {section}: Change {idx1} ({change1.change_type.value}) "
                                    f"vs Change {idx2} ({change2.change_type.value})"
                                )
                        except Exception as e:
                            conflicts.append(f"Error checking conflict between changes {idx1} and {idx2}: {str(e)}")
        
        return conflicts
        
    except Exception as e:
        logger.error(f"Error detecting change conflicts: {str(e)}")
        return [f"Conflict detection error: {str(e)}"]


def _changes_conflict(change1: ProposedChange, change2: ProposedChange) -> bool:
    """Check if two changes conflict with each other with enhanced logic"""
    try:
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
        
        # Both try to add the same value
        if (change1.change_type == ChangeType.ADD and 
            change2.change_type == ChangeType.ADD and
            change1.proposed_value == change2.proposed_value):
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking change conflict: {str(e)}")
        return False  # Assume no conflict if we can't determine


# ==================== DUPLICATE DETECTION (IDEMPOTENT BEHAVIOR) ====================

def generate_change_hash(change: ProposedChange) -> str:
    """
    Generate a hash for a change to detect duplicates with enhanced algorithm
    
    Args:
        change: ProposedChange to hash
    
    Returns:
        MD5 hash string
    """
    try:
        if change is None:
            return "null_change"
        
        change_string = f"{change.canvas_section}_{change.change_type.value}_{change.proposed_value}_{change.current_value}"
        return hashlib.md5(change_string.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Error generating change hash: {str(e)}")
        return f"error_{str(hash(str(change)))}"


def generate_action_hash(action_data: Dict[str, Any]) -> str:
    """
    Generate a hash for an action to detect duplicate processing with enhanced algorithm
    
    Args:
        action_data: Dictionary containing action information
    
    Returns:
        MD5 hash string
    """
    try:
        if not action_data:
            return "empty_action"
        
        # Use title, results_data, and outcome as key components
        title = action_data.get('title', '')
        results = action_data.get('results_data', '')
        outcome = action_data.get('outcome', '')
        
        action_string = f"{title}_{results}_{outcome}"
        return hashlib.md5(action_string.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Error generating action hash: {str(e)}")
        return f"error_{str(hash(str(action_data)))}"


def deduplicate_changes(changes: List[ProposedChange]) -> List[ProposedChange]:
    """
    Remove duplicate changes based on content hash with enhanced logic
    
    Args:
        changes: List of proposed changes
    
    Returns:
        List of unique changes
    """
    try:
        if not changes:
            return []
        
        seen_hashes = set()
        unique_changes = []
        duplicate_count = 0
        
        for i, change in enumerate(changes):
            try:
                if not isinstance(change, ProposedChange):
                    logger.warning(f"Skipping invalid change {i}: not a ProposedChange object")
                    continue
                
                change_hash = generate_change_hash(change)
                if change_hash not in seen_hashes:
                    seen_hashes.add(change_hash)
                    unique_changes.append(change)
                else:
                    duplicate_count += 1
                    logger.info(f"Removing duplicate change {i}: {change.canvas_section} - {change.change_type.value}")
                    
            except Exception as e:
                logger.warning(f"Error processing change {i} for deduplication: {str(e)}")
                continue
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate changes")
        
        return unique_changes
        
    except Exception as e:
        logger.error(f"Error in deduplicate_changes: {str(e)}")
        return changes  # Return original list if deduplication fails


# ==================== UI HELPER FUNCTIONS ====================

def format_proposed_changes(changes: List[ProposedChange]) -> List[Dict[str, Any]]:
    """
    Format proposed changes for human-readable display with comprehensive error handling
    
    Args:
        changes: List of ProposedChange objects
    
    Returns:
        List of formatted change dictionaries
    """
    try:
        if not changes:
            return []
        
        formatted_changes = []
        
        for i, change in enumerate(changes):
            try:
                if not isinstance(change, ProposedChange):
                    logger.warning(f"Skipping invalid change {i}: not a ProposedChange object")
                    continue
                
                # Safe access to change properties
                section = getattr(change, 'canvas_section', 'unknown').replace("_", " ").title()
                change_type = getattr(change, 'change_type', ChangeType.ADD).value.title()
                confidence_score = getattr(change, 'confidence_score', 0.0)
                reasoning = getattr(change, 'reasoning', 'No reasoning provided')
                impact = getattr(change, 'impact_assessment', 'medium')
                
                # Format description safely
                try:
                    description = _format_change_description(change)
                except Exception as e:
                    description = f"Error formatting description: {str(e)}"
                
                # Format confidence safely
                try:
                    confidence_str = f"{confidence_score:.0%}"
                    confidence_level = _get_confidence_category(confidence_score)
                except Exception as e:
                    confidence_str = "Unknown"
                    confidence_level = "Unknown"
                
                # Determine safety level
                try:
                    is_safe, safety_reasons = validate_change_safety(change)
                    safety_level = "Safe for Auto-mode" if is_safe else "Manual Review Required"
                except Exception as e:
                    safety_level = "Safety check failed"
                
                formatted_change = {
                    "section": section,
                    "action": change_type,
                    "description": description,
                    "reasoning": reasoning,
                    "confidence": confidence_str,
                    "confidence_level": confidence_level,
                    "safety_level": safety_level,
                    "impact_assessment": impact
                }
                
                formatted_changes.append(formatted_change)
                
            except Exception as e:
                logger.warning(f"Error formatting change {i}: {str(e)}")
                # Add error entry to maintain index consistency
                formatted_changes.append({
                    "section": "Error",
                    "action": "Error",
                    "description": f"Error formatting change: {str(e)}",
                    "reasoning": "Error occurred during formatting",
                    "confidence": "0%",
                    "confidence_level": "Error",
                    "safety_level": "Error",
                    "impact_assessment": "unknown"
                })
        
        return formatted_changes
        
    except Exception as e:
        logger.error(f"Critical error in format_proposed_changes: {str(e)}")
        return [{"section": "Critical Error", "action": "Error", 
                "description": f"Critical formatting error: {str(e)}", 
                "reasoning": "System error", "confidence": "0%", 
                "confidence_level": "Error", "safety_level": "Error", 
                "impact_assessment": "unknown"}]


def _format_change_description(change: ProposedChange) -> str:
    """Format individual change description for display with error handling"""
    try:
        if change.change_type == ChangeType.ADD:
            return f"Add: {_truncate_text(change.proposed_value, 80)}"
        elif change.change_type == ChangeType.MODIFY and change.current_value:
            return f"Change '{_truncate_text(change.current_value, 30)}' to '{_truncate_text(change.proposed_value, 30)}'"
        elif change.change_type == ChangeType.REMOVE and change.current_value:
            return f"Remove: {_truncate_text(change.current_value, 60)}"
        else:
            return _truncate_text(change.proposed_value, 60)
    except Exception as e:
        return f"Error formatting description: {str(e)}"


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to specified length with ellipsis and error handling"""
    try:
        if not text:
            return ""
        
        text_str = str(text)
        if len(text_str) <= max_length:
            return text_str
        return text_str[:max_length-3] + "..."
        
    except Exception as e:
        return f"[Error: {str(e)}]"


def _get_confidence_category(score: float) -> str:
    """Get confidence category label for UI display with error handling"""
    try:
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"
    except Exception:
        return "Unknown"


# ==================== EXPORT FUNCTIONS ====================

def export_business_model_to_csv(bmc: BusinessModelCanvas, file_path: str = "business_model.csv") -> Tuple[bool, str]:
    """
    Export business model canvas to CSV format with comprehensive error handling
    
    Args:
        bmc: BusinessModelCanvas to export
        file_path: Path for CSV export
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if bmc is None:
            return False, "Cannot export None business model"
        
        sections = [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]
        
        # Find the maximum number of items in any section
        max_items = 1
        for section in sections:
            try:
                items = getattr(bmc, section, [])
                max_items = max(max_items, len(items))
            except Exception:
                continue
        
        # Create data dictionary
        data = {}
        for section in sections:
            try:
                section_items = getattr(bmc, section, [])
                column_name = section.replace('_', ' ').title()
                
                # Pad with empty strings to match max_items length
                padded_items = section_items + [''] * (max_items - len(section_items))
                data[column_name] = padded_items
            except Exception as e:
                logger.warning(f"Error processing section {section}: {str(e)}")
                data[section.replace('_', ' ').title()] = [''] * max_items
        
        # Add metadata
        try:
            data['Last Updated'] = [bmc.last_updated.strftime('%Y-%m-%d %H:%M')] + [''] * (max_items - 1)
            data['Version'] = [bmc.version] + [''] * (max_items - 1)
        except Exception:
            data['Last Updated'] = ['Unknown'] + [''] * (max_items - 1)
            data['Version'] = ['Unknown'] + [''] * (max_items - 1)
        
        # Create DataFrame and save
        try:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            return True, f"Successfully exported to {file_path}"
        except Exception as e:
            return False, f"Failed to save CSV file: {str(e)}"
        
    except Exception as e:
        logger.error(f"Critical error in CSV export: {str(e)}")
        return False, f"Critical export error: {str(e)}"


# ==================== VALIDATION AND UTILITY FUNCTIONS ====================

def validate_bmc_integrity(bmc: BusinessModelCanvas) -> Tuple[bool, List[str]]:
    """
    Validate business model canvas integrity with comprehensive checks
    
    Args:
        bmc: BusinessModelCanvas to validate
    
    Returns:
        Tuple of (is_valid: bool, issues: List[str])
    """
    try:
        if bmc is None:
            return False, ["Business model is None"]
        
        issues = []
        
        sections = [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]
        
        # Check for completely empty sections
        empty_sections = []
        for section in sections:
            try:
                values = getattr(bmc, section, [])
                if not values:
                    empty_sections.append(section.replace('_', ' ').title())
            except Exception as e:
                issues.append(f"Error accessing section {section}: {str(e)}")
        
        if len(empty_sections) > 5:  # More than half empty is concerning
            issues.append(f"Too many empty sections ({len(empty_sections)}): {', '.join(empty_sections)}")
        
        # Check for placeholder content
        try:
            all_values = []
            for section in sections:
                try:
                    values = getattr(bmc, section, [])
                    all_values.extend(values)
                except Exception:
                    continue
            
            if all_values:
                placeholder_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo'}
                placeholder_count = sum(1 for value in all_values if str(value).lower().strip() in placeholder_terms)
                
                if placeholder_count > 0:
                    issues.append(f"{placeholder_count} placeholder values found - replace with specific content")
                
                # Check for very short descriptions
                short_items = [item for item in all_values if len(str(item).strip()) < 10]
                if len(short_items) > len(all_values) * 0.5:  # More than 50% short items
                    issues.append(f"{len(short_items)} items are very short (< 10 characters)")
                    
        except Exception as e:
            issues.append(f"Error validating content quality: {str(e)}")
        
        # Check metadata
        try:
            if not hasattr(bmc, 'version') or not bmc.version:
                issues.append("Missing version information")
            
            if not hasattr(bmc, 'last_updated') or not bmc.last_updated:
                issues.append("Missing last updated timestamp")
                
        except Exception as e:
            issues.append(f"Error validating metadata: {str(e)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
        
    except Exception as e:
        logger.error(f"Error validating BMC integrity: {str(e)}")
        return False, [f"Validation error: {str(e)}"]


def format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as 'time ago' string with error handling"""
    try:
        if timestamp is None:
            return "Unknown"
        
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
            
    except Exception as e:
        logger.error(f"Error formatting time: {str(e)}")
        return "Time unknown"


def get_bmc_section_icon(section_name: str) -> str:
    """Generate appropriate icon for BMC section with error handling"""
    try:
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
    except Exception:
        return '📋'


# Export key functions for clean imports
__all__ = [
    # Data management
    'load_business_model', 'save_business_model', 'create_fallback_bmc',
    'create_change_history', 'save_change_history', 'load_change_history',
    
    # Change application and validation
    'apply_changes_to_bmc', 'validate_change_safety', 'detect_change_conflicts',
    
    # Duplicate detection (idempotent behavior)
    'generate_change_hash', 'generate_action_hash', 'deduplicate_changes',
    
    # UI helpers and formatting
    'format_proposed_changes', 'format_time_ago', 'get_bmc_section_icon',
    
    # Validation and utilities
    'validate_bmc_integrity', 'safe_file_operation',
    
    # Export
    'export_business_model_to_csv'
]