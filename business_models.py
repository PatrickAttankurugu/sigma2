"""
Enhanced Pydantic models for the Agentic AI Actions Co-pilot system.
Comprehensive error handling, validation, and production-ready reliability.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import re

from pydantic import BaseModel, Field, field_validator, model_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CORE ENUMS ====================

class ActionOutcome(str, Enum):
    """Enumeration of possible action outcomes."""
    SUCCESSFUL = "successful"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def from_string(cls, value: str) -> 'ActionOutcome':
        """Safely convert string to ActionOutcome"""
        try:
            if isinstance(value, cls):
                return value
            
            normalized = str(value).lower().strip()
            for outcome in cls:
                if outcome.value.lower() == normalized:
                    return outcome
            
            logger.warning(f"Unknown action outcome '{value}', defaulting to INCONCLUSIVE")
            return cls.INCONCLUSIVE
        except Exception as e:
            logger.error(f"Error converting action outcome: {str(e)}")
            return cls.INCONCLUSIVE


class ChangeType(str, Enum):
    """Types of changes that can be made to business model sections."""
    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"

    @classmethod
    def from_string(cls, value: str) -> 'ChangeType':
        """Safely convert string to ChangeType"""
        try:
            if isinstance(value, cls):
                return value
            
            normalized = str(value).lower().strip()
            for change_type in cls:
                if change_type.value.lower() == normalized:
                    return change_type
            
            logger.warning(f"Unknown change type '{value}', defaulting to ADD")
            return cls.ADD
        except Exception as e:
            logger.error(f"Error converting change type: {str(e)}")
            return cls.ADD


class ConfidenceLevel(str, Enum):
    """Overall confidence levels for agent recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        """Convert confidence score to confidence level"""
        try:
            if score >= 0.8:
                return cls.HIGH
            elif score >= 0.6:
                return cls.MEDIUM
            else:
                return cls.LOW
        except Exception:
            return cls.LOW


class AgentStatus(str, Enum):
    """Status of individual agents in the processing pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== UTILITY FUNCTIONS ====================

def safe_string_list(value: Any) -> List[str]:
    """Safely convert various input types to list of strings"""
    try:
        if value is None:
            return []
        
        if isinstance(value, str):
            # Handle comma-separated strings
            if ',' in value:
                return [item.strip() for item in value.split(',') if item.strip()]
            return [value.strip()] if value.strip() else []
        
        if isinstance(value, list):
            result = []
            for item in value:
                if item is not None:
                    str_item = str(item).strip()
                    if str_item:
                        result.append(str_item)
            return result
        
        # Try to convert other types to string
        str_value = str(value).strip()
        return [str_value] if str_value else []
        
    except Exception as e:
        logger.warning(f"Error converting to string list: {str(e)}")
        return []


def validate_business_section_name(section_name: str) -> bool:
    """Validate that a section name is valid for business model canvas"""
    valid_sections = {
        'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
        'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
    }
    return section_name in valid_sections


def clean_html_content(text: str) -> str:
    """Clean HTML content and prevent XSS"""
    try:
        if not text:
            return ""
        
        # Basic HTML entity encoding
        cleaned = str(text).replace('<', '&lt;').replace('>', '&gt;')
        cleaned = cleaned.replace('"', '&quot;').replace("'", '&#x27;')
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    except Exception as e:
        logger.warning(f"Error cleaning HTML content: {str(e)}")
        return str(text) if text else ""


# ==================== CORE BUSINESS MODELS ====================

class CompletedAction(BaseModel):
    """Model representing a completed business action with comprehensive validation."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the action")
    title: str = Field(..., min_length=3, max_length=500, description="Descriptive name of the action")
    description: str = Field(..., min_length=10, max_length=5000, description="Detailed explanation of what was done")
    outcome: ActionOutcome = Field(..., description="The result outcome of the action")
    results_data: str = Field(..., min_length=10, max_length=50000, description="Detailed results and metrics from the action")
    completion_date: datetime = Field(default_factory=datetime.now, description="When the action was completed")
    success_metrics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative metrics related to success")
    action_category: str = Field(default="General", max_length=100, description="Category of action")
    stakeholders_involved: List[str] = Field(default_factory=list, description="List of stakeholders involved in the action")
    budget_spent: Optional[float] = Field(None, ge=0, le=1000000000, description="Budget spent on this action")
    duration_days: Optional[int] = Field(None, ge=1, le=3650, description="Duration of the action in days")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True
        
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        """Validate title contains meaningful content and is safe"""
        try:
            if not v or not v.strip():
                raise ValueError('Title cannot be empty or whitespace only')
            
            cleaned_title = clean_html_content(v.strip())
            
            # Check for minimum meaningful content
            if len(cleaned_title) < 3:
                raise ValueError('Title must be at least 3 characters long')
            
            return cleaned_title
        except Exception as e:
            logger.error(f"Title validation error: {str(e)}")
            raise ValueError(f"Invalid title: {str(e)}")

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        """Validate description contains substantial information"""
        try:
            if not v or not v.strip():
                raise ValueError('Description cannot be empty')
            
            cleaned_desc = clean_html_content(v.strip())
            
            if len(cleaned_desc) < 10:
                raise ValueError('Description must contain at least 10 characters of meaningful content')
            
            return cleaned_desc
        except Exception as e:
            logger.error(f"Description validation error: {str(e)}")
            raise ValueError(f"Invalid description: {str(e)}")

    @field_validator('results_data')
    @classmethod
    def validate_results_data(cls, v):
        """Validate results data contains substantial information"""
        try:
            if not v or not v.strip():
                raise ValueError('Results data cannot be empty')
            
            cleaned_results = clean_html_content(v.strip())
            
            if len(cleaned_results) < 10:
                raise ValueError('Results data must contain at least 10 characters of meaningful content')
            
            return cleaned_results
        except Exception as e:
            logger.error(f"Results data validation error: {str(e)}")
            raise ValueError(f"Invalid results data: {str(e)}")

    @field_validator('outcome', mode='before')
    @classmethod
    def validate_outcome(cls, v):
        """Validate and convert outcome to proper enum"""
        try:
            return ActionOutcome.from_string(v)
        except Exception as e:
            logger.error(f"Outcome validation error: {str(e)}")
            return ActionOutcome.INCONCLUSIVE

    @field_validator('stakeholders_involved', mode='before')
    @classmethod
    def validate_stakeholders(cls, v):
        """Validate and clean stakeholders list"""
        try:
            return safe_string_list(v)
        except Exception as e:
            logger.warning(f"Stakeholders validation error: {str(e)}")
            return []

    @field_validator('success_metrics')
    @classmethod
    def validate_success_metrics(cls, v):
        """Validate success metrics are reasonable"""
        try:
            if not isinstance(v, dict):
                return {}
            
            # Clean and validate metrics
            cleaned_metrics = {}
            for key, value in v.items():
                try:
                    clean_key = clean_html_content(str(key))
                    if clean_key and value is not None:
                        cleaned_metrics[clean_key] = value
                except Exception:
                    continue
            
            return cleaned_metrics
        except Exception as e:
            logger.warning(f"Success metrics validation error: {str(e)}")
            return {}

    def get_action_summary(self) -> str:
        """Get a brief summary of the action with error handling"""
        try:
            date_str = self.completion_date.strftime('%Y-%m-%d') if self.completion_date else 'Unknown date'
            return f"{self.title} ({self.outcome.value}) - completed {date_str}"
        except Exception as e:
            logger.warning(f"Error generating action summary: {str(e)}")
            return f"Action {self.id}"

    def calculate_roi(self) -> Optional[float]:
        """Calculate ROI if sufficient data is available with error handling"""
        try:
            if not self.budget_spent or not self.success_metrics:
                return None
            
            revenue_generated = self.success_metrics.get('revenue_generated', 0)
            if isinstance(revenue_generated, (int, float)) and revenue_generated > 0 and self.budget_spent > 0:
                return (revenue_generated - self.budget_spent) / self.budget_spent
            
            return None
        except Exception as e:
            logger.warning(f"Error calculating ROI: {str(e)}")
            return None


class BusinessModelCanvas(BaseModel):
    """Model representing the 9 sections of a Business Model Canvas with comprehensive validation."""

    customer_segments: List[str] = Field(default_factory=list, description="Target customer groups")
    value_propositions: List[str] = Field(default_factory=list, description="Bundle of products/services that create value")
    channels: List[str] = Field(default_factory=list, description="How the company communicates with customers")
    customer_relationships: List[str] = Field(default_factory=list, description="Types of relationships established")
    revenue_streams: List[str] = Field(default_factory=list, description="Cash generated from each customer segment")
    key_resources: List[str] = Field(default_factory=list, description="Most important assets required")
    key_activities: List[str] = Field(default_factory=list, description="Most important activities to make business model work")
    key_partnerships: List[str] = Field(default_factory=list, description="Network of suppliers and partners")
    cost_structure: List[str] = Field(default_factory=list, description="All costs incurred to operate")
    last_updated: datetime = Field(default_factory=datetime.now, description="Timestamp of last modification")
    version: str = Field(default="1.0.0", description="Version number for change tracking")
    created_by: Optional[str] = Field(None, max_length=200, description="User or system that created this BMC")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization and search")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @field_validator('customer_segments', 'value_propositions', 'channels', 'customer_relationships',
              'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 
              'cost_structure', 'tags', mode='before')
    @classmethod
    def validate_string_lists(cls, v, info):
        """Validate and clean all string list fields"""
        try:
            cleaned_list = safe_string_list(v)
            
            # Additional validation for specific fields
            if info.field_name in ['revenue_streams', 'cost_structure']:
                # Financial sections should not have empty items
                cleaned_list = [item for item in cleaned_list if item and len(item.strip()) > 5]
            
            # Clean HTML content in all items
            cleaned_list = [clean_html_content(item) for item in cleaned_list]
            
            # Remove duplicates while preserving order
            seen = set()
            deduped_list = []
            for item in cleaned_list:
                if item and item not in seen:
                    seen.add(item)
                    deduped_list.append(item)
            
            return deduped_list
        except Exception as e:
            logger.warning(f"Error validating {info.field_name}: {str(e)}")
            return []

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Validate version format"""
        try:
            if not v:
                return "1.0.0"
            
            # Basic semantic version validation
            version_pattern = r'^\d+\.\d+\.\d+

    def get_section_by_name(self, section_name: str) -> List[str]:
        """Get a canvas section by its name with error handling"""
        try:
            if not validate_business_section_name(section_name):
                logger.warning(f"Invalid section name: {section_name}")
                return []
            
            return getattr(self, section_name, [])
        except Exception as e:
            logger.error(f"Error getting section {section_name}: {str(e)}")
            return []

    def update_section(self, section_name: str, new_values: List[str]) -> bool:
        """Update a canvas section with new values"""
        try:
            if not validate_business_section_name(section_name):
                logger.error(f"Invalid section name: {section_name}")
                return False
            
            if not hasattr(self, section_name):
                logger.error(f"Section {section_name} does not exist")
                return False
            
            # Validate and clean new values
            cleaned_values = safe_string_list(new_values)
            cleaned_values = [clean_html_content(item) for item in cleaned_values]
            
            setattr(self, section_name, cleaned_values)
            self.last_updated = datetime.now()
            self._increment_version()
            return True
            
        except Exception as e:
            logger.error(f"Error updating section {section_name}: {str(e)}")
            return False

    def _increment_version(self) -> None:
        """Increment the patch version number with error handling"""
        try:
            parts = self.version.split('.')
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                self.version = f"{major}.{minor}.{patch + 1}"
            else:
                self.version = "1.0.1"
        except Exception as e:
            logger.warning(f"Error incrementing version: {str(e)}")
            self.version = "1.0.1"

    def get_total_elements(self) -> int:
        """Get total number of elements across all sections"""
        try:
            sections = [
                'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
            ]
            return sum(len(getattr(self, section, [])) for section in sections)
        except Exception as e:
            logger.error(f"Error calculating total elements: {str(e)}")
            return 0

    def get_empty_sections(self) -> List[str]:
        """Get list of empty sections"""
        try:
            sections = [
                'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
            ]
            return [section for section in sections if not getattr(self, section, [])]
        except Exception as e:
            logger.error(f"Error finding empty sections: {str(e)}")
            return []

    def get_completeness_score(self) -> float:
        """Calculate completeness score (0.0 to 1.0) with error handling"""
        try:
            empty_sections = len(self.get_empty_sections())
            return max(0.0, (9 - empty_sections) / 9.0)
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.0

    def get_section_quality_score(self, section_name: str) -> float:
        """Get quality score for a specific section based on content"""
        try:
            if not validate_business_section_name(section_name):
                return 0.0
            
            values = getattr(self, section_name, [])
            if not values:
                return 0.0

            # Base score from having content
            quality_score = 0.3
            
            # Length quality
            total_length = sum(len(str(item)) for item in values)
            avg_length = total_length / len(values) if values else 0
            
            if avg_length > 20:
                quality_score += 0.3
            if avg_length > 50:
                quality_score += 0.2

            # Content quality - avoid generic terms
            generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo'}
            non_generic = sum(1 for item in values if str(item).lower().strip() not in generic_terms)
            if values:
                quality_score += (non_generic / len(values)) * 0.2

            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating section quality score: {str(e)}")
            return 0.0

    def validate_integrity(self) -> List[str]:
        """Validate BMC integrity and return list of issues"""
        try:
            issues = []
            
            # Check for empty sections
            empty_sections = self.get_empty_sections()
            if len(empty_sections) > 5:  # More than half empty
                issues.append(f"Too many empty sections: {', '.join(empty_sections)}")
            
            # Check for very short descriptions
            all_items = []
            sections = ['customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                       'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure']
            
            for section in sections:
                try:
                    items = getattr(self, section, [])
                    all_items.extend(items)
                except Exception:
                    continue
            
            if all_items:
                short_items = [item for item in all_items if len(str(item).strip()) < 10]
                if len(short_items) > len(all_items) * 0.4:  # More than 40% short items
                    issues.append(f"Many items are too brief (< 10 characters): {len(short_items)} items")
                
                # Check for placeholder content
                placeholder_patterns = ['tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo']
                placeholder_items = [item for item in all_items 
                                   if any(pattern in str(item).lower() for pattern in placeholder_patterns)]
                if len(placeholder_items) > 0:
                    issues.append(f"{len(placeholder_items)} placeholder items found")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating BMC integrity: {str(e)}")
            return [f"Validation error: {str(e)}"]


class ProposedChange(BaseModel):
    """Model representing a proposed change to a business model canvas with comprehensive validation."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the change")
    canvas_section: str = Field(..., description="Which BMC section to update")
    change_type: ChangeType = Field(..., description="Type of change to make")
    current_value: Optional[str] = Field(None, max_length=5000, description="Current value being changed")
    proposed_value: str = Field(..., min_length=1, max_length=5000, description="New value to add or modify to")
    reasoning: str = Field(..., min_length=15, max_length=10000, description="AI explanation for the change")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    impact_assessment: str = Field(default="medium", description="Expected impact level")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    validation_suggestions: List[str] = Field(default_factory=list, description="Suggested validation methods")
    estimated_effort: Optional[str] = Field(None, max_length=200, description="Estimated effort to implement")
    dependencies: List[str] = Field(default_factory=list, description="Other changes this depends on")
    created_at: datetime = Field(default_factory=datetime.now, description="When this change was proposed")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @field_validator('canvas_section')
    @classmethod
    def validate_canvas_section(cls, v):
        """Validate that the canvas section is valid"""
        try:
            if not validate_business_section_name(v):
                raise ValueError(f"Invalid canvas section: {v}")
            return v
        except Exception as e:
            logger.error(f"Canvas section validation error: {str(e)}")
            raise ValueError(f"Invalid canvas section: {str(e)}")

    @field_validator('change_type', mode='before')
    @classmethod
    def validate_change_type(cls, v):
        """Validate and convert change type"""
        try:
            return ChangeType.from_string(v)
        except Exception as e:
            logger.error(f"Change type validation error: {str(e)}")
            return ChangeType.ADD

    @field_validator('current_value', 'proposed_value', mode='before')
    @classmethod
    def validate_text_fields(cls, v, info):
        """Validate and clean text fields"""
        try:
            if v is None:
                return None if info.field_name == 'current_value' else ""
            
            cleaned = clean_html_content(str(v).strip())
            
            if info.field_name == 'proposed_value' and not cleaned:
                raise ValueError("Proposed value cannot be empty")
            
            return cleaned
        except Exception as e:
            logger.error(f"Text field validation error for {info.field_name}: {str(e)}")
            if info.field_name == 'proposed_value':
                raise ValueError(f"Invalid proposed value: {str(e)}")
            return None

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning_quality(cls, v):
        """Validate reasoning contains substantial explanation"""
        try:
            if not v or not v.strip():
                raise ValueError("Reasoning cannot be empty")
            
            cleaned = clean_html_content(v.strip())
            
            if len(cleaned) < 15:
                raise ValueError("Reasoning must contain at least 15 characters")
            
            # Check for evidence-based language indicators
            evidence_indicators = ['data shows', 'research indicates', 'analysis suggests', 
                                 'evidence', 'based on', 'results show', 'findings']
            has_evidence = any(indicator in cleaned.lower() for indicator in evidence_indicators)
            
            if not has_evidence and len(cleaned) < 50:
                logger.warning("Reasoning lacks evidence-based language")
            
            return cleaned
        except Exception as e:
            logger.error(f"Reasoning validation error: {str(e)}")
            raise ValueError(f"Invalid reasoning: {str(e)}")

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence_calibration(cls, v):
        """Ensure confidence score is properly calibrated"""
        try:
            score = float(v)
            if score < 0.1:
                logger.warning("Confidence score very low, adjusting to 0.1")
                return 0.1
            if score > 0.99:
                logger.warning("Confidence score very high, adjusting to 0.99")
                return 0.99
            return score
        except Exception as e:
            logger.error(f"Confidence score validation error: {str(e)}")
            return 0.5  # Default to medium confidence

    @field_validator('impact_assessment')
    @classmethod
    def validate_impact_assessment(cls, v):
        """Validate impact assessment value"""
        try:
            valid_impacts = {'low', 'medium', 'high'}
            normalized = str(v).lower().strip()
            if normalized in valid_impacts:
                return normalized
            else:
                logger.warning(f"Invalid impact assessment '{v}', defaulting to 'medium'")
                return "medium"
        except Exception as e:
            logger.warning(f"Impact assessment validation error: {str(e)}")
            return "medium"

    @field_validator('risk_factors', 'validation_suggestions', 'dependencies', mode='before')
    @classmethod
    def validate_string_lists(cls, v):
        """Validate and clean string list fields"""
        try:
            return safe_string_list(v)
        except Exception as e:
            logger.warning(f"String list validation error: {str(e)}")
            return []

    def get_confidence_category(self) -> ConfidenceLevel:
        """Get confidence category based on score"""
        try:
            return ConfidenceLevel.from_score(self.confidence_score)
        except Exception as e:
            logger.error(f"Error getting confidence category: {str(e)}")
            return ConfidenceLevel.LOW

    def is_safe_for_auto_application(self) -> bool:
        """Determine if this change is safe for automatic application"""
        try:
            # High confidence threshold for auto-application
            if self.confidence_score < 0.8:
                return False
            
            # No removal operations in auto-mode (too risky)
            if self.change_type == ChangeType.REMOVE:
                return False
            
            # Critical financial sections need manual approval
            critical_sections = {'revenue_streams', 'cost_structure'}
            if self.canvas_section in critical_sections:
                return False
            
            # Check for high-risk indicators
            high_risk_indicators = ['major', 'significant', 'fundamental', 'critical', 'breaking', 'removing']
            if any(indicator in self.reasoning.lower() for indicator in high_risk_indicators):
                return False
            
            # Check proposed value quality
            if not self.proposed_value or len(self.proposed_value.strip()) < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking auto-application safety: {str(e)}")
            return False

    def get_estimated_business_impact(self) -> str:
        """Estimate business impact based on section and change type"""
        try:
            high_impact_sections = {'value_propositions', 'customer_segments', 'revenue_streams'}
            
            if self.canvas_section in high_impact_sections:
                if self.change_type == ChangeType.REMOVE:
                    return "high"
                elif self.confidence_score >= 0.8:
                    return "medium"
                else:
                    return "high"  # Low confidence on high-impact sections
            
            if self.change_type == ChangeType.REMOVE:
                return "medium"
            
            return "low"
        except Exception as e:
            logger.error(f"Error estimating business impact: {str(e)}")
            return "medium"

    def get_change_hash(self) -> str:
        """Get unique hash for this change to detect duplicates"""
        try:
            import hashlib
            change_string = f"{self.canvas_section}_{self.change_type.value}_{self.proposed_value}"
            return hashlib.md5(change_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating change hash: {str(e)}")
            return str(hash(str(self)))


class AgentRecommendation(BaseModel):
    """Model representing the complete recommendation from the agentic system."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    proposed_changes: List[ProposedChange] = Field(default_factory=list, description="List of proposed changes")
    next_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions")
    reasoning: str = Field(..., min_length=10, max_length=10000, description="Overall AI logic and rationale")
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence in recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="When recommendations were generated")
    processing_time_ms: Optional[int] = Field(None, ge=0, le=600000, description="Processing time in milliseconds")
    model_version: str = Field(default="gemini-2.0-flash", max_length=100, description="AI model version used")
    source_action_id: Optional[str] = Field(None, max_length=100, description="ID of triggering action")
    risk_assessment: str = Field(default="medium", description="Overall risk assessment")
    implementation_priority: str = Field(default="medium", description="Implementation priority")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        """Validate reasoning content"""
        try:
            if not v or not v.strip():
                return "Analysis completed with limited reasoning available"
            
            cleaned = clean_html_content(v.strip())
            
            if len(cleaned) < 10:
                return "Analysis completed with minimal reasoning"
            
            return cleaned
        except Exception as e:
            logger.warning(f"Reasoning validation error: {str(e)}")
            return "Analysis completed with validation errors"

    @field_validator('confidence_level', mode='before')
    @classmethod
    def validate_confidence_level(cls, v):
        """Validate confidence level"""
        try:
            if isinstance(v, ConfidenceLevel):
                return v
            return ConfidenceLevel(str(v).lower().strip())
        except Exception as e:
            logger.warning(f"Confidence level validation error: {str(e)}")
            return ConfidenceLevel.LOW

    @field_validator('next_actions', mode='before')
    @classmethod
    def validate_next_actions(cls, v):
        """Validate and clean next actions"""
        try:
            actions = safe_string_list(v)
            # Clean and validate each action
            cleaned_actions = []
            for action in actions:
                cleaned = clean_html_content(action)
                if cleaned and len(cleaned) > 5:  # Minimum meaningful length
                    cleaned_actions.append(cleaned)
            
            # If no valid actions, provide default
            if not cleaned_actions:
                cleaned_actions = ["Review the analysis and determine next steps"]
            
            return cleaned_actions
        except Exception as e:
            logger.warning(f"Next actions validation error: {str(e)}")
            return ["Review the analysis and determine next steps"]

    @field_validator('risk_assessment', 'implementation_priority')
    @classmethod
    def validate_level_fields(cls, v, info):
        """Validate level fields (risk_assessment, implementation_priority)"""
        try:
            valid_levels = {'low', 'medium', 'high'}
            normalized = str(v).lower().strip()
            if normalized in valid_levels:
                return normalized
            else:
                logger.warning(f"Invalid {info.field_name} '{v}', defaulting to 'medium'")
                return "medium"
        except Exception as e:
            logger.warning(f"{info.field_name} validation error: {str(e)}")
            return "medium"

    def get_high_confidence_changes(self) -> List[ProposedChange]:
        """Return only changes with high confidence scores (>= 0.8)"""
        try:
            return [change for change in self.proposed_changes if change.confidence_score >= 0.8]
        except Exception as e:
            logger.error(f"Error filtering high confidence changes: {str(e)}")
            return []

    def get_changes_by_section(self, section_name: str) -> List[ProposedChange]:
        """Return changes for a specific canvas section"""
        try:
            if not validate_business_section_name(section_name):
                return []
            return [change for change in self.proposed_changes if change.canvas_section == section_name]
        except Exception as e:
            logger.error(f"Error filtering changes by section: {str(e)}")
            return []

    def get_safe_auto_changes(self) -> List[ProposedChange]:
        """Return changes that are safe for automatic application"""
        try:
            return [change for change in self.proposed_changes if change.is_safe_for_auto_application()]
        except Exception as e:
            logger.error(f"Error filtering safe auto changes: {str(e)}")
            return []

    def get_sections_affected(self) -> List[str]:
        """Get list of BMC sections that would be affected by these changes"""
        try:
            sections = set()
            for change in self.proposed_changes:
                try:
                    sections.add(change.canvas_section)
                except Exception:
                    continue
            return list(sections)
        except Exception as e:
            logger.error(f"Error getting affected sections: {str(e)}")
            return []

    def calculate_overall_impact_score(self) -> float:
        """Calculate an overall impact score for these recommendations"""
        try:
            if not self.proposed_changes:
                return 0.0
            
            impact_weights = {'low': 1, 'medium': 2, 'high': 3}
            total_impact = 0.0
            
            for change in self.proposed_changes:
                try:
                    impact = change.get_estimated_business_impact()
                    weight = impact_weights.get(impact, 1)
                    total_impact += weight * change.confidence_score
                except Exception:
                    continue
            
            if self.proposed_changes:
                return min(total_impact / len(self.proposed_changes), 3.0) / 3.0  # Normalize to 0-1
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating impact score: {str(e)}")
            return 0.0

    def get_implementation_timeline(self) -> str:
        """Suggest implementation timeline based on changes complexity"""
        try:
            change_count = len(self.proposed_changes)
            high_impact_count = len([c for c in self.proposed_changes 
                                   if c.get_estimated_business_impact() == "high"])
            
            if change_count == 0:
                return "No implementation needed"
            elif change_count <= 2 and high_impact_count == 0:
                return "1-2 weeks"
            elif change_count <= 4 and high_impact_count <= 1:
                return "2-3 weeks"
            else:
                return "1 month with phased approach"
        except Exception as e:
            logger.error(f"Error calculating implementation timeline: {str(e)}")
            return "Timeline assessment failed"

    def has_auto_applicable_changes(self) -> bool:
        """Check if there are any changes safe for auto-application"""
        try:
            return len(self.get_safe_auto_changes()) > 0
        except Exception as e:
            logger.error(f"Error checking auto-applicable changes: {str(e)}")
            return False


class ChangeHistory(BaseModel):
    """Model for tracking the history of changes made to business models."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the change was made")
    trigger_action_id: str = Field(..., max_length=200, description="ID of triggering action")
    changes_applied: List[ProposedChange] = Field(..., description="List of changes that were applied")
    previous_state_snapshot: Dict[str, Any] = Field(..., description="BMC state before changes")
    new_state_snapshot: Dict[str, Any] = Field(..., description="BMC state after changes")
    auto_applied: bool = Field(False, description="Whether changes were applied automatically")
    applied_by: Optional[str] = Field(None, max_length=200, description="User or system that applied changes")
    rollback_data: Optional[Dict[str, Any]] = Field(None, description="Data needed to rollback")
    notes: Optional[str] = Field(None, max_length=5000, description="Additional notes")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @field_validator('trigger_action_id')
    @classmethod
    def validate_trigger_action_id(cls, v):
        """Validate trigger action ID"""
        try:
            cleaned = clean_html_content(str(v).strip())
            return cleaned if cleaned else "unknown"
        except Exception:
            return "unknown"

    @field_validator('notes', mode='before')
    @classmethod
    def validate_notes(cls, v):
        """Validate and clean notes"""
        try:
            if v is None:
                return None
            return clean_html_content(str(v).strip())[:5000]  # Truncate to max length
        except Exception:
            return None

    @field_validator('applied_by', mode='before')
    @classmethod
    def validate_applied_by(cls, v):
        """Validate and clean applied_by field"""
        try:
            if v is None:
                return None
            return clean_html_content(str(v).strip())[:200]  # Truncate to max length
        except Exception:
            return None

    def get_changes_summary(self) -> Dict[str, int]:
        """Get summary of changes by type and section"""
        try:
            summary = {
                "total_changes": len(self.changes_applied),
                "sections_affected": 0,
                "add_operations": 0,
                "modify_operations": 0,
                "remove_operations": 0,
            }
            
            sections_set = set()
            for change in self.changes_applied:
                try:
                    sections_set.add(change.canvas_section)
                    
                    if change.change_type == ChangeType.ADD:
                        summary["add_operations"] += 1
                    elif change.change_type == ChangeType.MODIFY:
                        summary["modify_operations"] += 1
                    elif change.change_type == ChangeType.REMOVE:
                        summary["remove_operations"] += 1
                except Exception:
                    continue
            
            summary["sections_affected"] = len(sections_set)
            return summary
        except Exception as e:
            logger.error(f"Error generating changes summary: {str(e)}")
            return {"total_changes": 0, "sections_affected": 0, "add_operations": 0, 
                   "modify_operations": 0, "remove_operations": 0}

    def calculate_change_impact(self) -> float:
        """Calculate the overall impact of these changes"""
        try:
            if not self.changes_applied:
                return 0.0
            
            total_impact = sum(change.confidence_score for change in self.changes_applied)
            return total_impact / len(self.changes_applied)
        except Exception as e:
            logger.error(f"Error calculating change impact: {str(e)}")
            return 0.0

    def can_rollback(self) -> bool:
        """Check if these changes can be rolled back"""
        try:
            return bool(self.previous_state_snapshot)
        except Exception as e:
            logger.error(f"Error checking rollback capability: {str(e)}")
            return False

    def get_rollback_summary(self) -> str:
        """Get summary for rollback confirmation"""
        try:
            change_count = len(self.changes_applied)
            timestamp = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "Unknown time"
            mode = "Auto" if self.auto_applied else "Manual"
            
            return f"Rollback {change_count} {mode.lower()} change(s) from {timestamp}?"
        except Exception as e:
            logger.error(f"Error generating rollback summary: {str(e)}")
            return "Rollback changes?"


# ==================== ANALYTICS AND REPORTING ====================

class AnalyticsSummary(BaseModel):
    """Model for tracking system usage and performance analytics."""

    total_actions_processed: int = Field(default=0, ge=0, description="Total actions processed")
    total_changes_applied: int = Field(default=0, ge=0, description="Total changes applied")
    auto_mode_usage_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Auto-mode usage rate")
    average_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average confidence score")
    most_updated_sections: List[str] = Field(default_factory=list, description="Most frequently updated sections")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate percentage")
    generated_at: datetime = Field(default_factory=datetime.now, description="When analytics were generated")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @field_validator('most_updated_sections', mode='before')
    @classmethod
    def validate_most_updated_sections(cls, v):
        """Validate most updated sections"""
        try:
            sections = safe_string_list(v)
            valid_sections = [s for s in sections if validate_business_section_name(s)]
            return valid_sections
        except Exception as e:
            logger.warning(f"Most updated sections validation error: {str(e)}")
            return []


# ==================== VALIDATION UTILITIES ====================

def validate_bmc_change_compatibility(change: ProposedChange, current_bmc: BusinessModelCanvas) -> List[str]:
    """Validate that a proposed change is compatible with current BMC state"""
    try:
        issues = []
        
        if change is None or current_bmc is None:
            issues.append("Cannot validate None change or BMC")
            return issues
        
        # Check if section exists
        if not hasattr(current_bmc, change.canvas_section):
            issues.append(f"Invalid section: {change.canvas_section}")
            return issues
        
        current_values = getattr(current_bmc, change.canvas_section, [])
        
        # Check MODIFY and REMOVE operations have valid current_value
        if change.change_type in [ChangeType.MODIFY, ChangeType.REMOVE]:
            if not change.current_value:
                issues.append("MODIFY/REMOVE operations require current_value")
            elif change.current_value not in current_values:
                issues.append(f"Current value '{change.current_value}' not found in {change.canvas_section}")
        
        # Check for duplicate ADD operations
        if change.change_type == ChangeType.ADD:
            if change.proposed_value in current_values:
                issues.append(f"Value '{change.proposed_value}' already exists in {change.canvas_section}")
        
        return issues
        
    except Exception as e:
        logger.error(f"Error validating change compatibility: {str(e)}")
        return [f"Validation error: {str(e)}"]


# Export all models for easy importing
__all__ = [
    # Enums
    'ActionOutcome', 'ChangeType', 'ConfidenceLevel', 'AgentStatus',
    
    # Core Business Models
    'CompletedAction', 'BusinessModelCanvas', 'ProposedChange',
    'AgentRecommendation', 'ChangeHistory',
    
    # Analytics Models
    'AnalyticsSummary',
    
    # Utility Functions
    'validate_bmc_change_compatibility', 'validate_business_section_name',
    'safe_string_list', 'clean_html_content'
]
            if re.match(version_pattern, str(v).strip()):
                return str(v).strip()
            else:
                logger.warning(f"Invalid version format '{v}', using default")
                return "1.0.0"
        except Exception as e:
            logger.warning(f"Version validation error: {str(e)}")
            return "1.0.0"

    @field_validator('created_by', mode='before')
    @classmethod
    def validate_created_by(cls, v):
        """Validate and clean created_by field"""
        try:
            if v is None:
                return None
            return clean_html_content(str(v).strip())[:200]  # Truncate to max length
        except Exception:
            return None

    @model_validator(mode='after')
    def validate_bmc_integrity(self):
        """Validate overall BMC integrity"""
        try:
            # Ensure at least some content exists
            section_fields = [
                'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
            ]
            
            total_items = sum(len(getattr(self, field, [])) for field in section_fields)
            
            if total_items == 0:
                logger.warning("BMC has no content in any section")
                # Add default content to prevent completely empty BMC
                self.value_propositions = ["Core value proposition to be defined"]
                self.customer_segments = ["Target customer segment to be defined"]
            
            return self
        except Exception as e:
            logger.error(f"BMC integrity validation error: {str(e)}")
            return self

    def get_section_by_name(self, section_name: str) -> List[str]:
        """Get a canvas section by its name with error handling"""
        try:
            if not validate_business_section_name(section_name):
                logger.warning(f"Invalid section name: {section_name}")
                return []
            
            return getattr(self, section_name, [])
        except Exception as e:
            logger.error(f"Error getting section {section_name}: {str(e)}")
            return []

    def update_section(self, section_name: str, new_values: List[str]) -> bool:
        """Update a canvas section with new values"""
        try:
            if not validate_business_section_name(section_name):
                logger.error(f"Invalid section name: {section_name}")
                return False
            
            if not hasattr(self, section_name):
                logger.error(f"Section {section_name} does not exist")
                return False
            
            # Validate and clean new values
            cleaned_values = safe_string_list(new_values)
            cleaned_values = [clean_html_content(item) for item in cleaned_values]
            
            setattr(self, section_name, cleaned_values)
            self.last_updated = datetime.now()
            self._increment_version()
            return True
            
        except Exception as e:
            logger.error(f"Error updating section {section_name}: {str(e)}")
            return False

    def _increment_version(self) -> None:
        """Increment the patch version number with error handling"""
        try:
            parts = self.version.split('.')
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                self.version = f"{major}.{minor}.{patch + 1}"
            else:
                self.version = "1.0.1"
        except Exception as e:
            logger.warning(f"Error incrementing version: {str(e)}")
            self.version = "1.0.1"

    def get_total_elements(self) -> int:
        """Get total number of elements across all sections"""
        try:
            sections = [
                'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
            ]
            return sum(len(getattr(self, section, [])) for section in sections)
        except Exception as e:
            logger.error(f"Error calculating total elements: {str(e)}")
            return 0

    def get_empty_sections(self) -> List[str]:
        """Get list of empty sections"""
        try:
            sections = [
                'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
            ]
            return [section for section in sections if not getattr(self, section, [])]
        except Exception as e:
            logger.error(f"Error finding empty sections: {str(e)}")
            return []

    def get_completeness_score(self) -> float:
        """Calculate completeness score (0.0 to 1.0) with error handling"""
        try:
            empty_sections = len(self.get_empty_sections())
            return max(0.0, (9 - empty_sections) / 9.0)
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.0

    def get_section_quality_score(self, section_name: str) -> float:
        """Get quality score for a specific section based on content"""
        try:
            if not validate_business_section_name(section_name):
                return 0.0
            
            values = getattr(self, section_name, [])
            if not values:
                return 0.0

            # Base score from having content
            quality_score = 0.3
            
            # Length quality
            total_length = sum(len(str(item)) for item in values)
            avg_length = total_length / len(values) if values else 0
            
            if avg_length > 20:
                quality_score += 0.3
            if avg_length > 50:
                quality_score += 0.2

            # Content quality - avoid generic terms
            generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo'}
            non_generic = sum(1 for item in values if str(item).lower().strip() not in generic_terms)
            if values:
                quality_score += (non_generic / len(values)) * 0.2

            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating section quality score: {str(e)}")
            return 0.0

    def validate_integrity(self) -> List[str]:
        """Validate BMC integrity and return list of issues"""
        try:
            issues = []
            
            # Check for empty sections
            empty_sections = self.get_empty_sections()
            if len(empty_sections) > 5:  # More than half empty
                issues.append(f"Too many empty sections: {', '.join(empty_sections)}")
            
            # Check for very short descriptions
            all_items = []
            sections = ['customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                       'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure']
            
            for section in sections:
                try:
                    items = getattr(self, section, [])
                    all_items.extend(items)
                except Exception:
                    continue
            
            if all_items:
                short_items = [item for item in all_items if len(str(item).strip()) < 10]
                if len(short_items) > len(all_items) * 0.4:  # More than 40% short items
                    issues.append(f"Many items are too brief (< 10 characters): {len(short_items)} items")
                
                # Check for placeholder content
                placeholder_patterns = ['tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo']
                placeholder_items = [item for item in all_items 
                                   if any(pattern in str(item).lower() for pattern in placeholder_patterns)]
                if len(placeholder_items) > 0:
                    issues.append(f"{len(placeholder_items)} placeholder items found")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating BMC integrity: {str(e)}")
            return [f"Validation error: {str(e)}"]


class ProposedChange(BaseModel):
    """Model representing a proposed change to a business model canvas with comprehensive validation."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the change")
    canvas_section: str = Field(..., description="Which BMC section to update")
    change_type: ChangeType = Field(..., description="Type of change to make")
    current_value: Optional[str] = Field(None, max_length=5000, description="Current value being changed")
    proposed_value: str = Field(..., min_length=1, max_length=5000, description="New value to add or modify to")
    reasoning: str = Field(..., min_length=15, max_length=10000, description="AI explanation for the change")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    impact_assessment: str = Field(default="medium", description="Expected impact level")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    validation_suggestions: List[str] = Field(default_factory=list, description="Suggested validation methods")
    estimated_effort: Optional[str] = Field(None, max_length=200, description="Estimated effort to implement")
    dependencies: List[str] = Field(default_factory=list, description="Other changes this depends on")
    created_at: datetime = Field(default_factory=datetime.now, description="When this change was proposed")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @validator('canvas_section')
    def validate_canvas_section(cls, v):
        """Validate that the canvas section is valid"""
        try:
            if not validate_business_section_name(v):
                raise ValueError(f"Invalid canvas section: {v}")
            return v
        except Exception as e:
            logger.error(f"Canvas section validation error: {str(e)}")
            raise ValueError(f"Invalid canvas section: {str(e)}")

    @validator('change_type', pre=True)
    def validate_change_type(cls, v):
        """Validate and convert change type"""
        try:
            return ChangeType.from_string(v)
        except Exception as e:
            logger.error(f"Change type validation error: {str(e)}")
            return ChangeType.ADD

    @validator('current_value', 'proposed_value', pre=True)
    def validate_text_fields(cls, v, field):
        """Validate and clean text fields"""
        try:
            if v is None:
                return None if field.name == 'current_value' else ""
            
            cleaned = clean_html_content(str(v).strip())
            
            if field.name == 'proposed_value' and not cleaned:
                raise ValueError("Proposed value cannot be empty")
            
            return cleaned
        except Exception as e:
            logger.error(f"Text field validation error for {field.name}: {str(e)}")
            if field.name == 'proposed_value':
                raise ValueError(f"Invalid proposed value: {str(e)}")
            return None

    @validator('reasoning')
    def validate_reasoning_quality(cls, v):
        """Validate reasoning contains substantial explanation"""
        try:
            if not v or not v.strip():
                raise ValueError("Reasoning cannot be empty")
            
            cleaned = clean_html_content(v.strip())
            
            if len(cleaned) < 15:
                raise ValueError("Reasoning must contain at least 15 characters")
            
            # Check for evidence-based language indicators
            evidence_indicators = ['data shows', 'research indicates', 'analysis suggests', 
                                 'evidence', 'based on', 'results show', 'findings']
            has_evidence = any(indicator in cleaned.lower() for indicator in evidence_indicators)
            
            if not has_evidence and len(cleaned) < 50:
                logger.warning("Reasoning lacks evidence-based language")
            
            return cleaned
        except Exception as e:
            logger.error(f"Reasoning validation error: {str(e)}")
            raise ValueError(f"Invalid reasoning: {str(e)}")

    @validator('confidence_score')
    def validate_confidence_calibration(cls, v):
        """Ensure confidence score is properly calibrated"""
        try:
            score = float(v)
            if score < 0.1:
                logger.warning("Confidence score very low, adjusting to 0.1")
                return 0.1
            if score > 0.99:
                logger.warning("Confidence score very high, adjusting to 0.99")
                return 0.99
            return score
        except Exception as e:
            logger.error(f"Confidence score validation error: {str(e)}")
            return 0.5  # Default to medium confidence

    @validator('impact_assessment')
    def validate_impact_assessment(cls, v):
        """Validate impact assessment value"""
        try:
            valid_impacts = {'low', 'medium', 'high'}
            normalized = str(v).lower().strip()
            if normalized in valid_impacts:
                return normalized
            else:
                logger.warning(f"Invalid impact assessment '{v}', defaulting to 'medium'")
                return "medium"
        except Exception as e:
            logger.warning(f"Impact assessment validation error: {str(e)}")
            return "medium"

    @validator('risk_factors', 'validation_suggestions', 'dependencies', pre=True)
    def validate_string_lists(cls, v):
        """Validate and clean string list fields"""
        try:
            return safe_string_list(v)
        except Exception as e:
            logger.warning(f"String list validation error: {str(e)}")
            return []

    def get_confidence_category(self) -> ConfidenceLevel:
        """Get confidence category based on score"""
        try:
            return ConfidenceLevel.from_score(self.confidence_score)
        except Exception as e:
            logger.error(f"Error getting confidence category: {str(e)}")
            return ConfidenceLevel.LOW

    def is_safe_for_auto_application(self) -> bool:
        """Determine if this change is safe for automatic application"""
        try:
            # High confidence threshold for auto-application
            if self.confidence_score < 0.8:
                return False
            
            # No removal operations in auto-mode (too risky)
            if self.change_type == ChangeType.REMOVE:
                return False
            
            # Critical financial sections need manual approval
            critical_sections = {'revenue_streams', 'cost_structure'}
            if self.canvas_section in critical_sections:
                return False
            
            # Check for high-risk indicators
            high_risk_indicators = ['major', 'significant', 'fundamental', 'critical', 'breaking', 'removing']
            if any(indicator in self.reasoning.lower() for indicator in high_risk_indicators):
                return False
            
            # Check proposed value quality
            if not self.proposed_value or len(self.proposed_value.strip()) < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking auto-application safety: {str(e)}")
            return False

    def get_estimated_business_impact(self) -> str:
        """Estimate business impact based on section and change type"""
        try:
            high_impact_sections = {'value_propositions', 'customer_segments', 'revenue_streams'}
            
            if self.canvas_section in high_impact_sections:
                if self.change_type == ChangeType.REMOVE:
                    return "high"
                elif self.confidence_score >= 0.8:
                    return "medium"
                else:
                    return "high"  # Low confidence on high-impact sections
            
            if self.change_type == ChangeType.REMOVE:
                return "medium"
            
            return "low"
        except Exception as e:
            logger.error(f"Error estimating business impact: {str(e)}")
            return "medium"

    def get_change_hash(self) -> str:
        """Get unique hash for this change to detect duplicates"""
        try:
            import hashlib
            change_string = f"{self.canvas_section}_{self.change_type.value}_{self.proposed_value}"
            return hashlib.md5(change_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating change hash: {str(e)}")
            return str(hash(str(self)))


class AgentRecommendation(BaseModel):
    """Model representing the complete recommendation from the agentic system."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    proposed_changes: List[ProposedChange] = Field(default_factory=list, description="List of proposed changes")
    next_actions: List[str] = Field(default_factory=list, description="Suggested follow-up actions")
    reasoning: str = Field(..., min_length=10, max_length=10000, description="Overall AI logic and rationale")
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence in recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="When recommendations were generated")
    processing_time_ms: Optional[int] = Field(None, ge=0, le=600000, description="Processing time in milliseconds")
    model_version: str = Field(default="gemini-2.0-flash", max_length=100, description="AI model version used")
    source_action_id: Optional[str] = Field(None, max_length=100, description="ID of triggering action")
    risk_assessment: str = Field(default="medium", description="Overall risk assessment")
    implementation_priority: str = Field(default="medium", description="Implementation priority")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Validate reasoning content"""
        try:
            if not v or not v.strip():
                return "Analysis completed with limited reasoning available"
            
            cleaned = clean_html_content(v.strip())
            
            if len(cleaned) < 10:
                return "Analysis completed with minimal reasoning"
            
            return cleaned
        except Exception as e:
            logger.warning(f"Reasoning validation error: {str(e)}")
            return "Analysis completed with validation errors"

    @validator('confidence_level', pre=True)
    def validate_confidence_level(cls, v):
        """Validate confidence level"""
        try:
            if isinstance(v, ConfidenceLevel):
                return v
            return ConfidenceLevel(str(v).lower().strip())
        except Exception as e:
            logger.warning(f"Confidence level validation error: {str(e)}")
            return ConfidenceLevel.LOW

    @validator('next_actions', pre=True)
    def validate_next_actions(cls, v):
        """Validate and clean next actions"""
        try:
            actions = safe_string_list(v)
            # Clean and validate each action
            cleaned_actions = []
            for action in actions:
                cleaned = clean_html_content(action)
                if cleaned and len(cleaned) > 5:  # Minimum meaningful length
                    cleaned_actions.append(cleaned)
            
            # If no valid actions, provide default
            if not cleaned_actions:
                cleaned_actions = ["Review the analysis and determine next steps"]
            
            return cleaned_actions
        except Exception as e:
            logger.warning(f"Next actions validation error: {str(e)}")
            return ["Review the analysis and determine next steps"]

    @validator('risk_assessment', 'implementation_priority')
    def validate_level_fields(cls, v, field):
        """Validate level fields (risk_assessment, implementation_priority)"""
        try:
            valid_levels = {'low', 'medium', 'high'}
            normalized = str(v).lower().strip()
            if normalized in valid_levels:
                return normalized
            else:
                logger.warning(f"Invalid {field.name} '{v}', defaulting to 'medium'")
                return "medium"
        except Exception as e:
            logger.warning(f"{field.name} validation error: {str(e)}")
            return "medium"

    def get_high_confidence_changes(self) -> List[ProposedChange]:
        """Return only changes with high confidence scores (>= 0.8)"""
        try:
            return [change for change in self.proposed_changes if change.confidence_score >= 0.8]
        except Exception as e:
            logger.error(f"Error filtering high confidence changes: {str(e)}")
            return []

    def get_changes_by_section(self, section_name: str) -> List[ProposedChange]:
        """Return changes for a specific canvas section"""
        try:
            if not validate_business_section_name(section_name):
                return []
            return [change for change in self.proposed_changes if change.canvas_section == section_name]
        except Exception as e:
            logger.error(f"Error filtering changes by section: {str(e)}")
            return []

    def get_safe_auto_changes(self) -> List[ProposedChange]:
        """Return changes that are safe for automatic application"""
        try:
            return [change for change in self.proposed_changes if change.is_safe_for_auto_application()]
        except Exception as e:
            logger.error(f"Error filtering safe auto changes: {str(e)}")
            return []

    def get_sections_affected(self) -> List[str]:
        """Get list of BMC sections that would be affected by these changes"""
        try:
            sections = set()
            for change in self.proposed_changes:
                try:
                    sections.add(change.canvas_section)
                except Exception:
                    continue
            return list(sections)
        except Exception as e:
            logger.error(f"Error getting affected sections: {str(e)}")
            return []

    def calculate_overall_impact_score(self) -> float:
        """Calculate an overall impact score for these recommendations"""
        try:
            if not self.proposed_changes:
                return 0.0
            
            impact_weights = {'low': 1, 'medium': 2, 'high': 3}
            total_impact = 0.0
            
            for change in self.proposed_changes:
                try:
                    impact = change.get_estimated_business_impact()
                    weight = impact_weights.get(impact, 1)
                    total_impact += weight * change.confidence_score
                except Exception:
                    continue
            
            if self.proposed_changes:
                return min(total_impact / len(self.proposed_changes), 3.0) / 3.0  # Normalize to 0-1
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating impact score: {str(e)}")
            return 0.0

    def get_implementation_timeline(self) -> str:
        """Suggest implementation timeline based on changes complexity"""
        try:
            change_count = len(self.proposed_changes)
            high_impact_count = len([c for c in self.proposed_changes 
                                   if c.get_estimated_business_impact() == "high"])
            
            if change_count == 0:
                return "No implementation needed"
            elif change_count <= 2 and high_impact_count == 0:
                return "1-2 weeks"
            elif change_count <= 4 and high_impact_count <= 1:
                return "2-3 weeks"
            else:
                return "1 month with phased approach"
        except Exception as e:
            logger.error(f"Error calculating implementation timeline: {str(e)}")
            return "Timeline assessment failed"

    def has_auto_applicable_changes(self) -> bool:
        """Check if there are any changes safe for auto-application"""
        try:
            return len(self.get_safe_auto_changes()) > 0
        except Exception as e:
            logger.error(f"Error checking auto-applicable changes: {str(e)}")
            return False


class ChangeHistory(BaseModel):
    """Model for tracking the history of changes made to business models."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the change was made")
    trigger_action_id: str = Field(..., max_length=200, description="ID of triggering action")
    changes_applied: List[ProposedChange] = Field(..., description="List of changes that were applied")
    previous_state_snapshot: Dict[str, Any] = Field(..., description="BMC state before changes")
    new_state_snapshot: Dict[str, Any] = Field(..., description="BMC state after changes")
    auto_applied: bool = Field(False, description="Whether changes were applied automatically")
    applied_by: Optional[str] = Field(None, max_length=200, description="User or system that applied changes")
    rollback_data: Optional[Dict[str, Any]] = Field(None, description="Data needed to rollback")
    notes: Optional[str] = Field(None, max_length=5000, description="Additional notes")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @validator('trigger_action_id')
    def validate_trigger_action_id(cls, v):
        """Validate trigger action ID"""
        try:
            cleaned = clean_html_content(str(v).strip())
            return cleaned if cleaned else "unknown"
        except Exception:
            return "unknown"

    @validator('notes', pre=True)
    def validate_notes(cls, v):
        """Validate and clean notes"""
        try:
            if v is None:
                return None
            return clean_html_content(str(v).strip())[:5000]  # Truncate to max length
        except Exception:
            return None

    @validator('applied_by', pre=True)
    def validate_applied_by(cls, v):
        """Validate and clean applied_by field"""
        try:
            if v is None:
                return None
            return clean_html_content(str(v).strip())[:200]  # Truncate to max length
        except Exception:
            return None

    def get_changes_summary(self) -> Dict[str, int]:
        """Get summary of changes by type and section"""
        try:
            summary = {
                "total_changes": len(self.changes_applied),
                "sections_affected": 0,
                "add_operations": 0,
                "modify_operations": 0,
                "remove_operations": 0,
            }
            
            sections_set = set()
            for change in self.changes_applied:
                try:
                    sections_set.add(change.canvas_section)
                    
                    if change.change_type == ChangeType.ADD:
                        summary["add_operations"] += 1
                    elif change.change_type == ChangeType.MODIFY:
                        summary["modify_operations"] += 1
                    elif change.change_type == ChangeType.REMOVE:
                        summary["remove_operations"] += 1
                except Exception:
                    continue
            
            summary["sections_affected"] = len(sections_set)
            return summary
        except Exception as e:
            logger.error(f"Error generating changes summary: {str(e)}")
            return {"total_changes": 0, "sections_affected": 0, "add_operations": 0, 
                   "modify_operations": 0, "remove_operations": 0}

    def calculate_change_impact(self) -> float:
        """Calculate the overall impact of these changes"""
        try:
            if not self.changes_applied:
                return 0.0
            
            total_impact = sum(change.confidence_score for change in self.changes_applied)
            return total_impact / len(self.changes_applied)
        except Exception as e:
            logger.error(f"Error calculating change impact: {str(e)}")
            return 0.0

    def can_rollback(self) -> bool:
        """Check if these changes can be rolled back"""
        try:
            return bool(self.previous_state_snapshot)
        except Exception as e:
            logger.error(f"Error checking rollback capability: {str(e)}")
            return False

    def get_rollback_summary(self) -> str:
        """Get summary for rollback confirmation"""
        try:
            change_count = len(self.changes_applied)
            timestamp = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "Unknown time"
            mode = "Auto" if self.auto_applied else "Manual"
            
            return f"Rollback {change_count} {mode.lower()} change(s) from {timestamp}?"
        except Exception as e:
            logger.error(f"Error generating rollback summary: {str(e)}")
            return "Rollback changes?"


# ==================== ANALYTICS AND REPORTING ====================

class AnalyticsSummary(BaseModel):
    """Model for tracking system usage and performance analytics."""

    total_actions_processed: int = Field(default=0, ge=0, description="Total actions processed")
    total_changes_applied: int = Field(default=0, ge=0, description="Total changes applied")
    auto_mode_usage_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Auto-mode usage rate")
    average_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average confidence score")
    most_updated_sections: List[str] = Field(default_factory=list, description="Most frequently updated sections")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate percentage")
    generated_at: datetime = Field(default_factory=datetime.now, description="When analytics were generated")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        validate_assignment = True

    @validator('most_updated_sections', pre=True)
    def validate_most_updated_sections(cls, v):
        """Validate most updated sections"""
        try:
            sections = safe_string_list(v)
            valid_sections = [s for s in sections if validate_business_section_name(s)]
            return valid_sections
        except Exception as e:
            logger.warning(f"Most updated sections validation error: {str(e)}")
            return []


# ==================== VALIDATION UTILITIES ====================

def validate_bmc_change_compatibility(change: ProposedChange, current_bmc: BusinessModelCanvas) -> List[str]:
    """Validate that a proposed change is compatible with current BMC state"""
    try:
        issues = []
        
        if change is None or current_bmc is None:
            issues.append("Cannot validate None change or BMC")
            return issues
        
        # Check if section exists
        if not hasattr(current_bmc, change.canvas_section):
            issues.append(f"Invalid section: {change.canvas_section}")
            return issues
        
        current_values = getattr(current_bmc, change.canvas_section, [])
        
        # Check MODIFY and REMOVE operations have valid current_value
        if change.change_type in [ChangeType.MODIFY, ChangeType.REMOVE]:
            if not change.current_value:
                issues.append("MODIFY/REMOVE operations require current_value")
            elif change.current_value not in current_values:
                issues.append(f"Current value '{change.current_value}' not found in {change.canvas_section}")
        
        # Check for duplicate ADD operations
        if change.change_type == ChangeType.ADD:
            if change.proposed_value in current_values:
                issues.append(f"Value '{change.proposed_value}' already exists in {change.canvas_section}")
        
        return issues
        
    except Exception as e:
        logger.error(f"Error validating change compatibility: {str(e)}")
        return [f"Validation error: {str(e)}"]


# Export all models for easy importing
__all__ = [
    # Enums
    'ActionOutcome', 'ChangeType', 'ConfidenceLevel', 'AgentStatus',
    
    # Core Business Models
    'CompletedAction', 'BusinessModelCanvas', 'ProposedChange',
    'AgentRecommendation', 'ChangeHistory',
    
    # Analytics Models
    'AnalyticsSummary',
    
    # Utility Functions
    'validate_bmc_change_compatibility', 'validate_business_section_name',
    'safe_string_list', 'clean_html_content'
]