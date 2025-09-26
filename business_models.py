"""
Enhanced Pydantic models for the Agentic AI Actions Co-pilot system.
Streamlined for Seedstars assignment focusing on core business model canvas management,
action outcomes, AI agent recommendations, and version control.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import re

from pydantic import BaseModel, Field, validator


# ==================== CORE ENUMS ====================

class ActionOutcome(str, Enum):
    """Enumeration of possible action outcomes."""
    SUCCESSFUL = "successful"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class ChangeType(str, Enum):
    """Types of changes that can be made to business model sections."""
    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"


class ConfidenceLevel(str, Enum):
    """Overall confidence levels for agent recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentStatus(str, Enum):
    """Status of individual agents in the processing pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== CORE BUSINESS MODELS ====================

class CompletedAction(BaseModel):
    """Model representing a completed business action with its outcomes."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the action")
    title: str = Field(..., min_length=3, max_length=200, description="Descriptive name of the action")
    description: str = Field(..., min_length=10, description="Detailed explanation of what was done")
    outcome: ActionOutcome = Field(..., description="The result outcome of the action")
    results_data: str = Field(..., min_length=10, description="Detailed results and metrics from the action")
    completion_date: datetime = Field(default_factory=datetime.now, description="When the action was completed")
    success_metrics: Optional[Dict[str, Any]] = Field(None, description="Quantitative metrics related to success")
    action_category: Optional[str] = Field(None, description="Category of action (e.g., 'Market Research', 'Product Test')")
    stakeholders_involved: List[str] = Field(default_factory=list, description="List of stakeholders involved in the action")
    budget_spent: Optional[float] = Field(None, ge=0, description="Budget spent on this action in local currency")
    duration_days: Optional[int] = Field(None, ge=1, description="Duration of the action in days")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "act_123456789",
                "title": "Customer Survey - Lagos Market",
                "description": "Conducted survey with 200 potential customers in Lagos to understand payment preferences and pricing sensitivity",
                "outcome": "successful",
                "results_data": "Survey achieved 85% response rate with key findings: 72% prefer mobile payments, 58% willing to pay 2% transaction fee",
                "success_metrics": {
                    "response_rate": 0.85,
                    "sample_size": 200,
                    "confidence_level": 0.95
                },
                "action_category": "Market Research",
                "stakeholders_involved": ["Research Team", "Local Partners"],
                "budget_spent": 15000.0,
                "duration_days": 14
            }
        }

    @validator('title')
    def validate_title(cls, v):
        """Validate title contains meaningful content."""
        if not v or v.isspace():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()

    @validator('results_data')
    def validate_results_data(cls, v):
        """Validate results data contains substantial information."""
        if len(v.strip()) < 20:
            raise ValueError('Results data must contain at least 20 characters of meaningful content')
        return v.strip()

    def get_action_summary(self) -> str:
        """Get a brief summary of the action."""
        return f"{self.title} ({self.outcome.value}) - completed {self.completion_date.strftime('%Y-%m-%d')}"

    def calculate_roi(self) -> Optional[float]:
        """Calculate ROI if sufficient data is available."""
        if not self.budget_spent or not self.success_metrics:
            return None
        
        revenue_generated = self.success_metrics.get('revenue_generated', 0)
        if revenue_generated > 0:
            return (revenue_generated - self.budget_spent) / self.budget_spent
        
        return None


class BusinessModelCanvas(BaseModel):
    """Model representing the 9 sections of a Business Model Canvas."""

    customer_segments: List[str] = Field(
        default_factory=list,
        description="Target customer groups the business aims to reach"
    )
    value_propositions: List[str] = Field(
        default_factory=list,
        description="Bundle of products/services that create value for customers"
    )
    channels: List[str] = Field(
        default_factory=list,
        description="How the company communicates with and reaches customers"
    )
    customer_relationships: List[str] = Field(
        default_factory=list,
        description="Types of relationships established with customer segments"
    )
    revenue_streams: List[str] = Field(
        default_factory=list,
        description="Cash the company generates from each customer segment"
    )
    key_resources: List[str] = Field(
        default_factory=list,
        description="Most important assets required to make the business model work"
    )
    key_activities: List[str] = Field(
        default_factory=list,
        description="Most important things a company must do to make its business model work"
    )
    key_partnerships: List[str] = Field(
        default_factory=list,
        description="Network of suppliers and partners that make the business model work"
    )
    cost_structure: List[str] = Field(
        default_factory=list,
        description="All costs incurred to operate a business model"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last modification"
    )
    version: str = Field(default="1.0.0", description="Version number for change tracking")
    created_by: Optional[str] = Field(None, description="User or system that created this BMC")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization and search")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "customer_segments": [
                    "Small-scale traders in urban markets",
                    "Rural farmers with seasonal income",
                    "Young professionals in emerging cities"
                ],
                "value_propositions": [
                    "Mobile money accessible on basic phones",
                    "Transaction fees 50% lower than banks",
                    "24/7 availability without internet requirement"
                ],
                "version": "1.2.1",
                "tags": ["fintech", "emerging-markets", "mobile-payments"]
            }
        }

    def get_section_by_name(self, section_name: str) -> List[str]:
        """Get a canvas section by its name."""
        return getattr(self, section_name, [])

    def update_section(self, section_name: str, new_values: List[str]) -> None:
        """Update a canvas section with new values."""
        if hasattr(self, section_name):
            setattr(self, section_name, new_values)
            self.last_updated = datetime.now()
            self._increment_version()

    def _increment_version(self) -> None:
        """Increment the patch version number."""
        try:
            major, minor, patch = map(int, self.version.split('.'))
            self.version = f"{major}.{minor}.{patch + 1}"
        except (ValueError, AttributeError):
            self.version = "1.0.1"

    def get_total_elements(self) -> int:
        """Get total number of elements across all sections."""
        sections = [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]
        return sum(len(getattr(self, section, [])) for section in sections)

    def get_empty_sections(self) -> List[str]:
        """Get list of empty sections."""
        sections = [
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        ]
        return [section for section in sections if not getattr(self, section, [])]

    def get_completeness_score(self) -> float:
        """Calculate completeness score (0.0 to 1.0)."""
        empty_sections = len(self.get_empty_sections())
        return (9 - empty_sections) / 9.0

    def get_section_quality_score(self, section_name: str) -> float:
        """Get quality score for a specific section based on content."""
        values = getattr(self, section_name, [])
        if not values:
            return 0.0

        # Base score from having content
        quality_score = 0.3
        
        # Length quality
        avg_length = sum(len(item) for item in values) / len(values)
        if avg_length > 20:
            quality_score += 0.3
        if avg_length > 50:
            quality_score += 0.2

        # Content quality - avoid generic terms
        generic_terms = {'tbd', 'to be determined', 'placeholder', 'example', 'test'}
        non_generic = sum(1 for item in values if item.lower().strip() not in generic_terms)
        quality_score += (non_generic / len(values)) * 0.2

        return min(quality_score, 1.0)

    def validate_integrity(self) -> List[str]:
        """Validate BMC integrity and return list of issues."""
        issues = []
        
        # Check for empty sections
        empty_sections = self.get_empty_sections()
        if len(empty_sections) > 3:  # More than 3 empty sections is concerning
            issues.append(f"Multiple empty sections: {', '.join(empty_sections)}")
        
        # Check for very short descriptions
        all_items = []
        sections = ['customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                   'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure']
        
        for section in sections:
            all_items.extend(getattr(self, section, []))
        
        short_items = [item for item in all_items if len(item.strip()) < 10]
        if len(short_items) > len(all_items) * 0.3:  # More than 30% short items
            issues.append(f"Many items are too brief (< 10 characters): {len(short_items)} items")
        
        # Check for placeholder content
        placeholder_patterns = ['tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo']
        placeholder_items = [item for item in all_items 
                           if any(pattern in item.lower() for pattern in placeholder_patterns)]
        if placeholder_items:
            issues.append(f"{len(placeholder_items)} placeholder items found")
        
        return issues


class ProposedChange(BaseModel):
    """Model representing a proposed change to a business model canvas."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the change")
    canvas_section: str = Field(..., description="Which BMC section to update")
    change_type: ChangeType = Field(..., description="Type of change to make")
    current_value: Optional[str] = Field(None, description="Current value being changed (if applicable)")
    proposed_value: str = Field(..., description="New value to add or modify to")
    reasoning: str = Field(..., min_length=15, description="AI explanation for why this change is recommended")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    impact_assessment: Optional[str] = Field("medium", description="Expected impact level: low, medium, high")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors for this change")
    validation_suggestions: List[str] = Field(default_factory=list, description="Suggested ways to validate this change")
    estimated_effort: Optional[str] = Field(None, description="Estimated effort to implement")
    dependencies: List[str] = Field(default_factory=list, description="Other changes this depends on")
    created_at: datetime = Field(default_factory=datetime.now, description="When this change was proposed")

    @validator('canvas_section')
    def validate_canvas_section(cls, v):
        """Validate that the canvas section is valid."""
        valid_sections = {
            'customer_segments', 'value_propositions', 'channels', 'customer_relationships',
            'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure'
        }
        if v not in valid_sections:
            raise ValueError(f"Invalid canvas section: {v}. Must be one of {valid_sections}")
        return v

    @validator('reasoning')
    def validate_reasoning_quality(cls, v):
        """Validate reasoning contains substantial explanation."""
        if len(v.strip()) < 15:
            raise ValueError("Reasoning must contain at least 15 characters")
        
        # Check for evidence-based language indicators
        evidence_indicators = ['data shows', 'research indicates', 'analysis suggests', 'evidence', 'based on', 'results show']
        has_evidence = any(indicator in v.lower() for indicator in evidence_indicators)
        
        return v.strip()

    @validator('confidence_score')
    def validate_confidence_calibration(cls, v):
        """Ensure confidence score is properly calibrated."""
        if v < 0.1:
            raise ValueError("Confidence score too low - minimum 0.1 for any recommendation")
        if v > 0.99:
            raise ValueError("Confidence score too high - maximum 0.99 to maintain humility")
        return v

    def get_confidence_category(self) -> ConfidenceLevel:
        """Get confidence category based on score."""
        if self.confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def is_safe_for_auto_application(self) -> bool:
        """Determine if this change is safe for automatic application."""
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
        
        # Check for high-risk indicators in reasoning
        high_risk_indicators = ['major', 'significant', 'fundamental', 'critical', 'breaking', 'removing']
        if any(indicator in self.reasoning.lower() for indicator in high_risk_indicators):
            return False
        
        # Check proposed value quality
        if not self.proposed_value.strip() or len(self.proposed_value.strip()) < 10:
            return False
        
        return True

    def get_estimated_business_impact(self) -> str:
        """Estimate business impact based on section and change type."""
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

    def get_change_hash(self) -> str:
        """Get unique hash for this change to detect duplicates."""
        import hashlib
        change_string = f"{self.canvas_section}_{self.change_type.value}_{self.proposed_value}"
        return hashlib.md5(change_string.encode()).hexdigest()


class AgentRecommendation(BaseModel):
    """Model representing the complete recommendation from the agentic system."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the recommendation")
    proposed_changes: List[ProposedChange] = Field(
        default_factory=list,
        description="List of proposed changes to the business model"
    )
    next_actions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions to validate or implement changes"
    )
    reasoning: str = Field(..., description="Overall AI logic and rationale for recommendations")
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence in recommendations")
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When these recommendations were generated"
    )
    processing_time_ms: Optional[int] = Field(None, description="Time taken to generate recommendations in milliseconds")
    model_version: str = Field(default="gemini-2.0-flash", description="Version of the AI model used")
    source_action_id: Optional[str] = Field(None, description="ID of the action that triggered these recommendations")
    risk_assessment: Optional[str] = Field("medium", description="Overall risk assessment: low, medium, high")
    implementation_priority: Optional[str] = Field("medium", description="Suggested implementation priority")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_high_confidence_changes(self) -> List[ProposedChange]:
        """Return only changes with high confidence scores (>= 0.8)."""
        return [change for change in self.proposed_changes if change.confidence_score >= 0.8]

    def get_changes_by_section(self, section_name: str) -> List[ProposedChange]:
        """Return changes for a specific canvas section."""
        return [change for change in self.proposed_changes if change.canvas_section == section_name]

    def get_safe_auto_changes(self) -> List[ProposedChange]:
        """Return changes that are safe for automatic application."""
        return [change for change in self.proposed_changes if change.is_safe_for_auto_application()]

    def get_sections_affected(self) -> List[str]:
        """Get list of BMC sections that would be affected by these changes."""
        return list(set(change.canvas_section for change in self.proposed_changes))

    def calculate_overall_impact_score(self) -> float:
        """Calculate an overall impact score for these recommendations."""
        if not self.proposed_changes:
            return 0.0
        
        impact_weights = {'low': 1, 'medium': 2, 'high': 3}
        total_impact = sum(
            impact_weights.get(change.get_estimated_business_impact(), 1) * change.confidence_score 
            for change in self.proposed_changes
        )
        
        return min(total_impact / len(self.proposed_changes), 3.0) / 3.0  # Normalize to 0-1

    def get_implementation_timeline(self) -> str:
        """Suggest implementation timeline based on changes complexity."""
        change_count = len(self.proposed_changes)
        high_impact_count = len([c for c in self.proposed_changes if c.get_estimated_business_impact() == "high"])
        
        if change_count == 0:
            return "No implementation needed"
        elif change_count <= 2 and high_impact_count == 0:
            return "1-2 weeks"
        elif change_count <= 4 and high_impact_count <= 1:
            return "2-3 weeks"
        else:
            return "1 month with phased approach"

    def has_auto_applicable_changes(self) -> bool:
        """Check if there are any changes safe for auto-application."""
        return len(self.get_safe_auto_changes()) > 0


class ChangeHistory(BaseModel):
    """Model for tracking the history of changes made to business models."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this change")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the change was made")
    trigger_action_id: str = Field(..., description="ID of the action that triggered this change")
    changes_applied: List[ProposedChange] = Field(..., description="List of changes that were applied")
    previous_state_snapshot: Dict[str, Any] = Field(..., description="Snapshot of BMC state before changes")
    new_state_snapshot: Dict[str, Any] = Field(..., description="Snapshot of BMC state after changes")
    auto_applied: bool = Field(False, description="Whether changes were applied automatically")
    applied_by: Optional[str] = Field(None, description="User or system that applied these changes")
    rollback_data: Optional[Dict[str, Any]] = Field(None, description="Data needed to rollback these changes")
    notes: Optional[str] = Field(None, description="Additional notes about the change")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_changes_summary(self) -> Dict[str, int]:
        """Get summary of changes by type and section."""
        summary = {
            "total_changes": len(self.changes_applied),
            "sections_affected": len(set(change.canvas_section for change in self.changes_applied)),
            "add_operations": len([c for c in self.changes_applied if c.change_type == ChangeType.ADD]),
            "modify_operations": len([c for c in self.changes_applied if c.change_type == ChangeType.MODIFY]),
            "remove_operations": len([c for c in self.changes_applied if c.change_type == ChangeType.REMOVE]),
        }
        return summary

    def calculate_change_impact(self) -> float:
        """Calculate the overall impact of these changes."""
        if not self.changes_applied:
            return 0.0
        
        total_impact = sum(change.confidence_score for change in self.changes_applied)
        return total_impact / len(self.changes_applied)

    def can_rollback(self) -> bool:
        """Check if these changes can be rolled back."""
        return bool(self.previous_state_snapshot)

    def get_rollback_summary(self) -> str:
        """Get summary for rollback confirmation."""
        change_count = len(self.changes_applied)
        timestamp = self.timestamp.strftime("%H:%M:%S")
        mode = "Auto" if self.auto_applied else "Manual"
        
        return f"Rollback {change_count} {mode.lower()} change(s) from {timestamp}?"


class ProcessingStatus(BaseModel):
    """Model to track the status of the multi-agent processing pipeline."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this processing session")
    action_detection_status: AgentStatus = Field(default=AgentStatus.PENDING)
    outcome_analysis_status: AgentStatus = Field(default=AgentStatus.PENDING)
    canvas_update_status: AgentStatus = Field(default=AgentStatus.PENDING)
    next_step_status: AgentStatus = Field(default=AgentStatus.PENDING)

    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)
    current_agent: Optional[str] = Field(None, description="Currently active agent")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall progress percentage")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def is_complete(self) -> bool:
        """Check if all agents have completed processing."""
        return all(
            status == AgentStatus.COMPLETED
            for status in [
                self.action_detection_status,
                self.outcome_analysis_status,
                self.canvas_update_status,
                self.next_step_status
            ]
        )

    def has_failed(self) -> bool:
        """Check if any agent has failed."""
        return any(
            status == AgentStatus.FAILED
            for status in [
                self.action_detection_status,
                self.outcome_analysis_status,
                self.canvas_update_status,
                self.next_step_status
            ]
        )

    def get_failed_agents(self) -> List[str]:
        """Get list of failed agents."""
        failed_agents = []
        agent_mapping = {
            'action_detection': self.action_detection_status,
            'outcome_analysis': self.outcome_analysis_status,
            'canvas_update': self.canvas_update_status,
            'next_step': self.next_step_status
        }
        
        for agent_name, status in agent_mapping.items():
            if status == AgentStatus.FAILED:
                failed_agents.append(agent_name)
        
        return failed_agents

    def calculate_progress(self) -> float:
        """Calculate overall progress percentage."""
        status_weights = {
            AgentStatus.PENDING: 0,
            AgentStatus.RUNNING: 0.5,
            AgentStatus.COMPLETED: 1,
            AgentStatus.FAILED: 0
        }
        
        total_progress = (
            status_weights[self.action_detection_status] +
            status_weights[self.outcome_analysis_status] +
            status_weights[self.canvas_update_status] +
            status_weights[self.next_step_status]
        )
        
        return (total_progress / 4.0) * 100.0

    def update_agent_status(self, agent_name: str, status: AgentStatus) -> None:
        """Update the status of a specific agent."""
        agent_mapping = {
            'action_detection': 'action_detection_status',
            'outcome_analysis': 'outcome_analysis_status',
            'canvas_update': 'canvas_update_status',
            'next_step': 'next_step_status'
        }
        
        if agent_name in agent_mapping:
            setattr(self, agent_mapping[agent_name], status)
            self.current_agent = agent_name if status == AgentStatus.RUNNING else None
            self.progress_percentage = self.calculate_progress()
            
            if self.is_complete():
                self.completed_at = datetime.now()

    def get_processing_duration(self) -> Optional[timedelta]:
        """Get total processing duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        else:
            return datetime.now() - self.started_at


# ==================== SIMPLIFIED CHAT MODELS ====================

class ChatMessage(BaseModel):
    """Simplified model for chat messages in the conversation interface."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique message identifier")
    role: str = Field(..., description="Role of the message sender: user, assistant, system")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        now = datetime.now()
        diff = now - self.timestamp
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"


# ==================== ANALYTICS AND REPORTING ====================

class AnalyticsSummary(BaseModel):
    """Simplified analytics model for tracking system usage and performance."""

    total_actions_processed: int = Field(default=0, description="Total number of actions processed")
    total_changes_applied: int = Field(default=0, description="Total number of changes applied")
    auto_mode_usage_rate: float = Field(default=0.0, description="Percentage of changes applied via auto-mode")
    average_confidence_score: float = Field(default=0.0, description="Average confidence score of recommendations")
    most_updated_sections: List[str] = Field(default_factory=list, description="BMC sections updated most frequently")
    success_rate: float = Field(default=0.0, description="Percentage of successful processing attempts")
    generated_at: datetime = Field(default_factory=datetime.now, description="When these analytics were generated")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ==================== VALIDATION UTILITIES ====================

def validate_bmc_change_compatibility(change: ProposedChange, current_bmc: BusinessModelCanvas) -> List[str]:
    """Validate that a proposed change is compatible with current BMC state."""
    issues = []
    
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


# Export all models for easy importing
__all__ = [
    # Enums
    'ActionOutcome', 'ChangeType', 'ConfidenceLevel', 'AgentStatus',
    
    # Core Business Models
    'CompletedAction', 'BusinessModelCanvas', 'ProposedChange',
    'AgentRecommendation', 'ChangeHistory', 'ProcessingStatus',
    
    # Chat and Interface Models
    'ChatMessage',
    
    # Analytics Models
    'AnalyticsSummary',
    
    # Utility Functions
    'validate_bmc_change_compatibility'
]