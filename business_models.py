"""
Pydantic models for the Agentic AI Actions Co-pilot system.

This module defines the data structures for business model canvas management,
action outcomes, and AI agent recommendations.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ActionOutcome(str, Enum):
    """Enumeration of possible action outcomes."""
    SUCCESSFUL = "successful"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class CompletedAction(BaseModel):
    """Model representing a completed business action with its outcomes."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the action")
    title: str = Field(..., description="Descriptive name of the action")
    description: str = Field(..., description="Detailed explanation of what was done")
    outcome: ActionOutcome = Field(..., description="The result outcome of the action")
    results_data: str = Field(..., description="Detailed results and metrics from the action")
    completion_date: datetime = Field(default_factory=datetime.now, description="When the action was completed")
    success_metrics: Optional[Dict[str, Any]] = Field(None, description="Quantitative metrics related to success")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_section_by_name(self, section_name: str) -> List[str]:
        """Get a canvas section by its name."""
        return getattr(self, section_name, [])

    def update_section(self, section_name: str, new_values: List[str]) -> None:
        """Update a canvas section with new values."""
        if hasattr(self, section_name):
            setattr(self, section_name, new_values)
            self.last_updated = datetime.now()


class ValuePropositionCanvas(BaseModel):
    """Model representing a Value Proposition Canvas."""

    # Customer side
    jobs_to_be_done: List[str] = Field(
        default_factory=list,
        description="Tasks customers are trying to perform, problems they're solving"
    )
    pains: List[str] = Field(
        default_factory=list,
        description="Bad outcomes, risks, and obstacles related to customer jobs"
    )
    gains: List[str] = Field(
        default_factory=list,
        description="Benefits customers want, expect, desire or would be surprised by"
    )

    # Company side
    products_services: List[str] = Field(
        default_factory=list,
        description="List of products and services the value proposition is built around"
    )
    pain_relievers: List[str] = Field(
        default_factory=list,
        description="How products/services alleviate customer pains"
    )
    gain_creators: List[str] = Field(
        default_factory=list,
        description="How products/services create customer gains"
    )

    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last modification"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChangeType(str, Enum):
    """Types of changes that can be made to business model sections."""
    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"


class ProposedChange(BaseModel):
    """Model representing a proposed change to a business model canvas."""

    canvas_section: str = Field(..., description="Which BMC section to update")
    change_type: ChangeType = Field(..., description="Type of change to make")
    current_value: Optional[str] = Field(None, description="Current value being changed (if applicable)")
    proposed_value: str = Field(..., description="New value to add or modify to")
    reasoning: str = Field(..., description="AI explanation for why this change is recommended")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )

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


class ConfidenceLevel(str, Enum):
    """Overall confidence levels for agent recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentRecommendation(BaseModel):
    """Model representing the complete recommendation from the agentic system."""

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


class ChangeHistory(BaseModel):
    """Model for tracking the history of changes made to business models."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this change")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the change was made")
    trigger_action_id: str = Field(..., description="ID of the action that triggered this change")
    changes_applied: List[ProposedChange] = Field(..., description="List of changes that were applied")
    previous_state_snapshot: Dict[str, Any] = Field(..., description="Snapshot of BMC state before changes")
    new_state_snapshot: Dict[str, Any] = Field(..., description="Snapshot of BMC state after changes")
    auto_applied: bool = Field(False, description="Whether changes were applied automatically")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentStatus(str, Enum):
    """Status of individual agents in the processing pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStatus(BaseModel):
    """Model to track the status of the multi-agent processing pipeline."""

    action_detection_status: AgentStatus = Field(default=AgentStatus.PENDING)
    outcome_analysis_status: AgentStatus = Field(default=AgentStatus.PENDING)
    canvas_update_status: AgentStatus = Field(default=AgentStatus.PENDING)
    next_step_status: AgentStatus = Field(default=AgentStatus.PENDING)

    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None)

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