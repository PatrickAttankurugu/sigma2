"""
Enhanced Pydantic models for the Agentic AI Actions Co-pilot system.

This module defines comprehensive data structures for business model canvas management,
action outcomes, AI agent recommendations, chat interfaces, and analytics.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import re

from pydantic import BaseModel, Field, validator, root_validator


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


class MessageRole(str, Enum):
    """Chat message roles for conversation tracking."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    AGENT = "agent"


class MessageType(str, Enum):
    """Types of chat messages for better categorization."""
    TEXT = "text"
    ACTION_COMPLETION = "action_completion"
    ANALYSIS_RESULT = "analysis_result"
    CHANGE_PROPOSAL = "change_proposal"
    CHANGE_APPLIED = "change_applied"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


class CompletedAction(BaseModel):
    """Model representing a completed business action with its outcomes."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the action")
    title: str = Field(..., min_length=5, max_length=200, description="Descriptive name of the action")
    description: str = Field(..., min_length=10, description="Detailed explanation of what was done")
    outcome: ActionOutcome = Field(..., description="The result outcome of the action")
    results_data: str = Field(..., min_length=20, description="Detailed results and metrics from the action")
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
                "title": "Market Validation Survey - Lagos Region",
                "description": "Conducted comprehensive market validation survey with 500 potential customers across Lagos to understand payment preferences and pricing sensitivity",
                "outcome": "successful",
                "results_data": "Survey achieved 87% response rate with key findings: 67% prefer mobile payments, 45% willing to pay 2% transaction fee, strong preference for local language support",
                "success_metrics": {
                    "response_rate": 0.87,
                    "sample_size": 500,
                    "confidence_level": 0.95
                },
                "action_category": "Market Research",
                "stakeholders_involved": ["Research Team", "Local Partners", "Customer Support"],
                "budget_spent": 25000.0,
                "duration_days": 21
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
        if len(v.strip()) < 50:
            raise ValueError('Results data must contain at least 50 characters of meaningful content')
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

    def validate_integrity(self) -> List[str]:
        """Validate BMC integrity and return list of issues."""
        issues = []
        
        # Check for empty sections
        empty_sections = self.get_empty_sections()
        if empty_sections:
            issues.append(f"Empty sections: {', '.join(empty_sections)}")
        
        # Check for very short descriptions
        all_items = []
        for section in ['customer_segments', 'value_propositions', 'channels', 'customer_relationships',
                       'revenue_streams', 'key_resources', 'key_activities', 'key_partnerships', 'cost_structure']:
            all_items.extend(getattr(self, section, []))
        
        short_items = [item for item in all_items if len(item.strip()) < 10]
        if short_items:
            issues.append(f"{len(short_items)} items are too short (< 10 characters)")
        
        # Check for placeholder content
        placeholder_patterns = ['tbd', 'to be determined', 'placeholder', 'example', 'test', 'todo']
        placeholder_items = [item for item in all_items 
                           if any(pattern in item.lower() for pattern in placeholder_patterns)]
        if placeholder_items:
            issues.append(f"{len(placeholder_items)} placeholder items found")
        
        return issues


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
    linked_bmc_version: Optional[str] = Field(None, description="Version of BMC this VPC is linked to")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_customer_profile_completeness(self) -> float:
        """Get completeness score for customer profile (jobs, pains, gains)."""
        total_sections = 3
        filled_sections = sum(1 for section in [self.jobs_to_be_done, self.pains, self.gains] 
                            if len(section) > 0)
        return filled_sections / total_sections

    def get_value_map_completeness(self) -> float:
        """Get completeness score for value map (products, pain relievers, gain creators)."""
        total_sections = 3
        filled_sections = sum(1 for section in [self.products_services, self.pain_relievers, self.gain_creators] 
                            if len(section) > 0)
        return filled_sections / total_sections


class ProposedChange(BaseModel):
    """Model representing a proposed change to a business model canvas."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the change")
    canvas_section: str = Field(..., description="Which BMC section to update")
    change_type: ChangeType = Field(..., description="Type of change to make")
    current_value: Optional[str] = Field(None, description="Current value being changed (if applicable)")
    proposed_value: str = Field(..., description="New value to add or modify to")
    reasoning: str = Field(..., min_length=20, description="AI explanation for why this change is recommended")
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
        if len(v.strip()) < 20:
            raise ValueError("Reasoning must contain at least 20 characters")
        
        # Check for evidence-based language
        evidence_indicators = ['data shows', 'research indicates', 'analysis suggests', 'evidence', 'based on']
        has_evidence = any(indicator in v.lower() for indicator in evidence_indicators)
        
        if not has_evidence:
            # Still valid but note the lack of evidence-based language
            pass
        
        return v.strip()

    @validator('confidence_score')
    def validate_confidence_calibration(cls, v):
        """Ensure confidence score is properly calibrated."""
        if v < 0.1:
            raise ValueError("Confidence score too low - minimum 0.1 for any recommendation")
        if v > 0.99:
            raise ValueError("Confidence score too high - maximum 0.99 for any prediction")
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
        if self.confidence_score < 0.75:
            return False
        
        # No removal operations in auto-mode
        if self.change_type == ChangeType.REMOVE:
            return False
        
        # Critical sections need higher confidence
        critical_sections = {'revenue_streams', 'cost_structure', 'key_partnerships'}
        if self.canvas_section in critical_sections and self.confidence_score < 0.85:
            return False
        
        # Check for high-risk factors
        high_risk_indicators = ['major', 'significant', 'fundamental', 'critical', 'breaking']
        if any(indicator in self.reasoning.lower() for indicator in high_risk_indicators):
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
    model_version: str = Field(default="1.0.0", description="Version of the AI model used")
    source_action_id: Optional[str] = Field(None, description="ID of the action that triggered these recommendations")
    market_context: Optional[Dict[str, Any]] = Field(None, description="Market context considered during analysis")
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
        elif change_count <= 3 and high_impact_count == 0:
            return "1-2 weeks"
        elif change_count <= 6 and high_impact_count <= 2:
            return "3-4 weeks"
        else:
            return "1-2 months with phased approach"


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
    success_metrics: Optional[Dict[str, Any]] = Field(None, description="Metrics to track success of these changes")
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
        return self.rollback_data is not None and bool(self.rollback_data)


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
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs from each agent")

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


class ChatMessage(BaseModel):
    """Model for chat messages in the conversation interface."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique message identifier")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message content")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    parent_message_id: Optional[str] = Field(None, description="ID of parent message for threading")
    edited: bool = Field(default=False, description="Whether this message has been edited")
    reactions: Dict[str, int] = Field(default_factory=dict, description="Message reactions (emoji: count)")

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

    def add_reaction(self, emoji: str) -> None:
        """Add a reaction to this message."""
        if emoji in self.reactions:
            self.reactions[emoji] += 1
        else:
            self.reactions[emoji] = 1

    def remove_reaction(self, emoji: str) -> bool:
        """Remove a reaction from this message."""
        if emoji in self.reactions and self.reactions[emoji] > 0:
            self.reactions[emoji] -= 1
            if self.reactions[emoji] == 0:
                del self.reactions[emoji]
            return True
        return False


class UserProfile(BaseModel):
    """Model for user profile information to provide context to AI agents."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique user identifier")
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    company: Optional[str] = Field(None, description="Company name")
    role: Optional[str] = Field(None, description="User's role or position")
    country: Optional[str] = Field(None, description="Country of residence")
    industry: Optional[str] = Field(None, description="Industry sector")
    business_stage: Optional[str] = Field(None, description="Stage of business (idea, startup, growth, etc.)")
    target_market: Optional[str] = Field(None, description="Primary target market")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences and settings")
    created_at: datetime = Field(default_factory=datetime.now, description="When the profile was created")
    last_active: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")

    @validator('email')
    def validate_email(cls, v):
        """Basic email validation."""
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower()

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_context_for_ai(self) -> Dict[str, Any]:
        """Get context information for AI agents."""
        return {
            "company": self.company,
            "role": self.role,
            "country": self.country,
            "industry": self.industry,
            "business_stage": self.business_stage,
            "target_market": self.target_market,
            "preferences": self.preferences
        }

    def update_last_active(self) -> None:
        """Update the last active timestamp."""
        self.last_active = datetime.now()


class AnalyticsReport(BaseModel):
    """Model for analytics and reporting data."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique report identifier")
    report_type: str = Field(..., description="Type of analytics report")
    generated_at: datetime = Field(default_factory=datetime.now, description="When the report was generated")
    period_start: datetime = Field(..., description="Start of reporting period")
    period_end: datetime = Field(..., description="End of reporting period")
    data: Dict[str, Any] = Field(..., description="Report data and metrics")
    summary: str = Field(..., description="Executive summary of the report")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    generated_by: str = Field(default="system", description="Who or what generated this report")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def get_key_metrics(self) -> Dict[str, Any]:
        """Extract key metrics from the report data."""
        key_metrics = {}
        
        if 'total_changes' in self.data:
            key_metrics['total_changes'] = self.data['total_changes']
        
        if 'auto_applied_ratio' in self.data:
            key_metrics['automation_rate'] = f"{self.data['auto_applied_ratio']:.1%}"
        
        if 'average_confidence' in self.data:
            key_metrics['avg_confidence'] = f"{self.data['average_confidence']:.1%}"
        
        return key_metrics


# Export all models for easy importing
__all__ = [
    # Enums
    'ActionOutcome', 'ChangeType', 'ConfidenceLevel', 'AgentStatus', 'MessageRole', 'MessageType',
    
    # Core Business Models
    'CompletedAction', 'BusinessModelCanvas', 'ValuePropositionCanvas', 'ProposedChange',
    'AgentRecommendation', 'ChangeHistory', 'ProcessingStatus',
    
    # Chat and User Models
    'ChatMessage', 'UserProfile',
    
    # Analytics Models
    'AnalyticsReport'
]