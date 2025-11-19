"""
Lesson and Content Models for Azuma AI
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class DifficultyLevel(str, Enum):
    """Lesson difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentType(str, Enum):
    """Types of learning content"""
    TEXT = "text"
    CODE = "code"
    VISUALIZATION = "visualization"
    QUIZ = "quiz"
    INTERACTIVE = "interactive"
    VIDEO = "video"
    EXERCISE = "exercise"


class QuestionType(str, Enum):
    """Types of quiz questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    CODE_COMPLETION = "code_completion"
    SHORT_ANSWER = "short_answer"
    FILL_IN_BLANK = "fill_in_blank"


class Topic(BaseModel):
    """AI/ML Topic definition"""
    topic_id: str = Field(..., description="Unique topic identifier")
    title: str = Field(..., description="Topic title")
    description: str = Field(..., description="Brief description")
    category: str = Field(..., description="Topic category")
    difficulty: DifficultyLevel
    estimated_hours: float = Field(..., description="Estimated time to master")

    # Prerequisites
    prerequisites: List[str] = Field(default_factory=list, description="Required topic IDs")
    recommended_prerequisites: List[str] = Field(default_factory=list)

    # Content
    subtopics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    applications: List[str] = Field(default_factory=list)

    # Resources
    external_resources: List[Dict[str, str]] = Field(default_factory=list)


class ContentBlock(BaseModel):
    """Individual content block within a lesson"""
    block_id: str
    type: ContentType
    content: str = Field(..., description="Main content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    order: int = Field(default=0)

    # For code blocks
    language: Optional[str] = None
    executable: bool = Field(default=False)
    expected_output: Optional[str] = None

    # For visualizations
    visualization_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class QuizQuestion(BaseModel):
    """Individual quiz question"""
    question_id: str
    question_text: str
    question_type: QuestionType
    options: List[str] = Field(default_factory=list)
    correct_answer: str
    explanation: str = Field(..., description="Explanation of correct answer")
    difficulty: DifficultyLevel
    points: int = Field(default=10)
    hints: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class Lesson(BaseModel):
    """Complete lesson structure"""
    lesson_id: str = Field(..., description="Unique lesson identifier")
    topic_id: str = Field(..., description="Parent topic ID")
    title: str = Field(..., description="Lesson title")
    description: str = Field(..., description="What learner will achieve")
    difficulty: DifficultyLevel
    estimated_minutes: int = Field(..., description="Estimated completion time")

    # Learning objectives
    objectives: List[str] = Field(..., description="Learning objectives")
    key_concepts: List[str] = Field(default_factory=list)

    # Content
    content_blocks: List[ContentBlock] = Field(default_factory=list)
    quiz_questions: List[QuizQuestion] = Field(default_factory=list)

    # Prerequisites
    prerequisites: List[str] = Field(default_factory=list, description="Required lesson IDs")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")
    tags: List[str] = Field(default_factory=list)


class Exercise(BaseModel):
    """Coding exercise"""
    exercise_id: str
    lesson_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    estimated_minutes: int

    # Code template
    starter_code: str = Field(..., description="Initial code template")
    solution_code: str = Field(..., description="Reference solution")
    language: str = Field(default="python")

    # Testing
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)

    # Learning
    concepts_practiced: List[str] = Field(default_factory=list)
    points_reward: int = Field(default=50)


class Project(BaseModel):
    """Hands-on ML project"""
    project_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    estimated_hours: int

    # Requirements
    topics_required: List[str] = Field(default_factory=list)
    skills_required: List[str] = Field(default_factory=list)

    # Project structure
    objectives: List[str] = Field(default_factory=list)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)

    # Resources
    dataset_info: Optional[Dict[str, Any]] = None
    starter_code: Optional[str] = None
    reference_notebooks: List[str] = Field(default_factory=list)

    # Evaluation
    evaluation_criteria: List[Dict[str, Any]] = Field(default_factory=list)
    points_reward: int = Field(default=500)


class LearningPath(BaseModel):
    """Structured learning path through topics"""
    path_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    estimated_weeks: int

    # Structure
    topics: List[str] = Field(..., description="Ordered list of topic IDs")
    lessons: List[str] = Field(..., description="Ordered list of lesson IDs")
    projects: List[str] = Field(default_factory=list)

    # Goals
    learning_outcomes: List[str] = Field(default_factory=list)
    target_audience: str = Field(..., description="Who this path is for")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="azuma_ai")  # Can be AI-generated or manual
    popularity_score: float = Field(default=0.0)
    completion_rate: float = Field(default=0.0)
