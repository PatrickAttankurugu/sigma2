"""
User and Learning Progress Models for Azuma AI
"""

from datetime import datetime
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum


class LearningLevel(str, Enum):
    """Learning progression levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(str, Enum):
    """Preferred learning styles"""
    VISUAL = "visual"  # Diagrams, charts, visualizations
    HANDS_ON = "hands_on"  # Code examples, projects
    THEORETICAL = "theoretical"  # Concepts, math, theory
    CONVERSATIONAL = "conversational"  # Q&A, discussions
    MIXED = "mixed"  # Combination


class TopicCategory(str, Enum):
    """AI/ML topic categories"""
    FUNDAMENTALS = "fundamentals"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MLOps = "mlops"
    ADVANCED_TOPICS = "advanced_topics"


class UserProfile(BaseModel):
    """User profile and preferences"""
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Display name")
    email: Optional[str] = None
    level: LearningLevel = Field(default=LearningLevel.BEGINNER)
    learning_style: LearningStyle = Field(default=LearningStyle.MIXED)
    goals: List[str] = Field(default_factory=list, description="Learning goals")
    interests: List[str] = Field(default_factory=list, description="Specific AI/ML interests")
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    timezone: str = Field(default="UTC")

    # Gamification
    total_points: int = Field(default=0)
    current_streak: int = Field(default=0)
    longest_streak: int = Field(default=0)
    badges: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)

    # Progress tracking
    completed_lessons: Set[str] = Field(default_factory=set)
    completed_projects: Set[str] = Field(default_factory=set)
    mastered_topics: Set[str] = Field(default_factory=set)
    current_learning_path: Optional[str] = None


class TopicMastery(BaseModel):
    """Mastery level for a specific topic"""
    topic_id: str
    topic_name: str
    category: TopicCategory
    mastery_score: float = Field(default=0.0, ge=0.0, le=1.0)
    lessons_completed: int = Field(default=0)
    quizzes_passed: int = Field(default=0)
    projects_completed: int = Field(default=0)
    last_practiced: Optional[datetime] = None
    needs_review: bool = Field(default=False)

    # Prerequisite tracking
    prerequisites_met: bool = Field(default=True)
    unlocked: bool = Field(default=True)


class LearningSession(BaseModel):
    """Individual learning session data"""
    session_id: str
    user_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    duration_minutes: int = Field(default=0)

    # Session content
    topics_covered: List[str] = Field(default_factory=list)
    lessons_completed: List[str] = Field(default_factory=list)
    questions_asked: int = Field(default=0)
    code_exercises_completed: int = Field(default=0)

    # Performance
    quiz_scores: List[float] = Field(default_factory=list)
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    points_earned: int = Field(default=0)

    # AI interaction
    primary_tutor: str = Field(default="Prof. Data")
    tutor_switches: int = Field(default=0)
    feedback_given: Optional[str] = None


class QuizAttempt(BaseModel):
    """Quiz attempt and results"""
    quiz_id: str
    user_id: str
    topic_id: str
    attempted_at: datetime = Field(default_factory=datetime.now)
    score: float = Field(..., ge=0.0, le=1.0)
    total_questions: int
    correct_answers: int
    time_taken_seconds: int
    passed: bool = Field(default=False)
    detailed_results: Dict[str, bool] = Field(default_factory=dict)


class CodeExercise(BaseModel):
    """Code exercise submission"""
    exercise_id: str
    user_id: str
    topic_id: str
    submitted_at: datetime = Field(default_factory=datetime.now)
    code: str
    passed_tests: int = Field(default=0)
    total_tests: int
    execution_time_ms: float
    memory_usage_mb: float
    completed: bool = Field(default=False)
    feedback: Optional[str] = None
    hints_used: int = Field(default=0)


class DailyProgress(BaseModel):
    """Daily learning progress tracking"""
    user_id: str
    date: datetime
    lessons_completed: int = Field(default=0)
    time_spent_minutes: int = Field(default=0)
    points_earned: int = Field(default=0)
    topics_practiced: Set[str] = Field(default_factory=set)
    streak_maintained: bool = Field(default=True)
    daily_goal_met: bool = Field(default=False)
