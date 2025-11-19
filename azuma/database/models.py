"""
Database Models for Azuma AI
Provides persistent storage for users, progress, and sessions
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()


class User(Base):
    """User account and profile"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=True)  # For future authentication

    # Learning profile
    level = Column(String(50), default="beginner")
    learning_style = Column(String(50), default="mixed")
    goals = Column(JSON, default=list)
    interests = Column(JSON, default=list)

    # Gamification
    total_points = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_activity_date = Column(DateTime, nullable=True)
    badges = Column(JSON, default=list)
    achievements = Column(JSON, default=list)

    # Progress tracking
    completed_lessons = Column(JSON, default=list)
    completed_projects = Column(JSON, default=list)
    mastered_topics = Column(JSON, default=list)

    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    sessions = relationship("LearningSession", back_populates="user", cascade="all, delete-orphan")
    progress = relationship("TopicProgress", back_populates="user", cascade="all, delete-orphan")
    chat_history = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "level": self.level,
            "learning_style": self.learning_style,
            "goals": self.goals or [],
            "interests": self.interests or [],
            "total_points": self.total_points,
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "badges": self.badges or [],
            "achievements": self.achievements or [],
            "completed_lessons": self.completed_lessons or [],
            "completed_projects": self.completed_projects or [],
            "mastered_topics": self.mastered_topics or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class LearningSession(Base):
    """Individual learning session tracking"""
    __tablename__ = "learning_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id_fk = Column(String(100), ForeignKey("users.user_id"), nullable=False)

    # Session details
    tutor_name = Column(String(100), nullable=True)
    topic = Column(String(255), nullable=True)
    lesson_id = Column(String(100), nullable=True)

    # Performance metrics
    duration_minutes = Column(Integer, default=0)
    messages_exchanged = Column(Integer, default=0)
    engagement_score = Column(Float, default=0.0)
    understanding_score = Column(Float, default=0.0)

    # Timestamps
    started_at = Column(DateTime, default=datetime.now)
    ended_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "user_id": self.user_id_fk,
            "tutor_name": self.tutor_name,
            "topic": self.topic,
            "duration_minutes": self.duration_minutes,
            "engagement_score": self.engagement_score,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None
        }


class TopicProgress(Base):
    """Topic mastery and progress tracking"""
    __tablename__ = "topic_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_fk = Column(String(100), ForeignKey("users.user_id"), nullable=False)
    topic_id = Column(String(100), nullable=False, index=True)
    topic_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)

    # Progress metrics
    mastery_score = Column(Float, default=0.0)
    lessons_completed = Column(Integer, default=0)
    quiz_average = Column(Float, default=0.0)
    practice_count = Column(Integer, default=0)

    # Timestamps
    first_practiced = Column(DateTime, default=datetime.now)
    last_practiced = Column(DateTime, default=datetime.now)

    # Relationships
    user = relationship("User", back_populates="progress")

    def to_dict(self):
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "category": self.category,
            "mastery_score": self.mastery_score,
            "lessons_completed": self.lessons_completed,
            "quiz_average": self.quiz_average,
            "last_practiced": self.last_practiced.isoformat() if self.last_practiced else None
        }


class ChatMessage(Base):
    """Chat message history"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_fk = Column(String(100), ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String(100), nullable=True)

    # Message details
    role = Column(String(20), nullable=False)  # user or assistant
    content = Column(Text, nullable=False)
    tutor_name = Column(String(100), nullable=True)

    # Response metadata (for assistant messages)
    response_type = Column(String(50), nullable=True)
    code_examples = Column(JSON, nullable=True)
    follow_up_questions = Column(JSON, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.now, index=True)

    # Relationships
    user = relationship("User", back_populates="chat_history")

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "tutor_name": self.tutor_name,
            "response_type": self.response_type,
            "code_examples": self.code_examples,
            "follow_up_questions": self.follow_up_questions,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class LessonCompletion(Base):
    """Lesson completion records"""
    __tablename__ = "lesson_completions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_fk = Column(String(100), ForeignKey("users.user_id"), nullable=False)
    lesson_id = Column(String(100), nullable=False, index=True)
    topic_id = Column(String(100), nullable=False)

    # Performance
    completed = Column(Boolean, default=False)
    quiz_score = Column(Float, nullable=True)
    exercise_score = Column(Float, nullable=True)
    time_spent_minutes = Column(Integer, default=0)

    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "lesson_id": self.lesson_id,
            "topic_id": self.topic_id,
            "completed": self.completed,
            "quiz_score": self.quiz_score,
            "time_spent_minutes": self.time_spent_minutes,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class ResponseCache(Base):
    """Cache for LLM responses to improve performance"""
    __tablename__ = "response_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)

    # Request context
    tutor_name = Column(String(100), nullable=False)
    question = Column(Text, nullable=False)
    user_level = Column(String(50), nullable=True)

    # Response data
    response_content = Column(Text, nullable=False)
    response_data = Column(JSON, nullable=True)

    # Cache management
    hit_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "tutor_name": self.tutor_name,
            "response_content": self.response_content,
            "response_data": self.response_data
        }
