"""
Database Services for Azuma AI
Business logic for database operations
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib

from .models import User, LearningSession, TopicProgress, ChatMessage, LessonCompletion, ResponseCache
from ..models.user import UserProfile, LearningLevel, LearningStyle


class UserService:
    """Service for user-related database operations"""

    @staticmethod
    def create_user(db: Session, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        user = User(
            user_id=user_data.get("user_id", f"user_{datetime.now().timestamp()}"),
            username=user_data["username"],
            email=user_data.get("email"),
            level=user_data.get("level", "beginner"),
            learning_style=user_data.get("learning_style", "mixed"),
            goals=user_data.get("goals", []),
            interests=user_data.get("interests", [])
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by user_id"""
        return db.query(User).filter(User.user_id == user_id).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def update_user(db: Session, user_id: str, updates: Dict[str, Any]) -> Optional[User]:
        """Update user information"""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None

        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        user.updated_at = datetime.now()
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update_streak(db: Session, user_id: str) -> User:
        """Update user's learning streak"""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None

        today = datetime.now().date()

        if user.last_activity_date:
            last_activity = user.last_activity_date.date()
            days_diff = (today - last_activity).days

            if days_diff == 1:
                # Continue streak
                user.current_streak += 1
            elif days_diff > 1:
                # Streak broken
                user.current_streak = 1
            # If days_diff == 0, already logged today, no change
        else:
            # First activity
            user.current_streak = 1

        user.longest_streak = max(user.longest_streak, user.current_streak)
        user.last_activity_date = datetime.now()
        user.last_login = datetime.now()

        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def add_points(db: Session, user_id: str, points: int) -> User:
        """Add points to user's total"""
        user = UserService.get_user_by_id(db, user_id)
        if user:
            user.total_points += points
            db.commit()
            db.refresh(user)
        return user

    @staticmethod
    def add_badge(db: Session, user_id: str, badge_id: str) -> User:
        """Add a badge to user's collection"""
        user = UserService.get_user_by_id(db, user_id)
        if user:
            badges = user.badges or []
            if badge_id not in badges:
                badges.append(badge_id)
                user.badges = badges
                db.commit()
                db.refresh(user)
        return user

    @staticmethod
    def get_all_users(db: Session, limit: int = 100) -> List[User]:
        """Get all users (for leaderboard, etc.)"""
        return db.query(User).order_by(User.total_points.desc()).limit(limit).all()

    @staticmethod
    def to_user_profile(user: User) -> UserProfile:
        """Convert database User to UserProfile model"""
        return UserProfile(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            level=LearningLevel(user.level),
            learning_style=LearningStyle(user.learning_style),
            goals=user.goals or [],
            interests=user.interests or [],
            total_points=user.total_points,
            current_streak=user.current_streak,
            longest_streak=user.longest_streak,
            last_activity_date=user.last_activity_date,
            badges=user.badges or [],
            achievements=user.achievements or [],
            completed_lessons=set(user.completed_lessons or []),
            completed_projects=set(user.completed_projects or []),
            mastered_topics=set(user.mastered_topics or [])
        )


class SessionService:
    """Service for learning session operations"""

    @staticmethod
    def create_session(db: Session, user_id: str, session_data: Dict[str, Any]) -> LearningSession:
        """Create a new learning session"""
        session = LearningSession(
            session_id=session_data.get("session_id", f"session_{datetime.now().timestamp()}"),
            user_id_fk=user_id,
            tutor_name=session_data.get("tutor_name"),
            topic=session_data.get("topic"),
            lesson_id=session_data.get("lesson_id")
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def end_session(db: Session, session_id: str, duration_minutes: int,
                   engagement_score: float = 0.0) -> Optional[LearningSession]:
        """End a learning session"""
        session = db.query(LearningSession).filter(
            LearningSession.session_id == session_id
        ).first()

        if session:
            session.ended_at = datetime.now()
            session.duration_minutes = duration_minutes
            session.engagement_score = engagement_score
            db.commit()
            db.refresh(session)

        return session

    @staticmethod
    def get_user_sessions(db: Session, user_id: str, limit: int = 50) -> List[LearningSession]:
        """Get user's learning sessions"""
        return db.query(LearningSession).filter(
            LearningSession.user_id_fk == user_id
        ).order_by(LearningSession.started_at.desc()).limit(limit).all()


class ProgressService:
    """Service for topic progress tracking"""

    @staticmethod
    def update_topic_progress(db: Session, user_id: str, topic_id: str,
                            topic_name: str, mastery_score: float) -> TopicProgress:
        """Update or create topic progress"""
        progress = db.query(TopicProgress).filter(
            TopicProgress.user_id_fk == user_id,
            TopicProgress.topic_id == topic_id
        ).first()

        if progress:
            # Update existing
            progress.mastery_score = max(progress.mastery_score, mastery_score)
            progress.lessons_completed += 1
            progress.last_practiced = datetime.now()
        else:
            # Create new
            progress = TopicProgress(
                user_id_fk=user_id,
                topic_id=topic_id,
                topic_name=topic_name,
                mastery_score=mastery_score,
                lessons_completed=1
            )
            db.add(progress)

        db.commit()
        db.refresh(progress)
        return progress

    @staticmethod
    def get_user_progress(db: Session, user_id: str) -> List[TopicProgress]:
        """Get all topic progress for a user"""
        return db.query(TopicProgress).filter(
            TopicProgress.user_id_fk == user_id
        ).all()


class ChatService:
    """Service for chat message history"""

    @staticmethod
    def save_message(db: Session, user_id: str, message_data: Dict[str, Any]) -> ChatMessage:
        """Save a chat message"""
        message = ChatMessage(
            user_id_fk=user_id,
            session_id=message_data.get("session_id"),
            role=message_data["role"],
            content=message_data["content"],
            tutor_name=message_data.get("tutor_name"),
            response_type=message_data.get("response_type"),
            code_examples=message_data.get("code_examples"),
            follow_up_questions=message_data.get("follow_up_questions")
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_chat_history(db: Session, user_id: str, limit: int = 100) -> List[ChatMessage]:
        """Get chat history for a user"""
        return db.query(ChatMessage).filter(
            ChatMessage.user_id_fk == user_id
        ).order_by(ChatMessage.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_recent_chat(db: Session, user_id: str, minutes: int = 30) -> List[ChatMessage]:
        """Get recent chat messages within specified minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return db.query(ChatMessage).filter(
            ChatMessage.user_id_fk == user_id,
            ChatMessage.created_at >= cutoff_time
        ).order_by(ChatMessage.created_at.asc()).all()


class CacheService:
    """Service for LLM response caching"""

    @staticmethod
    def generate_cache_key(tutor_name: str, question: str, user_level: str) -> str:
        """Generate cache key from request parameters"""
        key_string = f"{tutor_name}:{question.lower().strip()}:{user_level}"
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def get_cached_response(db: Session, cache_key: str) -> Optional[ResponseCache]:
        """Get cached response if exists and not expired"""
        cache = db.query(ResponseCache).filter(
            ResponseCache.cache_key == cache_key
        ).first()

        if cache:
            # Check if expired
            if cache.expires_at and cache.expires_at < datetime.now():
                db.delete(cache)
                db.commit()
                return None

            # Increment hit count
            cache.hit_count += 1
            db.commit()
            return cache

        return None

    @staticmethod
    def save_response(db: Session, tutor_name: str, question: str,
                     user_level: str, response_content: str,
                     response_data: Dict = None, ttl_hours: int = 24) -> ResponseCache:
        """Save response to cache"""
        cache_key = CacheService.generate_cache_key(tutor_name, question, user_level)

        # Check if exists
        existing = db.query(ResponseCache).filter(
            ResponseCache.cache_key == cache_key
        ).first()

        if existing:
            # Update
            existing.response_content = response_content
            existing.response_data = response_data
            existing.expires_at = datetime.now() + timedelta(hours=ttl_hours)
            db.commit()
            db.refresh(existing)
            return existing

        # Create new
        cache = ResponseCache(
            cache_key=cache_key,
            tutor_name=tutor_name,
            question=question,
            user_level=user_level,
            response_content=response_content,
            response_data=response_data,
            expires_at=datetime.now() + timedelta(hours=ttl_hours)
        )
        db.add(cache)
        db.commit()
        db.refresh(cache)
        return cache

    @staticmethod
    def clear_expired_cache(db: Session) -> int:
        """Clear expired cache entries"""
        count = db.query(ResponseCache).filter(
            ResponseCache.expires_at < datetime.now()
        ).delete()
        db.commit()
        return count
