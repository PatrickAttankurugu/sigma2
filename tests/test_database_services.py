"""
Tests for database services
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from azuma.database.models import Base, User, LearningSession, TopicProgress
from azuma.database.services import UserService, SessionService, ProgressService, CacheService


@pytest.fixture
def db_session():
    """Create an in-memory database session for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


class TestUserService:
    """Test UserService operations"""

    def test_create_user(self, db_session):
        """Test user creation"""
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "level": "beginner",
            "goals": ["Learn ML"]
        }

        user = UserService.create_user(db_session, user_data)

        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.level == "beginner"
        assert user.total_points == 0
        assert user.current_streak == 0

    def test_get_user_by_id(self, db_session):
        """Test getting user by ID"""
        # Create user
        user_data = {"username": "test_user"}
        user = UserService.create_user(db_session, user_data)

        # Retrieve user
        retrieved = UserService.get_user_by_id(db_session, user.user_id)

        assert retrieved is not None
        assert retrieved.user_id == user.user_id
        assert retrieved.username == "test_user"

    def test_get_user_by_username(self, db_session):
        """Test getting user by username"""
        user_data = {"username": "unique_user"}
        user = UserService.create_user(db_session, user_data)

        retrieved = UserService.get_user_by_username(db_session, "unique_user")

        assert retrieved is not None
        assert retrieved.username == "unique_user"

    def test_add_points(self, db_session):
        """Test adding points to user"""
        user_data = {"username": "points_user"}
        user = UserService.create_user(db_session, user_data)

        # Add points
        updated_user = UserService.add_points(db_session, user.user_id, 100)

        assert updated_user.total_points == 100

        # Add more points
        updated_user = UserService.add_points(db_session, user.user_id, 50)

        assert updated_user.total_points == 150

    def test_add_badge(self, db_session):
        """Test adding badge to user"""
        user_data = {"username": "badge_user"}
        user = UserService.create_user(db_session, user_data)

        # Add badge
        updated_user = UserService.add_badge(db_session, user.user_id, "first_lesson")

        assert "first_lesson" in updated_user.badges

        # Add duplicate badge (should not duplicate)
        updated_user = UserService.add_badge(db_session, user.user_id, "first_lesson")

        assert updated_user.badges.count("first_lesson") == 1

    def test_update_streak(self, db_session):
        """Test updating user streak"""
        user_data = {"username": "streak_user"}
        user = UserService.create_user(db_session, user_data)

        # First activity
        updated_user = UserService.update_streak(db_session, user.user_id)

        assert updated_user.current_streak == 1
        assert updated_user.longest_streak == 1


class TestSessionService:
    """Test SessionService operations"""

    def test_create_session(self, db_session):
        """Test creating learning session"""
        # Create user first
        user_data = {"username": "session_user"}
        user = UserService.create_user(db_session, user_data)

        # Create session
        session_data = {
            "tutor_name": "Prof. Data",
            "topic": "Linear Regression"
        }

        session = SessionService.create_session(db_session, user.user_id, session_data)

        assert session.user_id_fk == user.user_id
        assert session.tutor_name == "Prof. Data"
        assert session.topic == "Linear Regression"

    def test_end_session(self, db_session):
        """Test ending a session"""
        user_data = {"username": "session_user"}
        user = UserService.create_user(db_session, user_data)

        session_data = {"tutor_name": "Neural"}
        session = SessionService.create_session(db_session, user.user_id, session_data)

        # End session
        ended_session = SessionService.end_session(
            db_session,
            session.session_id,
            duration_minutes=30,
            engagement_score=0.85
        )

        assert ended_session.ended_at is not None
        assert ended_session.duration_minutes == 30
        assert ended_session.engagement_score == 0.85


class TestProgressService:
    """Test ProgressService operations"""

    def test_update_topic_progress(self, db_session):
        """Test updating topic progress"""
        user_data = {"username": "progress_user"}
        user = UserService.create_user(db_session, user_data)

        # Update progress
        progress = ProgressService.update_topic_progress(
            db_session,
            user.user_id,
            "ml_basics",
            "Machine Learning Basics",
            0.7
        )

        assert progress.topic_id == "ml_basics"
        assert progress.mastery_score == 0.7
        assert progress.lessons_completed == 1

        # Update again with higher score
        progress = ProgressService.update_topic_progress(
            db_session,
            user.user_id,
            "ml_basics",
            "Machine Learning Basics",
            0.9
        )

        assert progress.mastery_score == 0.9  # Should take max
        assert progress.lessons_completed == 2

    def test_get_user_progress(self, db_session):
        """Test getting all user progress"""
        user_data = {"username": "multi_progress_user"}
        user = UserService.create_user(db_session, user_data)

        # Add progress for multiple topics
        ProgressService.update_topic_progress(db_session, user.user_id, "topic1", "Topic 1", 0.5)
        ProgressService.update_topic_progress(db_session, user.user_id, "topic2", "Topic 2", 0.8)

        # Get all progress
        all_progress = ProgressService.get_user_progress(db_session, user.user_id)

        assert len(all_progress) == 2


class TestCacheService:
    """Test CacheService operations"""

    def test_generate_cache_key(self):
        """Test cache key generation"""
        key1 = CacheService.generate_cache_key("Prof. Data", "What is ML?", "beginner")
        key2 = CacheService.generate_cache_key("Prof. Data", "What is ML?", "beginner")
        key3 = CacheService.generate_cache_key("Prof. Data", "What is DL?", "beginner")

        # Same input should generate same key
        assert key1 == key2

        # Different input should generate different key
        assert key1 != key3

    def test_save_and_get_response(self, db_session):
        """Test saving and retrieving cached responses"""
        # Save response
        cache = CacheService.save_response(
            db_session,
            "Neural",
            "What is backpropagation?",
            "intermediate",
            "Backpropagation is...",
            response_data={"type": "explanation"},
            ttl_hours=24
        )

        assert cache.tutor_name == "Neural"
        assert cache.hit_count == 0

        # Get cached response
        cache_key = CacheService.generate_cache_key("Neural", "What is backpropagation?", "intermediate")
        cached = CacheService.get_cached_response(db_session, cache_key)

        assert cached is not None
        assert cached.response_content == "Backpropagation is..."
        assert cached.hit_count == 1  # Should increment

    def test_cache_miss(self, db_session):
        """Test cache miss"""
        cache_key = CacheService.generate_cache_key("Vision", "Non-existent question", "expert")
        cached = CacheService.get_cached_response(db_session, cache_key)

        assert cached is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
