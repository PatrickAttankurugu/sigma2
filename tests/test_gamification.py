"""
Tests for Gamification System
"""

import pytest
from datetime import datetime, timedelta
from azuma.gamification.achievement_system import GamificationEngine, Badge, BadgeType


class TestGamificationEngine:
    """Test GamificationEngine functionality"""

    @pytest.fixture
    def engine(self):
        """Create a GamificationEngine instance"""
        return GamificationEngine()

    def test_initialization(self, engine):
        """Test that engine initializes with badges and levels"""
        assert len(engine.badges) > 0
        assert len(engine.level_thresholds) == 10
        assert "first_lesson" in engine.badges
        assert "7_day_streak" in engine.badges

    def test_get_user_level_novice(self, engine):
        """Test getting level for novice user"""
        level_info = engine.get_user_level(0)

        assert level_info["level"] == 1
        assert level_info["title"] == "AI Novice"
        assert level_info["points"] == 0
        assert 0.0 <= level_info["progress_to_next"] <= 1.0

    def test_get_user_level_intermediate(self, engine):
        """Test getting level for intermediate user"""
        level_info = engine.get_user_level(2000)

        assert level_info["level"] >= 3
        assert level_info["points"] == 2000

    def test_get_user_level_max(self, engine):
        """Test getting level for max level user"""
        level_info = engine.get_user_level(50000)

        assert level_info["level"] == 10
        assert level_info["title"] == "AI Guru"
        assert level_info["progress_to_next"] == 1.0

    def test_calculate_points_for_activity(self, engine):
        """Test point calculation for different activities"""
        lesson_points = engine.calculate_points_for_activity("lesson_completed")
        assert lesson_points == 50

        quiz_points = engine.calculate_points_for_activity("quiz_perfect")
        assert quiz_points == 100

        project_points = engine.calculate_points_for_activity("project_completed")
        assert project_points == 500

    def test_calculate_points_with_performance(self, engine):
        """Test point calculation with performance multiplier"""
        base_points = engine.calculate_points_for_activity("lesson_completed", performance=1.0)
        high_performance_points = engine.calculate_points_for_activity("lesson_completed", performance=1.5)

        assert high_performance_points > base_points
        assert high_performance_points == int(50 * 1.5)

    def test_check_badge_earned_first_lesson(self, engine):
        """Test checking if first lesson badge is earned"""
        user_stats = {"lessons_completed": 1}

        earned = engine.check_badge_earned("first_lesson", user_stats)
        assert earned == True

        user_stats = {"lessons_completed": 0}
        earned = engine.check_badge_earned("first_lesson", user_stats)
        assert earned == False

    def test_check_badge_earned_streak(self, engine):
        """Test checking streak badges"""
        # 7-day streak
        user_stats = {"current_streak": 7}
        earned = engine.check_badge_earned("7_day_streak", user_stats)
        assert earned == True

        # Not enough streak
        user_stats = {"current_streak": 3}
        earned = engine.check_badge_earned("7_day_streak", user_stats)
        assert earned == False

    def test_check_badge_earned_perfect_score(self, engine):
        """Test checking perfect score badge"""
        user_stats = {"has_perfect_score": True}
        earned = engine.check_badge_earned("perfect_score", user_stats)
        assert earned == True

        user_stats = {"has_perfect_score": False}
        earned = engine.check_badge_earned("perfect_score", user_stats)
        assert earned == False

    def test_get_newly_earned_badges(self, engine):
        """Test getting newly earned badges"""
        user_stats = {
            "lessons_completed": 10,
            "current_streak": 3,
            "projects_completed": 0,
            "has_perfect_score": False,
            "code_exercises_completed": 0
        }

        previously_earned = set()

        new_badges = engine.get_newly_earned_badges(user_stats, previously_earned)

        # Should earn multiple badges
        assert len(new_badges) > 0

        # Should include first_lesson and 10_lessons
        badge_ids = [b.badge_id for b in new_badges]
        assert "first_lesson" in badge_ids
        assert "10_lessons" in badge_ids

    def test_get_newly_earned_badges_no_duplicates(self, engine):
        """Test that already earned badges are not returned"""
        user_stats = {
            "lessons_completed": 10,
            "current_streak": 3
        }

        previously_earned = {"first_lesson", "3_day_streak"}

        new_badges = engine.get_newly_earned_badges(user_stats, previously_earned)

        badge_ids = [b.badge_id for b in new_badges]

        # Should not include already earned badges
        assert "first_lesson" not in badge_ids
        assert "3_day_streak" not in badge_ids

    def test_get_leaderboard_data(self, engine):
        """Test generating leaderboard"""
        users_data = [
            {"username": "Alice", "total_points": 5000, "current_streak": 10, "badges": ["a", "b"]},
            {"username": "Bob", "total_points": 3000, "current_streak": 5, "badges": ["a"]},
            {"username": "Charlie", "total_points": 7000, "current_streak": 15, "badges": ["a", "b", "c"]}
        ]

        leaderboard = engine.get_leaderboard_data(users_data)

        # Should be sorted by points (descending)
        assert len(leaderboard) == 3
        assert leaderboard[0]["username"] == "Charlie"
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[1]["username"] == "Alice"
        assert leaderboard[2]["username"] == "Bob"

    def test_get_daily_challenge(self, engine):
        """Test getting daily challenge"""
        # Monday
        monday = datetime(2024, 1, 1)  # A Monday
        challenge = engine.get_daily_challenge(monday)

        assert challenge["title"] == "Monday Mastery"
        assert challenge["goal"] == 3
        assert challenge["type"] == "lessons"

        # Friday
        friday = datetime(2024, 1, 5)
        challenge = engine.get_daily_challenge(friday)

        assert challenge["title"] == "Friday Focus"
        assert challenge["type"] == "mastery"

    def test_badge_types(self, engine):
        """Test that all badge types are represented"""
        badge_types = set(badge.badge_type for badge in engine.badges.values())

        assert BadgeType.MILESTONE in badge_types
        assert BadgeType.STREAK in badge_types
        assert BadgeType.MASTERY in badge_types
        assert BadgeType.SPECIAL in badge_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
