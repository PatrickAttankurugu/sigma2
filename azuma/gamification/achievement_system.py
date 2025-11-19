"""
Gamification and Achievement System for Azuma AI
Makes learning engaging, fun, and rewarding!
"""

from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum


class BadgeType(str, Enum):
    """Types of badges"""
    MILESTONE = "milestone"
    STREAK = "streak"
    MASTERY = "mastery"
    SPECIAL = "special"
    COMPLETION = "completion"
    CHALLENGE = "challenge"


class Badge(BaseModel):
    """Achievement badge"""
    badge_id: str
    name: str
    description: str
    badge_type: BadgeType
    icon: str
    rarity: str = Field(..., description="common, rare, epic, legendary")
    points: int = Field(default=100)
    requirement: str = Field(..., description="What's needed to earn it")
    unlocked_at: Optional[datetime] = None


class Achievement(BaseModel):
    """Specific achievement"""
    achievement_id: str
    title: str
    description: str
    category: str
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    total_required: int
    current_count: int = Field(default=0)
    completed: bool = Field(default=False)
    completed_at: Optional[datetime] = None
    reward_points: int = Field(default=50)
    reward_badge: Optional[str] = None


class GamificationEngine:
    """Core gamification engine"""

    def __init__(self):
        self.badges = self._initialize_badges()
        self.achievements = self._initialize_achievements()
        self.level_thresholds = self._initialize_levels()

    def _initialize_badges(self) -> Dict[str, Badge]:
        """Initialize all available badges"""
        return {
            # Milestone Badges
            "first_lesson": Badge(
                badge_id="first_lesson",
                name="First Steps",
                description="Complete your first lesson",
                badge_type=BadgeType.MILESTONE,
                icon="🎯",
                rarity="common",
                points=50,
                requirement="Complete 1 lesson"
            ),
            "10_lessons": Badge(
                badge_id="10_lessons",
                name="Dedicated Learner",
                description="Complete 10 lessons",
                badge_type=BadgeType.MILESTONE,
                icon="📚",
                rarity="rare",
                points=200,
                requirement="Complete 10 lessons"
            ),
            "50_lessons": Badge(
                badge_id="50_lessons",
                name="Knowledge Seeker",
                description="Complete 50 lessons",
                badge_type=BadgeType.MILESTONE,
                icon="🎓",
                rarity="epic",
                points=500,
                requirement="Complete 50 lessons"
            ),

            # Streak Badges
            "3_day_streak": Badge(
                badge_id="3_day_streak",
                name="Getting Started",
                description="3-day learning streak",
                badge_type=BadgeType.STREAK,
                icon="🔥",
                rarity="common",
                points=75,
                requirement="Learn for 3 consecutive days"
            ),
            "7_day_streak": Badge(
                badge_id="7_day_streak",
                name="Week Warrior",
                description="7-day learning streak",
                badge_type=BadgeType.STREAK,
                icon="⚡",
                rarity="rare",
                points=150,
                requirement="Learn for 7 consecutive days"
            ),
            "30_day_streak": Badge(
                badge_id="30_day_streak",
                name="Monthly Master",
                description="30-day learning streak",
                badge_type=BadgeType.STREAK,
                icon="💎",
                rarity="epic",
                points=500,
                requirement="Learn for 30 consecutive days"
            ),
            "100_day_streak": Badge(
                badge_id="100_day_streak",
                name="Unstoppable",
                description="100-day learning streak",
                badge_type=BadgeType.STREAK,
                icon="🏆",
                rarity="legendary",
                points=1000,
                requirement="Learn for 100 consecutive days"
            ),

            # Mastery Badges
            "ml_basics_master": Badge(
                badge_id="ml_basics_master",
                name="ML Fundamentals Master",
                description="Master all ML fundamentals topics",
                badge_type=BadgeType.MASTERY,
                icon="🎯",
                rarity="epic",
                points=300,
                requirement="Master all fundamental ML topics"
            ),
            "deep_learning_master": Badge(
                badge_id="deep_learning_master",
                name="Deep Learning Expert",
                description="Master deep learning concepts",
                badge_type=BadgeType.MASTERY,
                icon="🧠",
                rarity="epic",
                points=400,
                requirement="Master all deep learning topics"
            ),
            "computer_vision_master": Badge(
                badge_id="computer_vision_master",
                name="Vision Virtuoso",
                description="Master computer vision",
                badge_type=BadgeType.MASTERY,
                icon="👁️",
                rarity="epic",
                points=400,
                requirement="Master all computer vision topics"
            ),
            "nlp_master": Badge(
                badge_id="nlp_master",
                name="Language Lord",
                description="Master natural language processing",
                badge_type=BadgeType.MASTERY,
                icon="📝",
                rarity="epic",
                points=400,
                requirement="Master all NLP topics"
            ),

            # Special Badges
            "early_bird": Badge(
                badge_id="early_bird",
                name="Early Bird",
                description="Learn before 8 AM",
                badge_type=BadgeType.SPECIAL,
                icon="🌅",
                rarity="rare",
                points=100,
                requirement="Complete lesson before 8 AM"
            ),
            "night_owl": Badge(
                badge_id="night_owl",
                name="Night Owl",
                description="Learn after 10 PM",
                badge_type=BadgeType.SPECIAL,
                icon="🦉",
                rarity="rare",
                points=100,
                requirement="Complete lesson after 10 PM"
            ),
            "perfect_score": Badge(
                badge_id="perfect_score",
                name="Perfectionist",
                description="Get 100% on a quiz",
                badge_type=BadgeType.SPECIAL,
                icon="💯",
                rarity="rare",
                points=150,
                requirement="Score 100% on any quiz"
            ),
            "code_master": Badge(
                badge_id="code_master",
                name="Code Master",
                description="Complete 20 code exercises",
                badge_type=BadgeType.COMPLETION,
                icon="💻",
                rarity="rare",
                points=250,
                requirement="Complete 20 code exercises"
            ),
            "project_pioneer": Badge(
                badge_id="project_pioneer",
                name="Project Pioneer",
                description="Complete your first project",
                badge_type=BadgeType.COMPLETION,
                icon="🚀",
                rarity="epic",
                points=500,
                requirement="Complete 1 full project"
            ),

            # Challenge Badges
            "speed_learner": Badge(
                badge_id="speed_learner",
                name="Speed Learner",
                description="Complete 5 lessons in one day",
                badge_type=BadgeType.CHALLENGE,
                icon="⚡",
                rarity="rare",
                points=200,
                requirement="Complete 5 lessons in 24 hours"
            ),
            "marathon_learner": Badge(
                badge_id="marathon_learner",
                name="Marathon Learner",
                description="Learn for 3+ hours in one session",
                badge_type=BadgeType.CHALLENGE,
                icon="🏃",
                rarity="epic",
                points=300,
                requirement="Single session >= 3 hours"
            ),
        }

    def _initialize_achievements(self) -> Dict[str, Achievement]:
        """Initialize all achievements"""
        return {
            "complete_10_lessons": Achievement(
                achievement_id="complete_10_lessons",
                title="Lesson Starter",
                description="Complete your first 10 lessons",
                category="learning",
                total_required=10,
                reward_points=100,
                reward_badge="10_lessons"
            ),
            "master_3_topics": Achievement(
                achievement_id="master_3_topics",
                title="Topic Triumphs",
                description="Master 3 different topics",
                category="mastery",
                total_required=3,
                reward_points=300,
                reward_badge=None
            ),
            "earn_1000_points": Achievement(
                achievement_id="earn_1000_points",
                title="Point Collector",
                description="Earn 1000 total points",
                category="points",
                total_required=1000,
                reward_points=200,
                reward_badge=None
            ),
            "7_day_streak": Achievement(
                achievement_id="7_day_streak",
                title="Consistent Learner",
                description="Maintain a 7-day streak",
                category="streak",
                total_required=7,
                reward_points=150,
                reward_badge="7_day_streak"
            ),
        }

    def _initialize_levels(self) -> Dict[int, Dict[str, any]]:
        """Initialize level progression system"""
        return {
            1: {"title": "AI Novice", "points_required": 0, "perks": []},
            2: {"title": "Data Apprentice", "points_required": 500, "perks": ["Unlock Neural tutor"]},
            3: {"title": "ML Practitioner", "points_required": 1500, "perks": ["Unlock Vision tutor"]},
            4: {"title": "Neural Architect", "points_required": 3000, "perks": ["Unlock Linguist tutor"]},
            5: {"title": "AI Engineer", "points_required": 5000, "perks": ["Unlock advanced projects"]},
            6: {"title": "ML Expert", "points_required": 8000, "perks": ["Create custom learning paths"]},
            7: {"title": "AI Master", "points_required": 12000, "perks": ["Access to all content"]},
            8: {"title": "Data Scientist", "points_required": 17000, "perks": ["Community mentor access"]},
            9: {"title": "AI Researcher", "points_required": 23000, "perks": ["Research paper discussions"]},
            10: {"title": "AI Guru", "points_required": 30000, "perks": ["All features unlocked"]},
        }

    def get_user_level(self, total_points: int) -> Dict[str, any]:
        """Calculate user's current level"""
        current_level = 1
        for level, data in sorted(self.level_thresholds.items()):
            if total_points >= data["points_required"]:
                current_level = level
            else:
                break

        level_data = self.level_thresholds[current_level]

        # Calculate progress to next level
        if current_level < 10:
            next_level_data = self.level_thresholds[current_level + 1]
            points_for_next = next_level_data["points_required"] - level_data["points_required"]
            points_progress = total_points - level_data["points_required"]
            progress = points_progress / points_for_next if points_for_next > 0 else 1.0
        else:
            progress = 1.0  # Max level

        return {
            "level": current_level,
            "title": level_data["title"],
            "points": total_points,
            "progress_to_next": progress,
            "perks": level_data["perks"]
        }

    def check_badge_earned(self, badge_id: str, user_stats: Dict[str, any]) -> bool:
        """Check if user has earned a specific badge"""
        badge = self.badges.get(badge_id)
        if not badge:
            return False

        # Implement badge logic
        if badge_id == "first_lesson":
            return user_stats.get("lessons_completed", 0) >= 1
        elif badge_id == "10_lessons":
            return user_stats.get("lessons_completed", 0) >= 10
        elif badge_id == "50_lessons":
            return user_stats.get("lessons_completed", 0) >= 50
        elif badge_id == "3_day_streak":
            return user_stats.get("current_streak", 0) >= 3
        elif badge_id == "7_day_streak":
            return user_stats.get("current_streak", 0) >= 7
        elif badge_id == "30_day_streak":
            return user_stats.get("current_streak", 0) >= 30
        elif badge_id == "100_day_streak":
            return user_stats.get("current_streak", 0) >= 100
        elif badge_id == "perfect_score":
            return user_stats.get("has_perfect_score", False)
        elif badge_id == "code_master":
            return user_stats.get("code_exercises_completed", 0) >= 20
        elif badge_id == "project_pioneer":
            return user_stats.get("projects_completed", 0) >= 1

        return False

    def get_newly_earned_badges(self, user_stats: Dict[str, any],
                                 previously_earned: Set[str]) -> List[Badge]:
        """Get list of newly earned badges"""
        newly_earned = []

        for badge_id, badge in self.badges.items():
            if badge_id not in previously_earned:
                if self.check_badge_earned(badge_id, user_stats):
                    badge.unlocked_at = datetime.now()
                    newly_earned.append(badge)

        return newly_earned

    def calculate_points_for_activity(self, activity_type: str, performance: float = 1.0) -> int:
        """Calculate points earned for an activity"""
        base_points = {
            "lesson_completed": 50,
            "quiz_passed": 30,
            "quiz_perfect": 100,
            "code_exercise": 40,
            "project_milestone": 200,
            "project_completed": 500,
            "daily_goal": 25,
            "streak_bonus": 10,
            "tutor_interaction": 5,
        }

        points = base_points.get(activity_type, 0)
        # Apply performance multiplier
        return int(points * performance)

    def get_leaderboard_data(self, users: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Generate leaderboard from user data"""
        sorted_users = sorted(users, key=lambda x: x.get("total_points", 0), reverse=True)

        leaderboard = []
        for rank, user in enumerate(sorted_users[:100], 1):  # Top 100
            level_info = self.get_user_level(user.get("total_points", 0))
            leaderboard.append({
                "rank": rank,
                "username": user.get("username", "Anonymous"),
                "points": user.get("total_points", 0),
                "level": level_info["level"],
                "level_title": level_info["title"],
                "streak": user.get("current_streak", 0),
                "badges": len(user.get("badges", [])),
            })

        return leaderboard

    def get_daily_challenge(self, date: datetime = None) -> Dict[str, any]:
        """Get daily challenge for engagement"""
        if date is None:
            date = datetime.now()

        # Rotate challenges based on day of week
        challenges = [
            {
                "title": "Monday Mastery",
                "description": "Complete 3 lessons today",
                "goal": 3,
                "reward_points": 100,
                "type": "lessons"
            },
            {
                "title": "Tuesday Trivia",
                "description": "Score 90%+ on 2 quizzes",
                "goal": 2,
                "reward_points": 150,
                "type": "quizzes"
            },
            {
                "title": "Wednesday Workout",
                "description": "Complete 5 code exercises",
                "goal": 5,
                "reward_points": 200,
                "type": "coding"
            },
            {
                "title": "Thursday Theory",
                "description": "Study for 2 hours",
                "goal": 120,  # minutes
                "reward_points": 150,
                "type": "time"
            },
            {
                "title": "Friday Focus",
                "description": "Master a new topic",
                "goal": 1,
                "reward_points": 250,
                "type": "mastery"
            },
            {
                "title": "Saturday Sprint",
                "description": "Complete a project milestone",
                "goal": 1,
                "reward_points": 300,
                "type": "project"
            },
            {
                "title": "Sunday Streak",
                "description": "Maintain your streak!",
                "goal": 1,
                "reward_points": 100,
                "type": "streak"
            },
        ]

        day_index = date.weekday()
        return challenges[day_index]
