"""
Adaptive Learning Path Generator for Azuma AI
Uses AI to create personalized learning journeys
"""

from typing import List, Dict, Optional, Set, Any
from datetime import datetime, timedelta
import json

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.gemini import GeminiModel
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

from ..models.user import UserProfile, TopicMastery, LearningLevel
from ..models.lesson import LearningPath, Topic, Lesson, DifficultyLevel


class AdaptiveLearningEngine:
    """
    AI-powered adaptive learning engine that:
    1. Assesses current knowledge
    2. Generates personalized learning paths
    3. Adapts difficulty based on performance
    4. Recommends next best topics
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

        if PYDANTIC_AI_AVAILABLE:
            self.agent = Agent(
                model=GeminiModel(model='gemini-2.0-flash-exp', api_key=api_key),
                system_prompt=self._get_system_prompt()
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash-exp",
                temperature=0.7
            )

    def _get_system_prompt(self) -> str:
        return """You are an expert AI/ML curriculum designer for Azuma AI.

Your role:
1. Create personalized learning paths based on user's background and goals
2. Order topics logically with proper prerequisites
3. Balance theory, practice, and projects
4. Adapt difficulty to user's level
5. Make learning engaging and achievable

AI/ML Topics Available:
- Python for ML (basics, numpy, pandas, matplotlib)
- Math Foundations (linear algebra, calculus, probability, statistics)
- Machine Learning Basics (supervised, unsupervised, evaluation)
- Deep Learning (neural networks, backprop, optimization)
- Computer Vision (CNNs, object detection, segmentation)
- NLP (transformers, BERT, GPT, attention mechanisms)
- Reinforcement Learning (Q-learning, policy gradients, DQN)
- MLOps (deployment, monitoring, scaling)

When creating paths:
- Start with fundamentals for beginners
- Include hands-on projects
- Build complexity gradually
- Estimate realistic timeframes
- Provide clear learning outcomes
"""

    async def generate_personalized_path(self, user_profile: UserProfile,
                                          goal: str = "Become an ML Engineer") -> LearningPath:
        """Generate a personalized learning path for the user"""

        user_context = f"""
User Profile:
- Level: {user_profile.level}
- Learning Style: {user_profile.learning_style}
- Goals: {', '.join(user_profile.goals) if user_profile.goals else goal}
- Interests: {', '.join(user_profile.interests) if user_profile.interests else 'General AI/ML'}
- Mastered Topics: {len(user_profile.mastered_topics)} topics
- Completed Lessons: {len(user_profile.completed_lessons)} lessons

Create a personalized learning path that:
1. Starts at their level
2. Aligns with their goals
3. Matches their learning style
4. Builds on what they know
5. Fills knowledge gaps

Return JSON with:
{{
    "path_id": "unique_id",
    "title": "descriptive title",
    "description": "what they'll achieve",
    "difficulty": "beginner/intermediate/advanced",
    "estimated_weeks": number,
    "topics": ["ordered list of topic IDs"],
    "learning_outcomes": ["specific skills they'll gain"],
    "target_audience": "who this is for"
}}
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=user_context)
            path_data = json.loads(result.data.content if hasattr(result.data, 'content') else result.data)
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=user_context)
            ]
            response = await self.llm.ainvoke(messages)

            content = response.content.strip()
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            else:
                json_part = content

            path_data = json.loads(json_part)

        # Convert to LearningPath object
        return LearningPath(
            path_id=path_data.get("path_id", f"path_{datetime.now().timestamp()}"),
            title=path_data.get("title", "Custom Learning Path"),
            description=path_data.get("description", "Personalized AI/ML journey"),
            difficulty=DifficultyLevel(path_data.get("difficulty", "beginner")),
            estimated_weeks=path_data.get("estimated_weeks", 12),
            topics=path_data.get("topics", []),
            lessons=[],  # Will be populated later
            learning_outcomes=path_data.get("learning_outcomes", []),
            target_audience=path_data.get("target_audience", "AI/ML enthusiasts"),
            created_at=datetime.now(),
            created_by="azuma_ai_adaptive_engine"
        )

    async def recommend_next_lesson(self, user_profile: UserProfile,
                                     topic_mastery: List[TopicMastery],
                                     current_path: Optional[LearningPath] = None) -> Dict[str, Any]:
        """Recommend the next best lesson for the user"""

        mastery_summary = "\n".join([
            f"- {tm.topic_name}: {tm.mastery_score:.0%} mastery, {tm.lessons_completed} lessons"
            for tm in topic_mastery[:10]  # Top 10
        ])

        prompt = f"""
User Learning State:
- Level: {user_profile.level}
- Total Lessons Completed: {len(user_profile.completed_lessons)}
- Current Streak: {user_profile.current_streak} days

Topic Mastery:
{mastery_summary}

Current Path: {current_path.title if current_path else "No active path"}

Recommend the next best lesson that:
1. Builds on what they know
2. Fills knowledge gaps
3. Maintains engagement
4. Appropriate difficulty

Return JSON:
{{
    "recommended_lesson_id": "lesson_id",
    "topic": "topic name",
    "reasoning": "why this lesson now",
    "estimated_minutes": number,
    "prerequisites_check": "ready/needs_review",
    "engagement_prediction": 0.0-1.0
}}
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=prompt)
            return json.loads(result.data.content if hasattr(result.data, 'content') else result.data)
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)

            content = response.content.strip()
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            else:
                json_part = content

            return json.loads(json_part)

    def adjust_difficulty(self, user_performance: Dict[str, float],
                          current_level: LearningLevel) -> LearningLevel:
        """Dynamically adjust difficulty based on performance"""

        avg_quiz_score = user_performance.get("avg_quiz_score", 0.7)
        completion_rate = user_performance.get("completion_rate", 0.8)
        time_efficiency = user_performance.get("time_efficiency", 1.0)

        # Performance score (0-1)
        performance_score = (avg_quiz_score * 0.5 +
                           completion_rate * 0.3 +
                           time_efficiency * 0.2)

        level_progression = {
            LearningLevel.BEGINNER: LearningLevel.INTERMEDIATE,
            LearningLevel.INTERMEDIATE: LearningLevel.ADVANCED,
            LearningLevel.ADVANCED: LearningLevel.EXPERT,
            LearningLevel.EXPERT: LearningLevel.EXPERT
        }

        level_regression = {
            LearningLevel.EXPERT: LearningLevel.ADVANCED,
            LearningLevel.ADVANCED: LearningLevel.INTERMEDIATE,
            LearningLevel.INTERMEDIATE: LearningLevel.BEGINNER,
            LearningLevel.BEGINNER: LearningLevel.BEGINNER
        }

        # High performance -> level up
        if performance_score > 0.85:
            return level_progression.get(current_level, current_level)
        # Low performance -> level down
        elif performance_score < 0.5:
            return level_regression.get(current_level, current_level)
        else:
            return current_level

    def identify_knowledge_gaps(self, topic_mastery: List[TopicMastery],
                                 target_path: LearningPath) -> List[str]:
        """Identify knowledge gaps for a target learning path"""

        required_topics = set(target_path.topics)
        mastered_topics = {tm.topic_id for tm in topic_mastery if tm.mastery_score > 0.8}

        gaps = list(required_topics - mastered_topics)
        return gaps

    def calculate_mastery_score(self, lesson_performance: Dict[str, Any]) -> float:
        """Calculate mastery score for a topic based on performance"""

        quiz_score = lesson_performance.get("quiz_score", 0.0)
        exercise_score = lesson_performance.get("exercise_score", 0.0)
        time_spent = lesson_performance.get("time_spent_minutes", 0)
        estimated_time = lesson_performance.get("estimated_minutes", 30)

        # Normalize time efficiency (1.0 = took expected time)
        time_efficiency = min(1.0, estimated_time / max(time_spent, 1))

        # Weighted mastery score
        mastery = (
            quiz_score * 0.5 +
            exercise_score * 0.4 +
            time_efficiency * 0.1
        )

        return min(1.0, max(0.0, mastery))

    def suggest_review_topics(self, topic_mastery: List[TopicMastery],
                              days_since_practice: int = 14) -> List[str]:
        """Suggest topics that need review based on spaced repetition"""

        review_needed = []

        for tm in topic_mastery:
            if not tm.last_practiced:
                continue

            days_since = (datetime.now() - tm.last_practiced).days

            # Spaced repetition intervals based on mastery
            if tm.mastery_score < 0.5 and days_since > 3:
                review_needed.append(tm.topic_id)
            elif tm.mastery_score < 0.7 and days_since > 7:
                review_needed.append(tm.topic_id)
            elif tm.mastery_score < 0.9 and days_since > 14:
                review_needed.append(tm.topic_id)
            elif days_since > 30:
                review_needed.append(tm.topic_id)

        return review_needed

    async def generate_daily_learning_plan(self, user_profile: UserProfile,
                                           available_minutes: int = 60) -> Dict[str, Any]:
        """Generate a daily learning plan based on available time"""

        plan_prompt = f"""
Create a daily learning plan for:
- User Level: {user_profile.level}
- Available Time: {available_minutes} minutes
- Current Streak: {user_profile.current_streak} days
- Learning Style: {user_profile.learning_style}

Create an achievable plan that includes:
1. Warmup/review (10-15 min)
2. New content (60-70% of time)
3. Practice exercises (20-30% of time)

Return JSON:
{{
    "warmup": {{"activity": "review topic", "minutes": 10}},
    "main_lesson": {{"lesson_id": "id", "topic": "name", "minutes": 35}},
    "practice": {{"exercises": 3, "minutes": 15}},
    "total_minutes": {available_minutes},
    "learning_outcomes": ["what they'll achieve today"]
}}
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=plan_prompt)
            return json.loads(result.data.content if hasattr(result.data, 'content') else result.data)
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=plan_prompt)
            ]
            response = await self.llm.ainvoke(messages)

            content = response.content.strip()
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            else:
                json_part = content

            return json.loads(json_part)

    def get_learning_insights(self, user_profile: UserProfile,
                              session_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning patterns and provide insights"""

        if not session_history:
            return {
                "total_time": 0,
                "avg_session": 0,
                "best_time_of_day": "Not enough data",
                "strongest_topics": [],
                "improvement_areas": [],
                "learning_velocity": 0.0
            }

        total_time = sum(s.get("duration_minutes", 0) for s in session_history)
        avg_session = total_time / len(session_history) if session_history else 0

        # Find best time of day
        time_performance = {}
        for session in session_history:
            hour = session.get("started_at", datetime.now()).hour
            time_slot = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
            if time_slot not in time_performance:
                time_performance[time_slot] = []
            time_performance[time_slot].append(session.get("engagement_score", 0.5))

        best_time = max(time_performance.items(),
                       key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)[0]

        # Learning velocity (lessons per week)
        recent_sessions = [s for s in session_history
                          if (datetime.now() - s.get("started_at", datetime.now())).days <= 7]
        lessons_this_week = sum(s.get("lessons_completed", 0) for s in recent_sessions)

        return {
            "total_time_minutes": total_time,
            "avg_session_minutes": int(avg_session),
            "best_time_of_day": best_time,
            "lessons_this_week": lessons_this_week,
            "learning_velocity": lessons_this_week / 7,  # lessons per day
            "streak": user_profile.current_streak,
            "consistency_score": min(1.0, user_profile.current_streak / 30)
        }
