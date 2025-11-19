"""
FastAPI Backend for Azuma AI
Provides REST API and WebSocket support for real-time learning
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from pydantic import BaseModel

# Import Azuma AI components
from ..agents.teaching_agents import get_tutor, TeachingContext
from ..models.user import UserProfile, LearningLevel, LearningStyle
from ..models.lesson import Lesson, Quiz Question, Exercise
from ..gamification.achievement_system import GamificationEngine
from ..learning.adaptive_path import AdaptiveLearningEngine
from ..knowledge.knowledge_graph import KnowledgeGraph


# Request/Response Models
class ChatMessage(BaseModel):
    user_id: str
    message: str
    tutor_name: str = "Prof. Data"
    context: Optional[Dict[str, Any]] = None


class UserRegistration(BaseModel):
    username: str
    email: Optional[str] = None
    level: LearningLevel = LearningLevel.BEGINNER
    learning_style: LearningStyle = LearningStyle.MIXED
    goals: List[str] = []
    interests: List[str] = []


class LessonProgress(BaseModel):
    user_id: str
    lesson_id: str
    topic_id: str
    completed: bool
    quiz_score: Optional[float] = None
    time_spent_minutes: int
    exercise_score: Optional[float] = None


# Create FastAPI app
app = FastAPI(
    title="Azuma AI API",
    description="Agentic AI Tutor for Machine Learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper database)
class AppState:
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.gamification = GamificationEngine()
        self.knowledge_graph = KnowledgeGraph()
        self.active_connections: Dict[str, WebSocket] = {}
        self.adaptive_engine: Optional[AdaptiveLearningEngine] = None
        self.tutors_cache: Dict[str, Any] = {}

state = AppState()


# Dependency to get API key
async def get_api_key():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    return api_key


# Initialize adaptive engine
@app.on_event("startup")
async def startup_event():
    api_key = await get_api_key()
    state.adaptive_engine = AdaptiveLearningEngine(api_key)
    print("✅ Azuma AI backend started successfully!")


# Health check
@app.get("/")
async def root():
    return {
        "message": "Welcome to Azuma AI!",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_users": len(state.users),
        "active_connections": len(state.active_connections)
    }


# User Management
@app.post("/api/users/register")
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    user_id = f"user_{datetime.now().timestamp()}"

    user_profile = UserProfile(
        user_id=user_id,
        username=user_data.username,
        email=user_data.email,
        level=user_data.level,
        learning_style=user_data.learning_style,
        goals=user_data.goals,
        interests=user_data.interests
    )

    state.users[user_id] = user_profile

    return {
        "user_id": user_id,
        "message": f"Welcome to Azuma AI, {user_data.username}!",
        "profile": user_profile.dict()
    }


@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get user profile"""
    if user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    return state.users[user_id].dict()


@app.get("/api/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user statistics and progress"""
    if user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    user = state.users[user_id]

    # Get level info
    level_info = state.gamification.get_user_level(user.total_points)

    # Get knowledge graph stats
    user_mastery = {topic: 0.8 for topic in user.mastered_topics}  # Simplified
    knowledge_stats = state.knowledge_graph.get_knowledge_stats(user_mastery)

    return {
        "user_id": user_id,
        "username": user.username,
        "level": level_info,
        "gamification": {
            "points": user.total_points,
            "streak": user.current_streak,
            "badges": len(user.badges),
            "achievements": len(user.achievements)
        },
        "learning_progress": {
            "lessons_completed": len(user.completed_lessons),
            "projects_completed": len(user.completed_projects),
            "topics_mastered": len(user.mastered_topics)
        },
        "knowledge_graph": knowledge_stats
    }


# Learning Paths
@app.post("/api/learning-paths/generate")
async def generate_learning_path(user_id: str, goal: str = "Become an ML Engineer"):
    """Generate personalized learning path"""
    if user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    user = state.users[user_id]

    if not state.adaptive_engine:
        api_key = await get_api_key()
        state.adaptive_engine = AdaptiveLearningEngine(api_key)

    learning_path = await state.adaptive_engine.generate_personalized_path(user, goal)

    return {
        "learning_path": learning_path.dict(),
        "message": f"Created personalized path: {learning_path.title}"
    }


@app.get("/api/learning-paths/{user_id}/next-lesson")
async def get_next_lesson(user_id: str):
    """Get recommended next lesson"""
    if user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    user = state.users[user_id]

    # Get next topics from knowledge graph
    mastered = user.mastered_topics
    next_topics = state.knowledge_graph.get_next_topics(mastered, limit=3)

    return {
        "recommended_topics": [
            {
                "topic_id": topic.node_id,
                "title": topic.title,
                "category": topic.category,
                "difficulty": topic.difficulty,
                "prerequisites_met": topic.is_ready_to_learn(state.knowledge_graph)
            }
            for topic in next_topics
        ]
    }


# Progress Tracking
@app.post("/api/progress/lesson")
async def record_lesson_progress(progress: LessonProgress):
    """Record lesson completion and update progress"""
    if progress.user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    user = state.users[progress.user_id]

    # Update user progress
    if progress.completed:
        user.completed_lessons.add(progress.lesson_id)

    # Calculate points
    points = state.gamification.calculate_points_for_activity("lesson_completed")
    if progress.quiz_score and progress.quiz_score == 1.0:
        points += state.gamification.calculate_points_for_activity("quiz_perfect")
    elif progress.quiz_score and progress.quiz_score >= 0.7:
        points += state.gamification.calculate_points_for_activity("quiz_passed")

    user.total_points += points

    # Update knowledge graph
    if progress.quiz_score:
        state.knowledge_graph.update_mastery(progress.topic_id, progress.quiz_score)

    # Check for new badges
    user_stats = {
        "lessons_completed": len(user.completed_lessons),
        "current_streak": user.current_streak,
        "code_exercises_completed": 0,  # TODO: track this
        "projects_completed": len(user.completed_projects),
        "has_perfect_score": progress.quiz_score == 1.0 if progress.quiz_score else False
    }

    new_badges = state.gamification.get_newly_earned_badges(user_stats, set(user.badges))

    for badge in new_badges:
        user.badges.append(badge.badge_id)
        user.total_points += badge.points

    return {
        "message": "Progress recorded successfully",
        "points_earned": points,
        "total_points": user.total_points,
        "new_badges": [badge.dict() for badge in new_badges],
        "level": state.gamification.get_user_level(user.total_points)
    }


# Gamification
@app.get("/api/gamification/leaderboard")
async def get_leaderboard():
    """Get global leaderboard"""
    users_data = [
        {
            "username": user.username,
            "total_points": user.total_points,
            "current_streak": user.current_streak,
            "badges": user.badges
        }
        for user in state.users.values()
    ]

    leaderboard = state.gamification.get_leaderboard_data(users_data)
    return {"leaderboard": leaderboard}


@app.get("/api/gamification/daily-challenge")
async def get_daily_challenge():
    """Get today's challenge"""
    challenge = state.gamification.get_daily_challenge()
    return {"challenge": challenge}


# Knowledge Graph
@app.get("/api/knowledge-graph/{user_id}")
async def get_knowledge_graph(user_id: str):
    """Get knowledge graph visualization data"""
    if user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    user = state.users[user_id]
    user_mastery = {topic: 0.8 for topic in user.mastered_topics}  # Simplified

    graph_data = state.knowledge_graph.get_graph_visualization_data(user_mastery)

    return {
        "graph": graph_data,
        "stats": state.knowledge_graph.get_knowledge_stats(user_mastery)
    }


# WebSocket for Real-time Chat
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat with AI tutors"""
    await websocket.accept()
    state.active_connections[user_id] = websocket

    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Azuma AI! Choose a tutor to start learning.",
            "tutors": ["Prof. Data", "Neural", "Vision", "Linguist"]
        })

        while True:
            # Receive message
            data = await websocket.receive_json()

            message = data.get("message", "")
            tutor_name = data.get("tutor", "Prof. Data")
            context_data = data.get("context", {})

            # Get user
            if user_id not in state.users:
                await websocket.send_json({
                    "type": "error",
                    "message": "User not found. Please register first."
                })
                continue

            user = state.users[user_id]

            # Create teaching context
            context = TeachingContext(
                user_id=user_id,
                current_topic=context_data.get("current_topic"),
                current_lesson=context_data.get("current_lesson"),
                user_level=user.level.value,
                learning_style=user.learning_style.value,
                mastered_topics=list(user.mastered_topics),
                session_history=[]
            )

            # Get or create tutor
            api_key = await get_api_key()
            tutor_cache_key = f"{tutor_name}_{api_key[:10]}"

            if tutor_cache_key not in state.tutors_cache:
                state.tutors_cache[tutor_cache_key] = get_tutor(tutor_name, api_key)

            tutor = state.tutors_cache[tutor_cache_key]

            # Get response from tutor
            try:
                response = await tutor.teach(message, context)

                await websocket.send_json({
                    "type": "tutor_response",
                    "tutor": tutor_name,
                    "content": response.content,
                    "response_type": response.response_type,
                    "follow_up_questions": response.follow_up_questions,
                    "code_examples": response.code_examples,
                    "next_steps": response.next_steps,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error getting response: {str(e)}"
                })

    except WebSocketDisconnect:
        if user_id in state.active_connections:
            del state.active_connections[user_id]
        print(f"User {user_id} disconnected")

    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Connection error: {str(e)}"
        })


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
