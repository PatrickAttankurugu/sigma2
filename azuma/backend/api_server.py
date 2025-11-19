"""
FastAPI Backend for Azuma AI v2.0
Provides REST API and WebSocket support with database persistence
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import json
import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Import Azuma AI components
from ..agents.teaching_agents import get_tutor, TeachingContext
from ..models.user import UserProfile, LearningLevel, LearningStyle
from ..gamification.achievement_system import GamificationEngine
from ..learning.adaptive_path import AdaptiveLearningEngine
from ..knowledge.knowledge_graph import KnowledgeGraph
from ..database import initialize_database, get_db, DatabaseManager
from ..database.services import UserService, SessionService, ProgressService, ChatService, CacheService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


# Request/Response Models
class UserRegistration(BaseModel):
    username: str
    email: Optional[str] = None
    level: str = "beginner"
    learning_style: str = "mixed"
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


# Global state
class AppState:
    def __init__(self):
        self.gamification = GamificationEngine()
        self.knowledge_graph = KnowledgeGraph()
        self.active_connections: Dict[str, WebSocket] = {}
        self.adaptive_engine: Optional[AdaptiveLearningEngine] = None
        self.tutors_cache: Dict[str, Any] = {}
        self.db_manager: Optional[DatabaseManager] = None

state = AppState()


# Lifespan context manager (replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Azuma AI backend v2.0...")

    try:
        # Initialize database
        state.db_manager = initialize_database()
        logger.info("Database initialized")

        # Initialize adaptive engine
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            state.adaptive_engine = AdaptiveLearningEngine(api_key)
            logger.info("Adaptive learning engine initialized")
        else:
            logger.warning("GOOGLE_API_KEY not set - some features will be limited")

        logger.info("Azuma AI backend started successfully!")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Azuma AI backend...")
    # Close any open connections
    for user_id in list(state.active_connections.keys()):
        try:
            await state.active_connections[user_id].close()
        except:
            pass


# Create FastAPI app
app = FastAPI(
    title="Azuma AI API",
    description="Agentic AI Tutor for Machine Learning - v2.0 with Database Persistence",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware - secure configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8501,http://127.0.0.1:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Dependencies
async def get_api_key():
    """Get Google API key from environment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured. Please set GOOGLE_API_KEY in .env file"
        )
    return api_key


# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Azuma AI!",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Database persistence",
            "Response caching",
            "Real-time WebSocket chat",
            "Gamification system",
            "Adaptive learning paths"
        ]
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute("SELECT 1")

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "active_connections": len(state.active_connections)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


# User Management Endpoints
@app.post("/api/users/register")
async def register_user(user_data: UserRegistration, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if username already exists
        existing = UserService.get_user_by_username(db, user_data.username)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{user_data.username}' already exists"
            )

        # Create user
        user = UserService.create_user(db, user_data.dict())

        logger.info(f"New user registered: {user.username} ({user.user_id})")

        return {
            "user_id": user.user_id,
            "message": f"Welcome to Azuma AI, {user.username}!",
            "profile": user.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@app.get("/api/users/{user_id}")
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user profile"""
    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    return user.to_dict()


@app.get("/api/users/{user_id}/stats")
async def get_user_stats(user_id: str, db: Session = Depends(get_db)):
    """Get comprehensive user statistics"""
    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # Get level info
    level_info = state.gamification.get_user_level(user.total_points)

    # Get progress
    topic_progress = ProgressService.get_user_progress(db, user_id)

    # Calculate knowledge stats
    user_mastery = {tp.topic_id: tp.mastery_score for tp in topic_progress}
    knowledge_stats = state.knowledge_graph.get_knowledge_stats(user_mastery)

    return {
        "user_id": user_id,
        "username": user.username,
        "level": level_info,
        "gamification": {
            "points": user.total_points,
            "streak": user.current_streak,
            "longest_streak": user.longest_streak,
            "badges": len(user.badges or []),
            "achievements": len(user.achievements or [])
        },
        "learning_progress": {
            "lessons_completed": len(user.completed_lessons or []),
            "projects_completed": len(user.completed_projects or []),
            "topics_mastered": len(user.mastered_topics or []),
            "topics_in_progress": len([tp for tp in topic_progress if 0 < tp.mastery_score < 0.8])
        },
        "knowledge_graph": knowledge_stats
    }


# Learning Path Endpoints
@app.post("/api/learning-paths/generate")
async def generate_learning_path(
    user_id: str,
    goal: str = "Become an ML Engineer",
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Generate personalized learning path"""
    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    if not state.adaptive_engine:
        state.adaptive_engine = AdaptiveLearningEngine(api_key)

    try:
        user_profile = UserService.to_user_profile(user)
        learning_path = await state.adaptive_engine.generate_personalized_path(user_profile, goal)

        return {
            "learning_path": learning_path.dict(),
            "message": f"Created personalized path: {learning_path.title}"
        }

    except Exception as e:
        logger.error(f"Path generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate learning path: {str(e)}"
        )


@app.get("/api/learning-paths/{user_id}/next-lesson")
async def get_next_lesson(user_id: str, db: Session = Depends(get_db)):
    """Get recommended next lesson"""
    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # Get next topics from knowledge graph
    mastered = set(user.mastered_topics or [])
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


# Progress Tracking Endpoints
@app.post("/api/progress/lesson")
async def record_lesson_progress(progress: LessonProgress, db: Session = Depends(get_db)):
    """Record lesson completion and update progress"""
    user = UserService.get_user_by_id(db, progress.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {progress.user_id} not found"
        )

    try:
        # Update user progress
        if progress.completed:
            completed_lessons = list(user.completed_lessons or [])
            if progress.lesson_id not in completed_lessons:
                completed_lessons.append(progress.lesson_id)
                UserService.update_user(db, progress.user_id, {
                    "completed_lessons": completed_lessons
                })

        # Calculate and award points
        points = state.gamification.calculate_points_for_activity("lesson_completed")
        if progress.quiz_score and progress.quiz_score == 1.0:
            points += state.gamification.calculate_points_for_activity("quiz_perfect")
        elif progress.quiz_score and progress.quiz_score >= 0.7:
            points += state.gamification.calculate_points_for_activity("quiz_passed")

        user = UserService.add_points(db, progress.user_id, points)

        # Update topic mastery
        if progress.quiz_score:
            ProgressService.update_topic_progress(
                db,
                progress.user_id,
                progress.topic_id,
                progress.topic_id,  # Use topic_id as name for now
                progress.quiz_score
            )

        # Update streak
        UserService.update_streak(db, progress.user_id)

        # Check for new badges
        user_stats = {
            "lessons_completed": len(user.completed_lessons or []),
            "current_streak": user.current_streak,
            "code_exercises_completed": 0,
            "projects_completed": len(user.completed_projects or []),
            "has_perfect_score": progress.quiz_score == 1.0 if progress.quiz_score else False
        }

        new_badges = state.gamification.get_newly_earned_badges(
            user_stats,
            set(user.badges or [])
        )

        # Award badge points
        for badge in new_badges:
            UserService.add_badge(db, progress.user_id, badge.badge_id)
            UserService.add_points(db, progress.user_id, badge.points)

        # Refresh user
        user = UserService.get_user_by_id(db, progress.user_id)

        return {
            "message": "Progress recorded successfully",
            "points_earned": points,
            "total_points": user.total_points,
            "new_badges": [{"name": b.name, "icon": b.icon, "points": b.points} for b in new_badges],
            "level": state.gamification.get_user_level(user.total_points),
            "streak": user.current_streak
        }

    except Exception as e:
        logger.error(f"Progress recording error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record progress: {str(e)}"
        )


# Gamification Endpoints
@app.get("/api/gamification/leaderboard")
async def get_leaderboard(limit: int = 100, db: Session = Depends(get_db)):
    """Get global leaderboard"""
    users = UserService.get_all_users(db, limit=limit)

    users_data = [
        {
            "username": user.username,
            "total_points": user.total_points,
            "current_streak": user.current_streak,
            "badges": user.badges or []
        }
        for user in users
    ]

    leaderboard = state.gamification.get_leaderboard_data(users_data)
    return {"leaderboard": leaderboard}


@app.get("/api/gamification/daily-challenge")
async def get_daily_challenge():
    """Get today's challenge"""
    challenge = state.gamification.get_daily_challenge()
    return {"challenge": challenge}


# Knowledge Graph Endpoints
@app.get("/api/knowledge-graph/{user_id}")
async def get_knowledge_graph(user_id: str, db: Session = Depends(get_db)):
    """Get knowledge graph visualization data"""
    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # Get topic progress
    topic_progress = ProgressService.get_user_progress(db, user_id)
    user_mastery = {tp.topic_id: tp.mastery_score for tp in topic_progress}

    graph_data = state.knowledge_graph.get_graph_visualization_data(user_mastery)

    return {
        "graph": graph_data,
        "stats": state.knowledge_graph.get_knowledge_stats(user_mastery)
    }


# WebSocket for Real-time Chat
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time chat with AI tutors"""
    await websocket.accept()
    state.active_connections[user_id] = websocket

    logger.info(f"WebSocket connection established for user {user_id}")

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
            user = UserService.get_user_by_id(db, user_id)
            if not user:
                await websocket.send_json({
                    "type": "error",
                    "message": "User not found. Please register first."
                })
                continue

            # Check cache first
            cache_key = CacheService.generate_cache_key(tutor_name, message, user.level)
            cached = CacheService.get_cached_response(db, cache_key)

            if cached:
                logger.info(f"Cache hit for {tutor_name} - {message[:30]}...")
                await websocket.send_json({
                    "type": "tutor_response",
                    "tutor": tutor_name,
                    "content": cached.response_content,
                    "cached": True,
                    "timestamp": datetime.now().isoformat()
                })
                continue

            # Create teaching context
            context = TeachingContext(
                user_id=user_id,
                current_topic=context_data.get("current_topic"),
                current_lesson=context_data.get("current_lesson"),
                user_level=user.level,
                learning_style=user.learning_style,
                mastered_topics=list(user.mastered_topics or []),
                session_history=[]
            )

            # Get or create tutor
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                await websocket.send_json({
                    "type": "error",
                    "message": "API key not configured"
                })
                continue

            tutor_cache_key = f"{tutor_name}_{api_key[:10]}"
            if tutor_cache_key not in state.tutors_cache:
                state.tutors_cache[tutor_cache_key] = get_tutor(tutor_name, api_key)

            tutor = state.tutors_cache[tutor_cache_key]

            # Get response from tutor
            try:
                response = await tutor.teach(message, context)

                # Save to cache
                CacheService.save_response(
                    db,
                    tutor_name,
                    message,
                    user.level,
                    response.content,
                    response_data=response.dict(),
                    ttl_hours=24
                )

                # Save chat message
                ChatService.save_message(db, user_id, {
                    "role": "user",
                    "content": message
                })

                ChatService.save_message(db, user_id, {
                    "role": "assistant",
                    "content": response.content,
                    "tutor_name": tutor_name,
                    "response_type": response.response_type,
                    "code_examples": response.code_examples,
                    "follow_up_questions": response.follow_up_questions
                })

                await websocket.send_json({
                    "type": "tutor_response",
                    "tutor": tutor_name,
                    "content": response.content,
                    "response_type": response.response_type,
                    "follow_up_questions": response.follow_up_questions,
                    "code_examples": response.code_examples,
                    "next_steps": response.next_steps,
                    "cached": False,
                    "timestamp": datetime.now().isoformat()
                })

                # Award points for interaction
                UserService.add_points(
                    db,
                    user_id,
                    state.gamification.calculate_points_for_activity("tutor_interaction")
                )

            except Exception as e:
                logger.error(f"Tutor response error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error getting response: {str(e)}"
                })

    except WebSocketDisconnect:
        if user_id in state.active_connections:
            del state.active_connections[user_id]
        logger.info(f"User {user_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
