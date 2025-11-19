"""
Base Teaching Agent for Azuma AI using PydanticAI
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from abc import ABC, abstractmethod


class TutorPersonality(BaseModel):
    """Tutor personality traits"""
    name: str
    specialty: str
    teaching_style: str
    personality_traits: List[str]
    catchphrase: str
    emoji: str


class TeachingContext(BaseModel):
    """Context for teaching interactions"""
    user_id: str
    current_topic: Optional[str] = None
    current_lesson: Optional[str] = None
    user_level: str = "beginner"
    learning_style: str = "mixed"
    session_history: List[Dict[str, Any]] = Field(default_factory=list)
    mastered_topics: List[str] = Field(default_factory=list)
    current_struggles: List[str] = Field(default_factory=list)


class TeachingResponse(BaseModel):
    """Structured response from a teaching agent"""
    content: str = Field(..., description="Main teaching content")
    response_type: str = Field(..., description="Type: explanation, question, hint, encouragement")
    follow_up_questions: List[str] = Field(default_factory=list)
    code_examples: List[Dict[str, str]] = Field(default_factory=list)
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    difficulty_adjustment: Optional[str] = None  # "easier", "harder", "same"
    engagement_score: float = Field(default=0.8, ge=0.0, le=1.0)
    tutor_name: str = Field(default="Azuma AI")


class BaseTutor(ABC):
    """Base class for all teaching agents"""

    def __init__(self, personality: TutorPersonality, llm_config: Optional[Dict[str, Any]] = None):
        self.personality = personality
        self.llm_config = llm_config or {}
        self.interaction_count = 0

    @abstractmethod
    async def teach(self, question: str, context: TeachingContext) -> TeachingResponse:
        """Main teaching method - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def assess(self, user_response: str, context: TeachingContext) -> Dict[str, Any]:
        """Assess student understanding"""
        pass

    @abstractmethod
    async def generate_practice(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Generate practice problems"""
        pass

    def get_introduction(self) -> str:
        """Get tutor introduction"""
        return f"""
        {self.personality.emoji} Hi! I'm **{self.personality.name}**, your {self.personality.specialty} tutor!

        {self.personality.catchphrase}

        **My teaching style**: {self.personality.teaching_style}

        **What makes me unique**: {', '.join(self.personality.personality_traits)}

        Let's learn together! Ask me anything about {self.personality.specialty}.
        """

    def _create_socratic_question(self, topic: str, level: str) -> str:
        """Create a Socratic question to guide learning"""
        # This will be enhanced with LLM
        questions = {
            "beginner": f"What do you think {topic} is used for in real life?",
            "intermediate": f"How would you explain {topic} to someone new to AI?",
            "advanced": f"What are the trade-offs when using {topic} in production?"
        }
        return questions.get(level, questions["beginner"])

    def _adjust_difficulty(self, context: TeachingContext, performance: float) -> str:
        """Determine if difficulty should be adjusted"""
        if performance < 0.4:
            return "easier"
        elif performance > 0.9 and context.user_level != "expert":
            return "harder"
        return "same"

    def _format_code_example(self, code: str, language: str = "python",
                            explanation: str = "") -> Dict[str, str]:
        """Format a code example"""
        return {
            "code": code,
            "language": language,
            "explanation": explanation
        }

    def update_interaction_count(self):
        """Track interactions for engagement metrics"""
        self.interaction_count += 1
