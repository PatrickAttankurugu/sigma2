"""
Utility functions for teaching agents
Reduces code duplication and provides common functionality
"""

import json
import logging
from typing import Dict, Any, Optional
from .base_tutor import TeachingResponse

logger = logging.getLogger(__name__)


def parse_json_response(content: str, tutor_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON response from LLM, handling common formats and errors

    Args:
        content: Raw content from LLM
        tutor_name: Name of the tutor for logging

    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    try:
        # Clean up the content
        content = content.strip()

        # Check for JSON code blocks
        if "```json" in content:
            json_part = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            # Try to extract from any code block
            json_part = content.split("```")[1].strip()
        else:
            json_part = content

        # Parse JSON
        data = json.loads(json_part)
        return data

    except json.JSONDecodeError as e:
        logger.warning(f"{tutor_name}: JSON decode error - {e}")
        return None
    except IndexError as e:
        logger.warning(f"{tutor_name}: Could not extract JSON from markdown - {e}")
        return None
    except Exception as e:
        logger.error(f"{tutor_name}: Unexpected error parsing JSON - {e}")
        return None


def create_fallback_response(content: str, tutor_name: str, response_type: str = "explanation") -> TeachingResponse:
    """
    Create a fallback TeachingResponse when JSON parsing fails

    Args:
        content: The raw content from LLM
        tutor_name: Name of the tutor
        response_type: Type of response

    Returns:
        TeachingResponse with basic structure
    """
    return TeachingResponse(
        content=content,
        response_type=response_type,
        follow_up_questions=["Does this make sense?", "Would you like me to explain differently?"],
        tutor_name=tutor_name,
        code_examples=[],
        next_steps=["Review the explanation", "Try a practice problem"],
        engagement_score=0.7
    )


def extract_code_examples(data: Dict[str, Any]) -> list:
    """
    Extract and format code examples from response data

    Args:
        data: Parsed JSON response

    Returns:
        List of code example dictionaries
    """
    examples = data.get("code_examples", [])

    # Ensure all examples have required fields
    formatted = []
    for ex in examples:
        if isinstance(ex, str):
            # Simple string code example
            formatted.append({
                "code": ex,
                "language": "python",
                "explanation": ""
            })
        elif isinstance(ex, dict):
            # Already formatted
            formatted.append({
                "code": ex.get("code", ""),
                "language": ex.get("language", "python"),
                "explanation": ex.get("explanation", "")
            })

    return formatted


def build_teaching_response(data: Dict[str, Any], tutor_name: str) -> TeachingResponse:
    """
    Build a TeachingResponse from parsed JSON data

    Args:
        data: Parsed JSON dictionary
        tutor_name: Name of the tutor

    Returns:
        Complete TeachingResponse object
    """
    return TeachingResponse(
        content=data.get("content", ""),
        response_type=data.get("response_type", "explanation"),
        follow_up_questions=data.get("follow_up_questions", []),
        code_examples=extract_code_examples(data),
        next_steps=data.get("next_steps", []),
        engagement_score=data.get("engagement_score", 0.7),
        tutor_name=tutor_name,
        visual_aids=data.get("visual_aids", []),
        difficulty_adjustment=data.get("difficulty_adjustment"),
        prerequisites=data.get("prerequisites", []),
        related_topics=data.get("related_topics", [])
    )


def validate_teaching_context(context: Any) -> bool:
    """
    Validate teaching context has required fields

    Args:
        context: TeachingContext object

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["user_id", "user_level", "learning_style"]

    for field in required_fields:
        if not hasattr(context, field):
            logger.warning(f"Teaching context missing required field: {field}")
            return False

    return True


def sanitize_user_input(user_input: str, max_length: int = 2000) -> str:
    """
    Sanitize user input to prevent injection attacks and ensure reasonable length

    Args:
        user_input: Raw user input
        max_length: Maximum allowed length

    Returns:
        Sanitized input string
    """
    # Remove control characters
    sanitized = ''.join(char for char in user_input if char.isprintable() or char in ['\n', '\t'])

    # Trim to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip()


def format_teaching_prompt(question: str, context: Any, personality: Any) -> str:
    """
    Create a consistent teaching prompt format across all tutors

    Args:
        question: User's question
        context: TeachingContext
        personality: TutorPersonality

    Returns:
        Formatted prompt string
    """
    # Sanitize input
    question = sanitize_user_input(question)

    # Build context summary
    mastered = context.mastered_topics if context.mastered_topics else []
    struggles = context.current_struggles if context.current_struggles else []

    prompt = f"""
Student Question: {question}

Student Context:
- Level: {context.user_level}
- Learning Style: {context.learning_style}
- Current Topic: {context.current_topic or 'Not specified'}
- Mastered Topics: {', '.join(mastered[:5]) if mastered else 'None yet'}
- Current Struggles: {', '.join(struggles) if struggles else 'None identified'}

Adapt your response to their level and learning style.
Be {personality.name} - {personality.teaching_style}!

Return a well-structured JSON response.
"""

    return prompt


def calculate_engagement_score(response_data: Dict[str, Any], context: Any) -> float:
    """
    Calculate engagement score based on response characteristics

    Args:
        response_data: The response data
        context: Teaching context

    Returns:
        Engagement score (0.0-1.0)
    """
    score = 0.5  # Base score

    # Has code examples (+0.2)
    if response_data.get("code_examples"):
        score += 0.2

    # Has follow-up questions (+0.15)
    if response_data.get("follow_up_questions"):
        score += 0.15

    # Has next steps (+0.1)
    if response_data.get("next_steps"):
        score += 0.1

    # Appropriate length (not too short, not too long)
    content_length = len(response_data.get("content", ""))
    if 200 < content_length < 1500:
        score += 0.05

    return min(1.0, score)


class ResponseCache:
    """Simple in-memory cache for tutor responses"""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, key: str, value: Dict[str, Any]):
        """Set cached response"""
        # Simple LRU eviction if cache full
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]

        self.cache[key] = value
        self.access_count[key] = 0

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()
