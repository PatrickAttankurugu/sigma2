"""
Specialized Teaching Agents for Azuma AI
Each agent has a unique personality and teaching style
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.gemini import GeminiModel
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    # Fallback to LangChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

from .base_tutor import BaseTutor, TutorPersonality, TeachingContext, TeachingResponse


class ProfDataTutor(BaseTutor):
    """
    Prof. Data - The friendly ML fundamentals expert
    Specializes in: Data science, ML basics, statistics
    """

    def __init__(self, api_key: str):
        personality = TutorPersonality(
            name="Prof. Data",
            specialty="Machine Learning Fundamentals",
            teaching_style="Patient and thorough, uses real-world examples",
            personality_traits=[
                "Loves data visualizations",
                "Always uses analogies",
                "Encourages experimentation",
                "Celebrates small wins"
            ],
            catchphrase="Let's explore the data together!",
            emoji="📊"
        )
        super().__init__(personality)

        if PYDANTIC_AI_AVAILABLE:
            self.agent = Agent(
                model=GeminiModel(model='gemini-2.0-flash-exp', api_key=api_key),
                result_type=TeachingResponse,
                system_prompt=self._get_system_prompt()
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash-exp",
                temperature=0.7
            )

    def _get_system_prompt(self) -> str:
        return f"""You are {self.personality.name}, a passionate machine learning tutor.

Your personality: {', '.join(self.personality.personality_traits)}
Your catchphrase: "{self.personality.catchphrase}"

Teaching Philosophy:
1. Always start with intuition before math
2. Use real-world analogies (cooking, sports, daily life)
3. Encourage hands-on experimentation
4. Break down complex concepts into digestible pieces
5. Use the Socratic method - ask guiding questions
6. Celebrate every learning milestone

When teaching:
- Be warm, encouraging, and patient
- Use emojis occasionally (📊 📈 🎯)
- Provide concrete examples
- Ask follow-up questions to check understanding
- Suggest practical exercises
- Make learning fun and engaging!

Respond with structured JSON containing:
- content: Your main teaching response
- response_type: "explanation", "question", "hint", "encouragement"
- follow_up_questions: List of Socratic questions to guide learning
- code_examples: Python code examples when relevant
- next_steps: Suggested next learning actions
- engagement_score: How engaged the student seems (0.0-1.0)
"""

    async def teach(self, question: str, context: TeachingContext) -> TeachingResponse:
        """Teach using adaptive, personalized approach"""
        self.update_interaction_count()

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(
                user_prompt=self._create_teaching_prompt(question, context)
            )
            return result.data
        else:
            return await self._teach_with_langchain(question, context)

    async def _teach_with_langchain(self, question: str, context: TeachingContext) -> TeachingResponse:
        """Fallback teaching using LangChain"""
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=self._create_teaching_prompt(question, context))
        ]

        response = await self.llm.ainvoke(messages)

        # Parse response
        try:
            content = response.content.strip()
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
            else:
                json_part = content

            data = json.loads(json_part)
            return TeachingResponse(**data, tutor_name=self.personality.name)
        except:
            # Fallback response
            return TeachingResponse(
                content=response.content,
                response_type="explanation",
                follow_up_questions=["Does this make sense?", "Would you like an example?"],
                tutor_name=self.personality.name
            )

    def _create_teaching_prompt(self, question: str, context: TeachingContext) -> str:
        """Create context-aware teaching prompt"""
        return f"""
Student Question: {question}

Student Context:
- Level: {context.user_level}
- Learning Style: {context.learning_style}
- Current Topic: {context.current_topic or 'Not specified'}
- Mastered Topics: {', '.join(context.mastered_topics) if context.mastered_topics else 'None yet'}
- Current Struggles: {', '.join(context.current_struggles) if context.current_struggles else 'None identified'}

Adapt your response to their level and learning style. If they're struggling, simplify. If they're excelling, challenge them!

Remember to be {self.personality.name} - {self.personality.teaching_style}!
"""

    async def assess(self, user_response: str, context: TeachingContext) -> Dict[str, Any]:
        """Assess student understanding"""
        assessment_prompt = f"""
Assess this student's response to check their understanding:

Student Response: {user_response}
Topic: {context.current_topic}
Student Level: {context.user_level}

Provide assessment in JSON format:
{{
    "understanding_score": 0.0-1.0,
    "strengths": ["what they got right"],
    "gaps": ["what they're missing"],
    "feedback": "encouraging feedback",
    "next_concept": "what to learn next"
}}
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=assessment_prompt)
            return json.loads(result.data.content)
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=assessment_prompt)
            ]
            response = await self.llm.ainvoke(messages)

            try:
                content = response.content.strip()
                if "```json" in content:
                    json_part = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_part = content
                return json.loads(json_part)
            except:
                return {
                    "understanding_score": 0.7,
                    "strengths": ["Engaged with the material"],
                    "gaps": ["Needs more practice"],
                    "feedback": "Great effort! Keep practicing!",
                    "next_concept": "Continue with current topic"
                }

    async def generate_practice(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Generate practice problems"""
        return {
            "problem": f"Practice problem for {topic}",
            "hints": ["Think about the fundamentals", "Break it into steps"],
            "solution_guide": "Step-by-step solution would be provided"
        }


class NeuralTutor(BaseTutor):
    """
    Neural - The deep learning enthusiast
    Specializes in: Neural networks, deep learning, PyTorch/TensorFlow
    """

    def __init__(self, api_key: str):
        personality = TutorPersonality(
            name="Neural",
            specialty="Deep Learning & Neural Networks",
            teaching_style="Build intuition through visualization and hands-on coding",
            personality_traits=[
                "Visual learner advocate",
                "Code-first approach",
                "Gradient descent jokes",
                "Loves architecture diagrams"
            ],
            catchphrase="Let's train some neurons! 🧠",
            emoji="🧠"
        )
        super().__init__(personality)

        if PYDANTIC_AI_AVAILABLE:
            self.agent = Agent(
                model=GeminiModel(model='gemini-2.0-flash-exp', api_key=api_key),
                result_type=TeachingResponse,
                system_prompt=self._get_system_prompt()
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash-exp",
                temperature=0.7
            )

    def _get_system_prompt(self) -> str:
        return f"""You are {self.personality.name}, an enthusiastic deep learning expert!

Personality: {', '.join(self.personality.personality_traits)}
Catchphrase: "{self.personality.catchphrase}"

Teaching Style:
1. Start with visual intuition (imagine network diagrams)
2. Show code examples early and often
3. Use PyTorch/TensorFlow syntax
4. Explain forward AND backward pass
5. Make gradient descent fun!
6. Connect to cutting-edge research

Tone:
- Energetic and passionate about neural nets
- Use brain/neuron emojis 🧠 ⚡ 🎯
- Reference famous architectures (ResNet, Transformers, etc.)
- Celebrate "aha!" moments

Always provide:
1. Conceptual explanation
2. Visual description (what would the diagram show?)
3. Code implementation
4. Common pitfalls to avoid
"""

    async def teach(self, question: str, context: TeachingContext) -> TeachingResponse:
        """Teach deep learning concepts"""
        self.update_interaction_count()

        prompt = f"""
Question: {question}

Student Level: {context.user_level}
Learning Style: {context.learning_style}

Explain this deep learning concept with:
1. Intuitive explanation (what's happening?)
2. Visual description (how to visualize it?)
3. Code example (PyTorch preferred)
4. Practical tips

Response as JSON with: content, response_type, code_examples, follow_up_questions, next_steps
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=prompt)
            return result.data
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)

            try:
                content = response.content.strip()
                if "```json" in content:
                    json_part = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_part = content
                data = json.loads(json_part)
                return TeachingResponse(**data, tutor_name=self.personality.name)
            except:
                return TeachingResponse(
                    content=response.content,
                    response_type="explanation",
                    tutor_name=self.personality.name
                )

    async def assess(self, user_response: str, context: TeachingContext) -> Dict[str, Any]:
        """Assess deep learning understanding"""
        return {
            "understanding_score": 0.75,
            "strengths": ["Good grasp of concept"],
            "gaps": ["Practice implementation"],
            "feedback": "Nice! Now implement it in code!",
            "next_concept": "Advanced architectures"
        }

    async def generate_practice(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Generate neural network exercises"""
        return {
            "problem": f"Build a neural network for {topic}",
            "starter_code": "import torch\nimport torch.nn as nn\n\nclass MyNetwork(nn.Module):\n    ...",
            "tests": ["Test forward pass", "Test backward pass"]
        }


class VisionTutor(BaseTutor):
    """
    Vision - The computer vision specialist
    Specializes in: CNNs, image processing, object detection
    """

    def __init__(self, api_key: str):
        personality = TutorPersonality(
            name="Vision",
            specialty="Computer Vision",
            teaching_style="Visual-first learning with image examples",
            personality_traits=[
                "Loves showing visual examples",
                "CNN architecture expert",
                "Dataset enthusiast",
                "Practical applications focus"
            ],
            catchphrase="Let's see the world through AI's eyes! 👁️",
            emoji="👁️"
        )
        super().__init__(personality)

        if PYDANTIC_AI_AVAILABLE:
            self.agent = Agent(
                model=GeminiModel(model='gemini-2.0-flash-exp', api_key=api_key),
                result_type=TeachingResponse,
                system_prompt=self._get_system_prompt()
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash-exp",
                temperature=0.7
            )

    def _get_system_prompt(self) -> str:
        return f"""You are {self.personality.name}, a computer vision expert!

Specialty: CNNs, Image Processing, Object Detection, Segmentation

Teaching Approach:
1. Always relate to how humans see
2. Explain convolution with visual examples
3. Reference real applications (face recognition, self-driving cars)
4. Show dataset examples (ImageNet, COCO, etc.)
5. Discuss data augmentation techniques

Make vision tasks tangible and visual! Use 👁️ 📸 🖼️ emojis.
"""

    async def teach(self, question: str, context: TeachingContext) -> TeachingResponse:
        """Teach computer vision"""
        self.update_interaction_count()

        prompt = f"""
Question: {question}
Level: {context.user_level}

Teach this computer vision concept with visual analogies and practical examples.
Include code using OpenCV or PyTorch vision libraries.

JSON response with: content, code_examples, follow_up_questions, next_steps
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=prompt)
            return result.data
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)

            try:
                content = response.content.strip()
                if "```json" in content:
                    json_part = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_part = content
                data = json.loads(json_part)
                return TeachingResponse(**data, tutor_name=self.personality.name)
            except:
                return TeachingResponse(
                    content=response.content,
                    response_type="explanation",
                    tutor_name=self.personality.name
                )

    async def assess(self, user_response: str, context: TeachingContext) -> Dict[str, Any]:
        return {
            "understanding_score": 0.8,
            "strengths": ["Good visual intuition"],
            "gaps": ["Need more practice with CNNs"],
            "feedback": "Great! Try implementing a CNN next!",
            "next_concept": "Advanced architectures like ResNet"
        }

    async def generate_practice(self, topic: str, difficulty: str) -> Dict[str, Any]:
        return {
            "problem": f"Computer vision task: {topic}",
            "dataset": "Sample dataset provided",
            "hints": ["Use data augmentation", "Try transfer learning"]
        }


class NLPTutor(BaseTutor):
    """
    Linguist - The NLP and language model expert
    Specializes in: NLP, transformers, LLMs
    """

    def __init__(self, api_key: str):
        personality = TutorPersonality(
            name="Linguist",
            specialty="Natural Language Processing",
            teaching_style="Language-focused with transformer deep dives",
            personality_traits=[
                "Transformer architecture fan",
                "Loves word embeddings",
                "BERT/GPT enthusiast",
                "Attention mechanism expert"
            ],
            catchphrase="Words are just vectors in disguise! 📝",
            emoji="📝"
        )
        super().__init__(personality)

        if PYDANTIC_AI_AVAILABLE:
            self.agent = Agent(
                model=GeminiModel(model='gemini-2.0-flash-exp', api_key=api_key),
                result_type=TeachingResponse,
                system_prompt=self._get_system_prompt()
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model="gemini-2.0-flash-exp",
                temperature=0.7
            )

    def _get_system_prompt(self) -> str:
        return f"""You are {self.personality.name}, an NLP specialist!

Expertise: Transformers, BERT, GPT, Word Embeddings, Attention Mechanisms

Teaching Style:
1. Explain how language becomes numbers
2. Deep dive into attention mechanisms
3. Use HuggingFace examples
4. Reference latest LLM developments
5. Make transformers intuitive

Use 📝 💬 🤖 emojis. Reference famous models (BERT, GPT, T5, etc.)
"""

    async def teach(self, question: str, context: TeachingContext) -> TeachingResponse:
        self.update_interaction_count()

        prompt = f"""
Question: {question}
Level: {context.user_level}

Explain this NLP concept with:
1. Linguistic intuition
2. How it works technically
3. HuggingFace code example
4. Real applications

JSON with: content, code_examples, follow_up_questions, next_steps
"""

        if PYDANTIC_AI_AVAILABLE:
            result = await self.agent.run(user_prompt=prompt)
            return result.data
        else:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)

            try:
                content = response.content.strip()
                if "```json" in content:
                    json_part = content.split("```json")[1].split("```")[0].strip()
                else:
                    json_part = content
                data = json.loads(json_part)
                return TeachingResponse(**data, tutor_name=self.personality.name)
            except:
                return TeachingResponse(
                    content=response.content,
                    response_type="explanation",
                    tutor_name=self.personality.name
                )

    async def assess(self, user_response: str, context: TeachingContext) -> Dict[str, Any]:
        return {
            "understanding_score": 0.75,
            "strengths": ["Good grasp of concepts"],
            "gaps": ["Transformer internals"],
            "feedback": "Excellent! Dive deeper into attention!",
            "next_concept": "Multi-head attention"
        }

    async def generate_practice(self, topic: str, difficulty: str) -> Dict[str, Any]:
        return {
            "problem": f"NLP task: {topic}",
            "dataset": "Text corpus provided",
            "hints": ["Use pre-trained models", "Fine-tune on your data"]
        }


# Tutor Factory
def get_tutor(tutor_name: str, api_key: str) -> BaseTutor:
    """Factory function to get appropriate tutor"""
    tutors = {
        "Prof. Data": ProfDataTutor,
        "Neural": NeuralTutor,
        "Vision": VisionTutor,
        "Linguist": NLPTutor
    }

    tutor_class = tutors.get(tutor_name, ProfDataTutor)
    return tutor_class(api_key)
