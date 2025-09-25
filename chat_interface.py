"""
Chat Interface Module for the Agentic AI Actions Co-pilot system.

This module handles chat-style interactions, message formatting, and conversational 
flows for the SIGMA-integrated agentic AI system.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import streamlit as st

from business_models import (
    CompletedAction, 
    AgentRecommendation, 
    ProposedChange,
    ProcessingStatus,
    AgentStatus,
    ActionOutcome,
    ConfidenceLevel
)


class MessageRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"
    AGENT = "agent"


class MessageType(str, Enum):
    """Types of chat messages"""
    TEXT = "text"
    ACTION_COMPLETION = "action_completion"
    ANALYSIS_RESULT = "analysis_result"
    CHANGE_PROPOSAL = "change_proposal"
    CHANGE_APPLIED = "change_applied"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


class ChatMessage:
    """Structured chat message class"""
    
    def __init__(
        self, 
        role: MessageRole,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        metadata: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ):
        self.role = role
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        self.id = f"{self.timestamp.timestamp()}_{role.value}"

    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "message_type": self.message_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create message from dictionary"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class ChatInterface:
    """Main chat interface handler"""
    
    def __init__(self):
        self.session_key = "chat_history"
        self.typing_key = "is_typing"
        self.current_action_key = "current_action"
        
    def initialize_chat(self):
        """Initialize chat session state"""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
            # Add welcome message
            welcome_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content="👋 Hi! I'm Horo, your AI business co-pilot. I'll help analyze your completed actions and suggest intelligent updates to your business model canvas.",
                message_type=MessageType.TEXT
            )
            st.session_state[self.session_key].append(welcome_msg)
        
        if self.typing_key not in st.session_state:
            st.session_state[self.typing_key] = False
            
        if self.current_action_key not in st.session_state:
            st.session_state[self.current_action_key] = None

    def add_message(
        self, 
        role: MessageRole, 
        content: str, 
        message_type: MessageType = MessageType.TEXT,
        metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """Add a message to chat history"""
        message = ChatMessage(role, content, message_type, metadata)
        st.session_state[self.session_key].append(message)
        return message

    def get_messages(self) -> List[ChatMessage]:
        """Get all chat messages"""
        return st.session_state.get(self.session_key, [])

    def clear_chat(self):
        """Clear chat history"""
        st.session_state[self.session_key] = []
        self.initialize_chat()

    def set_typing(self, is_typing: bool):
        """Set typing indicator state"""
        st.session_state[self.typing_key] = is_typing

    def is_typing(self) -> bool:
        """Check if AI is typing"""
        return st.session_state.get(self.typing_key, False)


class MessageFormatter:
    """Formats different types of messages for display"""
    
    @staticmethod
    def format_action_completion(action: CompletedAction) -> str:
        """Format action completion message"""
        outcome_emoji = {
            ActionOutcome.SUCCESSFUL: "✅",
            ActionOutcome.FAILED: "❌", 
            ActionOutcome.INCONCLUSIVE: "❓"
        }
        
        emoji = outcome_emoji.get(action.outcome, "📋")
        
        return f"""
{emoji} **Action Completed: {action.title}**

**Outcome**: {action.outcome.value.title()}
**Completed**: {action.completion_date.strftime('%Y-%m-%d')}

**Description**: {action.description[:200]}{'...' if len(action.description) > 200 else ''}

🤖 *Analyzing business implications...*
        """.strip()

    @staticmethod
    def format_analysis_result(recommendation: AgentRecommendation) -> str:
        """Format analysis result message"""
        confidence_emoji = {
            ConfidenceLevel.HIGH: "🎯",
            ConfidenceLevel.MEDIUM: "⚡",
            ConfidenceLevel.LOW: "💭"
        }
        
        emoji = confidence_emoji.get(recommendation.confidence_level, "🤖")
        
        changes_text = ""
        if recommendation.proposed_changes:
            changes_text = f"\n\n💡 **Found {len(recommendation.proposed_changes)} potential improvements** to your business model canvas."
        
        next_actions_text = ""
        if recommendation.next_actions:
            next_actions_text = f"\n\n🎯 **Suggested {len(recommendation.next_actions)} follow-up actions** for validation."

        return f"""
{emoji} **Analysis Complete**

**Confidence Level**: {recommendation.confidence_level.value.title()}

**Key Insight**: {recommendation.reasoning}
{changes_text}
{next_actions_text}

Would you like to review the detailed recommendations?
        """.strip()

    @staticmethod
    def format_change_proposal(changes: List[ProposedChange]) -> str:
        """Format change proposal message"""
        if not changes:
            return "💭 No changes recommended based on current analysis."
        
        high_confidence = [c for c in changes if c.confidence_score >= 0.8]
        medium_confidence = [c for c in changes if 0.6 <= c.confidence_score < 0.8]
        low_confidence = [c for c in changes if c.confidence_score < 0.6]
        
        sections_affected = len(set(c.canvas_section for c in changes))
        
        message = f"💡 **Proposed {len(changes)} changes** across {sections_affected} canvas sections:\n\n"
        
        if high_confidence:
            message += f"🎯 **{len(high_confidence)} high-confidence changes** (ready for auto-apply)\n"
        
        if medium_confidence:
            message += f"⚡ **{len(medium_confidence)} medium-confidence changes** (recommend review)\n"
        
        if low_confidence:
            message += f"💭 **{len(low_confidence)} low-confidence changes** (requires careful consideration)\n"
        
        message += "\n📋 Use the **Analysis Results** section below to review details and apply changes."
        
        return message

    @staticmethod
    def format_change_applied(changes: List[ProposedChange], auto_applied: bool = False) -> str:
        """Format change applied confirmation"""
        mode = "automatically" if auto_applied else "successfully"
        emoji = "🤖" if auto_applied else "✅"
        
        sections = list(set(c.canvas_section.replace('_', ' ').title() for c in changes))
        sections_text = ", ".join(sections[:3])
        if len(sections) > 3:
            sections_text += f" and {len(sections) - 3} more"
        
        return f"""
{emoji} **Changes Applied {mode.title()}!**

Updated **{len(changes)} elements** in: {sections_text}

Your business model canvas has been updated with the latest insights. The changes are now active and saved to your profile.

🔄 *Ready to analyze your next completed action.*
        """.strip()

    @staticmethod
    def format_agent_status(status: ProcessingStatus) -> str:
        """Format agent processing status"""
        agents = [
            ("Action Detection", status.action_detection_status),
            ("Outcome Analysis", status.outcome_analysis_status), 
            ("Canvas Updates", status.canvas_update_status),
            ("Next Steps", status.next_step_status)
        ]
        
        status_emoji = {
            AgentStatus.PENDING: "⏳",
            AgentStatus.RUNNING: "🔄",
            AgentStatus.COMPLETED: "✅",
            AgentStatus.FAILED: "❌"
        }
        
        message = "🤖 **Agent Processing Status**:\n\n"
        
        for agent_name, agent_status in agents:
            emoji = status_emoji.get(agent_status, "❓")
            message += f"{emoji} {agent_name}: {agent_status.value.title()}\n"
        
        return message

    @staticmethod
    def format_error_message(error: str) -> str:
        """Format error message"""
        return f"""
❌ **Processing Error**

Something went wrong during analysis: {error}

💡 **Suggestions**:
- Check your internet connection
- Verify API credentials are configured
- Try again with a different action
- Contact support if the problem persists
        """.strip()


class ChatComponentRenderer:
    """Renders chat components with SIGMA styling"""
    
    @staticmethod
    def render_message(message: ChatMessage):
        """Render a chat message with proper styling"""
        # Determine message styling based on role
        if message.role == MessageRole.USER:
            MessageComponentRenderer.render_user_message(message)
        elif message.role == MessageRole.ASSISTANT:
            MessageComponentRenderer.render_assistant_message(message)
        elif message.role == MessageRole.SYSTEM:
            MessageComponentRenderer.render_system_message(message)
        elif message.role == MessageRole.AGENT:
            MessageComponentRenderer.render_agent_message(message)

    @staticmethod
    def render_typing_indicator():
        """Render typing indicator"""
        st.markdown("""
        <div class="typing-indicator">
            🤖 Horo is analyzing your action outcome
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_chat_input():
        """Render chat input area"""
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Message Horo...",
                placeholder="Describe a completed action or ask a question...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.button("📤 Send", use_container_width=True)
        
        return user_input, send_clicked


class MessageComponentRenderer:
    """Individual message component renderers"""
    
    @staticmethod
    def render_user_message(message: ChatMessage):
        """Render user message"""
        timestamp = message.timestamp.strftime("%H:%M")
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message-content">{message.content}</div>
            <div class="message-time">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_assistant_message(message: ChatMessage):
        """Render assistant message"""
        timestamp = message.timestamp.strftime("%H:%M")
        
        # Special handling for different message types
        if message.message_type == MessageType.ANALYSIS_RESULT:
            MessageComponentRenderer._render_analysis_message(message, timestamp)
        elif message.message_type == MessageType.CHANGE_PROPOSAL:
            MessageComponentRenderer._render_proposal_message(message, timestamp)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-content">{message.content}</div>
                <div class="message-time">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def render_system_message(message: ChatMessage):
        """Render system message"""
        st.markdown(f"""
        <div class="chat-message system">
            <div class="message-content">{message.content}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_agent_message(message: ChatMessage):
        """Render agent processing message"""
        st.markdown(f"""
        <div class="chat-message agent">
            <div class="message-content">🤖 {message.content}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _render_analysis_message(message: ChatMessage, timestamp: str):
        """Render analysis result message with enhanced styling"""
        # Extract confidence level from metadata if available
        confidence = message.metadata.get('confidence_level', 'medium')
        confidence_class = f"confidence-{confidence}"
        
        st.markdown(f"""
        <div class="chat-message assistant {confidence_class}">
            <div class="message-content">{message.content}</div>
            <div class="message-time">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _render_proposal_message(message: ChatMessage, timestamp: str):
        """Render change proposal message"""
        st.markdown(f"""
        <div class="chat-message assistant proposal">
            <div class="message-content">{message.content}</div>
            <div class="message-actions">
                <button class="action-btn primary">Review Changes</button>
                <button class="action-btn secondary">Ask Questions</button>
            </div>
            <div class="message-time">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)


class ConversationFlow:
    """Manages conversation flow and state"""
    
    def __init__(self, chat_interface: ChatInterface):
        self.chat = chat_interface
        self.formatter = MessageFormatter()
    
    def handle_action_completion(self, action: CompletedAction):
        """Handle action completion workflow"""
        # Add user message about action completion
        self.chat.add_message(
            MessageRole.USER,
            f"I just completed: {action.title}",
            MessageType.ACTION_COMPLETION,
            {"action_id": action.id}
        )
        
        # Add system acknowledgment
        self.chat.add_message(
            MessageRole.SYSTEM,
            "🤖 Action detected. Initiating 4-agent analysis workflow...",
            MessageType.STATUS_UPDATE
        )
        
        # Set current action
        st.session_state[self.chat.current_action_key] = action

    def handle_analysis_complete(self, recommendation: AgentRecommendation):
        """Handle analysis completion"""
        # Add analysis result message
        content = self.formatter.format_analysis_result(recommendation)
        self.chat.add_message(
            MessageRole.ASSISTANT,
            content,
            MessageType.ANALYSIS_RESULT,
            {
                "confidence_level": recommendation.confidence_level.value,
                "changes_count": len(recommendation.proposed_changes),
                "next_actions_count": len(recommendation.next_actions)
            }
        )
        
        # Add change proposal if changes exist
        if recommendation.proposed_changes:
            proposal_content = self.formatter.format_change_proposal(recommendation.proposed_changes)
            self.chat.add_message(
                MessageRole.ASSISTANT,
                proposal_content,
                MessageType.CHANGE_PROPOSAL,
                {"changes": [c.dict() for c in recommendation.proposed_changes]}
            )

    def handle_changes_applied(self, changes: List[ProposedChange], auto_applied: bool = False):
        """Handle changes applied confirmation"""
        content = self.formatter.format_change_applied(changes, auto_applied)
        self.chat.add_message(
            MessageRole.SYSTEM,
            content,
            MessageType.CHANGE_APPLIED,
            {
                "changes_count": len(changes),
                "auto_applied": auto_applied,
                "sections_affected": len(set(c.canvas_section for c in changes))
            }
        )

    def handle_user_question(self, question: str):
        """Handle user question"""
        self.chat.add_message(MessageRole.USER, question, MessageType.TEXT)
        
        # Generate appropriate response based on context
        response = self._generate_contextual_response(question)
        self.chat.add_message(MessageRole.ASSISTANT, response, MessageType.TEXT)

    def _generate_contextual_response(self, question: str) -> str:
        """Generate contextual response to user question"""
        question_lower = question.lower()
        
        # Business model questions
        if any(word in question_lower for word in ['business model', 'canvas', 'bmc']):
            return "I can help you understand and update your Business Model Canvas. Your current canvas shows your key partnerships, activities, and value propositions. What specific aspect would you like to explore?"
        
        # Action-related questions
        elif any(word in question_lower for word in ['action', 'experiment', 'test']):
            return "Actions and experiments are crucial for validating your business assumptions. I analyze completed actions to suggest intelligent updates to your business model. What action would you like to discuss?"
        
        # Auto-mode questions
        elif any(word in question_lower for word in ['auto', 'automatic', 'apply']):
            return "Auto-mode allows me to automatically apply high-confidence changes (>70% confidence) to your business model. You can toggle this on/off at any time. Would you like me to explain how the confidence scoring works?"
        
        # General help
        elif any(word in question_lower for word in ['help', 'how', 'what']):
            return """I'm here to help you iterate on your business model based on real action outcomes! Here's what I can do:

🎯 **Analyze Actions**: Process completed experiments and activities
🤖 **Smart Updates**: Suggest intelligent BMC updates with confidence scores  
🔄 **Auto-mode**: Apply safe changes automatically
📊 **Track Changes**: Maintain version history and audit trail

What would you like to explore first?"""
        
        # Default response
        else:
            return "I'm focused on helping you analyze action outcomes and update your business model canvas. Could you tell me more about a recent action or experiment you've completed?"


class ChatDemoScenarios:
    """Pre-built demo scenarios for chat interface"""
    
    @staticmethod
    def load_sample_conversation(chat_interface: ChatInterface):
        """Load a sample conversation for demo purposes"""
        scenarios = [
            {
                "role": MessageRole.USER,
                "content": "I just completed a pricing survey with 50 traders in Makola Market",
                "type": MessageType.ACTION_COMPLETION
            },
            {
                "role": MessageRole.SYSTEM,
                "content": "🤖 Action detected: Market pricing survey. Analyzing business implications...",
                "type": MessageType.STATUS_UPDATE
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "🎯 **Analysis Complete**\n\n**Confidence Level**: High\n\n**Key Insight**: Survey results show strong price acceptance at current 2% transaction fee, with opportunities to optimize for different customer segments.\n\n💡 **Found 3 potential improvements** to your business model canvas.\n\n🎯 **Suggested 4 follow-up actions** for validation.",
                "type": MessageType.ANALYSIS_RESULT
            },
            {
                "role": MessageRole.USER,
                "content": "That's great! What specific changes are you recommending?",
                "type": MessageType.TEXT
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "💡 **Proposed 3 changes** across 2 canvas sections:\n\n🎯 **2 high-confidence changes** (ready for auto-apply)\n⚡ **1 medium-confidence change** (recommend review)\n\n📋 Use the **Analysis Results** section below to review details and apply changes.",
                "type": MessageType.CHANGE_PROPOSAL
            }
        ]
        
        for scenario in scenarios:
            chat_interface.add_message(
                scenario["role"],
                scenario["content"],
                scenario["type"]
            )

    @staticmethod
    def get_demo_user_inputs() -> List[str]:
        """Get sample user inputs for demo"""
        return [
            "I just completed a customer survey in Lagos",
            "We tested our USSD interface with 100 rural users",
            "Our partnership with ARB Apex Bank was successful",
            "The agent network pilot in Kumasi had mixed results",
            "How does auto-mode work?",
            "What changes do you recommend for my value proposition?"
        ]


# Utility functions for integration
def initialize_chat_interface() -> ChatInterface:
    """Initialize and return chat interface"""
    chat_interface = ChatInterface()
    chat_interface.initialize_chat()
    return chat_interface


def render_chat_container(chat_interface: ChatInterface, max_height: str = "500px"):
    """Render complete chat container"""
    st.markdown(f"""
    <div class="chat-container" style="max-height: {max_height};">
    """, unsafe_allow_html=True)
    
    # Render all messages
    renderer = ChatComponentRenderer()
    for message in chat_interface.get_messages():
        renderer.render_message(message)
    
    # Show typing indicator if active
    if chat_interface.is_typing():
        renderer.render_typing_indicator()
    
    st.markdown("</div>", unsafe_allow_html=True)


def handle_chat_interaction(chat_interface: ChatInterface, user_input: str) -> bool:
    """Handle user chat interaction, return True if processed"""
    if not user_input.strip():
        return False
    
    flow = ConversationFlow(chat_interface)
    flow.handle_user_question(user_input)
    return True


# Export main classes for external use
__all__ = [
    'ChatInterface',
    'MessageFormatter', 
    'ChatComponentRenderer',
    'ConversationFlow',
    'ChatDemoScenarios',
    'ChatMessage',
    'MessageRole',
    'MessageType',
    'initialize_chat_interface',
    'render_chat_container',
    'handle_chat_interaction'
]