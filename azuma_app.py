"""
Azuma AI - Agentic AI Tutor for Machine Learning
Main Streamlit Application

An innovative, engaging, and fun platform for learning AI/ML with intelligent tutors!
"""

import streamlit as st
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
import json

# Import Azuma AI components
from azuma.agents.teaching_agents import get_tutor, TeachingContext
from azuma.models.user import UserProfile, LearningLevel, LearningStyle
from azuma.gamification.achievement_system import GamificationEngine
from azuma.learning.adaptive_path import AdaptiveLearningEngine
from azuma.knowledge.knowledge_graph import KnowledgeGraph

load_dotenv()

# Page config
st.set_page_config(
    page_title="Azuma AI - Learn AI/ML with AI Tutors",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }

    .tutor-card {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    }

    .badge-display {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: bold;
    }

    .progress-bar {
        height: 25px;
        border-radius: 12px;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    }

    .stat-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.user_profile = None
        st.session_state.current_tutor = "Prof. Data"
        st.session_state.chat_history = []
        st.session_state.gamification = GamificationEngine()
        st.session_state.knowledge_graph = KnowledgeGraph()
        st.session_state.current_lesson = None
        st.session_state.adaptive_engine = None

        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.session_state.api_key = api_key
        else:
            st.session_state.api_key = None


def render_header():
    """Render the main header"""
    st.markdown('<div class="main-header">🤖 Azuma AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Intelligent AI/ML Learning Companion</p>',
        unsafe_allow_html=True
    )


def render_user_registration():
    """Render user registration form"""
    st.title("Welcome to Azuma AI! 🎓")
    st.write("Let's create your personalized learning profile!")

    with st.form("registration_form"):
        col1, col2 = st.columns(2)

        with col1:
            username = st.text_input("Username", placeholder="Enter your name")
            email = st.text_input("Email (optional)", placeholder="your.email@example.com")

            level = st.selectbox(
                "Current Level",
                options=[level.value for level in LearningLevel],
                format_func=lambda x: x.title()
            )

        with col2:
            learning_style = st.selectbox(
                "Preferred Learning Style",
                options=[style.value for style in LearningStyle],
                format_func=lambda x: x.replace("_", " ").title()
            )

            goals = st.multiselect(
                "Learning Goals",
                options=[
                    "Become an ML Engineer",
                    "Build AI Projects",
                    "Understand Deep Learning",
                    "Master Computer Vision",
                    "Excel in NLP",
                    "Learn Reinforcement Learning",
                    "Deploy ML Models"
                ]
            )

        interests = st.multiselect(
            "Specific Interests",
            options=[
                "Neural Networks",
                "Computer Vision",
                "Natural Language Processing",
                "Reinforcement Learning",
                "MLOps",
                "Math Foundations",
                "Practical Projects"
            ]
        )

        submitted = st.form_submit_button("Start Learning! 🚀", use_container_width=True)

        if submitted and username:
            # Create user profile
            user_id = f"user_{datetime.now().timestamp()}"
            st.session_state.user_profile = UserProfile(
                user_id=user_id,
                username=username,
                email=email if email else None,
                level=LearningLevel(level),
                learning_style=LearningStyle(learning_style),
                goals=goals,
                interests=interests
            )
            st.success(f"Welcome aboard, {username}! 🎉")
            st.rerun()


def render_sidebar():
    """Render the sidebar with user stats and navigation"""
    if not st.session_state.user_profile:
        return

    user = st.session_state.user_profile

    with st.sidebar:
        st.markdown("### 👤 Your Profile")
        st.write(f"**{user.username}**")

        # Level and Progress
        level_info = st.session_state.gamification.get_user_level(user.total_points)
        st.markdown(f"**Level {level_info['level']}**: {level_info['title']}")

        # Progress bar
        progress = level_info['progress_to_next']
        st.progress(progress)
        st.caption(f"{int(progress * 100)}% to next level")

        st.markdown("---")

        # Stats
        st.markdown("### 📊 Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Points", user.total_points)
            st.metric("Streak", f"{user.current_streak} 🔥")

        with col2:
            st.metric("Lessons", len(user.completed_lessons))
            st.metric("Badges", len(user.badges))

        st.markdown("---")

        # Tutor Selection
        st.markdown("### 🎓 Choose Your Tutor")

        tutors = {
            "Prof. Data": {"emoji": "📊", "specialty": "ML Fundamentals"},
            "Neural": {"emoji": "🧠", "specialty": "Deep Learning"},
            "Vision": {"emoji": "👁️", "specialty": "Computer Vision"},
            "Linguist": {"emoji": "📝", "specialty": "NLP"}
        }

        for tutor_name, info in tutors.items():
            if st.button(
                f"{info['emoji']} {tutor_name}",
                key=f"tutor_{tutor_name}",
                use_container_width=True,
                type="primary" if st.session_state.current_tutor == tutor_name else "secondary"
            ):
                st.session_state.current_tutor = tutor_name
                st.rerun()

        st.caption(f"Current: **{tutors[st.session_state.current_tutor]['specialty']}**")

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigate")

        if st.button("📚 Learning Dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"

        if st.button("💬 Chat with Tutor", use_container_width=True):
            st.session_state.current_page = "chat"

        if st.button("🗺️ Knowledge Map", use_container_width=True):
            st.session_state.current_page = "knowledge_map"

        if st.button("🏆 Achievements", use_container_width=True):
            st.session_state.current_page = "achievements"


def render_dashboard():
    """Render the learning dashboard"""
    user = st.session_state.user_profile

    st.title("📚 Your Learning Dashboard")

    # Daily Challenge
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Daily Challenge")
        challenge = st.session_state.gamification.get_daily_challenge()
        st.write(f"**{challenge['title']}**")
        st.write(challenge['description'])
        st.write(f"Reward: **{challenge['reward_points']} points**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Progress This Week")
        st.metric("Lessons Completed", len(user.completed_lessons))
        st.metric("Topics Mastered", len(user.mastered_topics))
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("#### 🏅 Latest Badges")
        if user.badges:
            for badge_id in user.badges[-3:]:
                badge = st.session_state.gamification.badges.get(badge_id)
                if badge:
                    st.write(f"{badge.icon} {badge.name}")
        else:
            st.write("Complete lessons to earn badges!")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Recommended Topics
    st.markdown("### 🎯 Recommended for You")

    mastered = user.mastered_topics
    next_topics = st.session_state.knowledge_graph.get_next_topics(mastered, limit=5)

    if next_topics:
        cols = st.columns(len(next_topics))
        for idx, topic in enumerate(next_topics):
            with cols[idx]:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 1rem; text-align: center;">
                    <h4>{topic.title}</h4>
                    <p><strong>{topic.difficulty.title()}</strong></p>
                    <p>{topic.category.replace('_', ' ').title()}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Start Learning", key=f"start_{topic.node_id}"):
                    st.session_state.current_topic = topic
                    st.session_state.current_page = "chat"
                    st.rerun()


def render_chat_interface():
    """Render the chat interface with AI tutor"""
    user = st.session_state.user_profile
    tutor_name = st.session_state.current_tutor

    st.title(f"💬 Chat with {tutor_name}")

    # Display tutor info
    tutor_info = {
        "Prof. Data": {
            "emoji": "📊",
            "description": "Your friendly ML fundamentals expert. Patient, thorough, and loves real-world examples!"
        },
        "Neural": {
            "emoji": "🧠",
            "description": "Deep learning enthusiast! Visual learner who loves showing you neural network diagrams."
        },
        "Vision": {
            "emoji": "👁️",
            "description": "Computer vision specialist. Sees the world through AI's eyes!"
        },
        "Linguist": {
            "emoji": "📝",
            "description": "NLP expert who knows that words are just vectors in disguise!"
        }
    }

    info = tutor_info.get(tutor_name, tutor_info["Prof. Data"])

    st.info(f"{info['emoji']} {info['description']}")

    # Chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar=info["emoji"]):
                    st.write(message["content"])

                    # Show code examples if any
                    if "code_examples" in message and message["code_examples"]:
                        for example in message["code_examples"]:
                            st.code(example.get("code", ""), language=example.get("language", "python"))
                            if example.get("explanation"):
                                st.caption(example["explanation"])

                    # Show follow-up questions
                    if "follow_up_questions" in message and message["follow_up_questions"]:
                        with st.expander("💡 Think about these questions"):
                            for q in message["follow_up_questions"]:
                                st.write(f"• {q}")

    # Chat input
    user_input = st.chat_input("Ask me anything about AI/ML...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Get tutor response
        with st.spinner(f"{tutor_name} is thinking..."):
            try:
                # Create context
                context = TeachingContext(
                    user_id=user.user_id,
                    user_level=user.level.value,
                    learning_style=user.learning_style.value,
                    mastered_topics=list(user.mastered_topics),
                    session_history=st.session_state.chat_history[-5:]  # Last 5 messages
                )

                # Get tutor
                if not st.session_state.api_key:
                    st.error("API key not configured. Please add GOOGLE_API_KEY to your .env file.")
                    return

                tutor = get_tutor(tutor_name, st.session_state.api_key)

                # Get response (Note: This is async, but we'll run it sync for Streamlit)
                # In production, use proper async handling
                response = asyncio.run(tutor.teach(user_input, context))

                # Add tutor response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "code_examples": response.code_examples,
                    "follow_up_questions": response.follow_up_questions,
                    "next_steps": response.next_steps
                })

                # Award points for interaction
                user.total_points += st.session_state.gamification.calculate_points_for_activity(
                    "tutor_interaction"
                )

                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure your GOOGLE_API_KEY is set in the .env file.")


def render_knowledge_map():
    """Render the knowledge graph visualization"""
    user = st.session_state.user_profile

    st.title("🗺️ Your Knowledge Map")

    st.write("Visualize your AI/ML learning journey!")

    # Get knowledge graph data
    user_mastery = {topic: 0.8 for topic in user.mastered_topics}
    stats = st.session_state.knowledge_graph.get_knowledge_stats(user_mastery)

    # Overall progress
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Topics", stats["total_topics"])

    with col2:
        st.metric("Mastered", stats["mastered_topics"], delta="+5")

    with col3:
        st.metric("In Progress", stats["in_progress_topics"])

    with col4:
        progress_pct = int(stats["overall_progress"] * 100)
        st.metric("Overall Progress", f"{progress_pct}%")

    st.markdown("---")

    # Category progress
    st.markdown("### 📚 Progress by Category")

    categories = stats.get("categories", {})

    for category, cat_stats in categories.items():
        st.markdown(f"#### {category.replace('_', ' ').title()}")

        progress = cat_stats["completion_rate"]
        st.progress(progress)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"✅ Mastered: {cat_stats['mastered']}")
        with col2:
            st.write(f"📖 Learning: {cat_stats['in_progress']}")
        with col3:
            st.write(f"🔒 Locked: {cat_stats['locked']}")

        st.markdown("---")


def render_achievements():
    """Render achievements and gamification"""
    user = st.session_state.user_profile

    st.title("🏆 Your Achievements")

    # Level info
    level_info = st.session_state.gamification.get_user_level(user.total_points)

    st.markdown(f"### Level {level_info['level']}: {level_info['title']}")

    progress = level_info['progress_to_next']
    st.progress(progress)
    st.write(f"{user.total_points} points • {int(progress * 100)}% to next level")

    if level_info['perks']:
        st.markdown("**Unlocked Perks:**")
        for perk in level_info['perks']:
            st.write(f"✨ {perk}")

    st.markdown("---")

    # Badges
    st.markdown("### 🏅 Your Badges")

    if user.badges:
        cols = st.columns(4)
        for idx, badge_id in enumerate(user.badges):
            badge = st.session_state.gamification.badges.get(badge_id)
            if badge:
                with cols[idx % 4]:
                    st.markdown(f"""
                    <div class="badge-display" style="text-align: center; padding: 1rem;">
                        <h1>{badge.icon}</h1>
                        <h4>{badge.name}</h4>
                        <p>{badge.description}</p>
                        <small>{badge.points} points</small>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Complete lessons and challenges to earn badges!")

    st.markdown("---")

    # All available badges
    st.markdown("### 📋 All Badges")

    badge_types = ["milestone", "streak", "mastery", "special", "completion", "challenge"]

    for badge_type in badge_types:
        st.markdown(f"#### {badge_type.title()} Badges")

        type_badges = [b for b in st.session_state.gamification.badges.values()
                      if b.badge_type.value == badge_type]

        cols = st.columns(3)
        for idx, badge in enumerate(type_badges):
            with cols[idx % 3]:
                earned = badge.badge_id in user.badges

                if earned:
                    st.success(f"{badge.icon} **{badge.name}** ✓")
                else:
                    st.info(f"{badge.icon} {badge.name}")

                st.caption(badge.description)
                st.caption(f"Requirement: {badge.requirement}")


def main():
    """Main application"""
    initialize_session()
    render_header()

    # Check for API key
    if not st.session_state.api_key:
        st.error("⚠️ API Key Not Found!")
        st.warning("Please create a `.env` file in the project root with your Google API key:")
        st.code("GOOGLE_API_KEY=your_actual_google_api_key_here")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        st.stop()

    # User registration or main app
    if not st.session_state.user_profile:
        render_user_registration()
    else:
        render_sidebar()

        # Page routing
        current_page = getattr(st.session_state, 'current_page', 'dashboard')

        if current_page == 'dashboard':
            render_dashboard()
        elif current_page == 'chat':
            render_chat_interface()
        elif current_page == 'knowledge_map':
            render_knowledge_map()
        elif current_page == 'achievements':
            render_achievements()

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Made with ❤️ by Azuma AI | Powered by PydanticAI & Google Gemini</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
