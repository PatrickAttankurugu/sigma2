# 🚀 Azuma AI - Quick Start Guide

## Welcome to Your AI/ML Learning Journey!

This guide will help you get started with Azuma AI in minutes.

---

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Python 3.8 or higher**
   ```bash
   python --version  # Should show 3.8+
   ```

2. **Google API Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy your API key

---

## 🔧 Installation Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- Streamlit (UI framework)
- FastAPI (Backend API)
- PydanticAI (Agentic AI framework)
- LangChain & Google Generative AI (LLM integration)
- NetworkX & Plotly (Visualizations)

### Step 2: Configure Environment

1. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```bash
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

---

## 🎮 Running Azuma AI

### Option 1: Streamlit App (Recommended for Learning)

```bash
streamlit run azuma_app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: FastAPI Backend (For API Access)

```bash
python -m azuma.backend.api_server
```

API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Option 3: Run Both (Full Experience)

Terminal 1:
```bash
python -m azuma.backend.api_server
```

Terminal 2:
```bash
streamlit run azuma_app.py
```

---

## 👤 First Time Setup

When you launch Azuma AI for the first time:

1. **Create Your Profile**
   - Enter your username
   - Select your experience level:
     - **Beginner**: New to AI/ML
     - **Intermediate**: Some ML experience
     - **Advanced**: Experienced ML practitioner
     - **Expert**: Deep ML expertise

2. **Choose Learning Style**
   - **Visual**: Prefer diagrams and visualizations
   - **Hands-on**: Learn by coding
   - **Theoretical**: Focus on concepts and math
   - **Conversational**: Learn through Q&A
   - **Mixed**: Combination of all

3. **Set Goals**
   - Become an ML Engineer
   - Build AI Projects
   - Understand Deep Learning
   - Master Computer Vision
   - Excel in NLP
   - Learn Reinforcement Learning
   - Deploy ML Models

4. **Pick Interests**
   - Neural Networks
   - Computer Vision
   - Natural Language Processing
   - Reinforcement Learning
   - MLOps
   - Math Foundations
   - Practical Projects

---

## 🎓 Meet Your Tutors

### Prof. Data 📊
**Specialty**: Machine Learning Fundamentals

**When to choose**:
- Learning ML basics
- Understanding algorithms
- Data preprocessing
- Feature engineering

**Teaching Style**: Patient, uses real-world analogies

---

### Neural 🧠
**Specialty**: Deep Learning & Neural Networks

**When to choose**:
- Learning about neural networks
- Understanding backpropagation
- Building deep learning models
- PyTorch/TensorFlow

**Teaching Style**: Visual, code-first approach

---

### Vision 👁️
**Specialty**: Computer Vision

**When to choose**:
- Image processing
- CNNs and architectures
- Object detection
- Image segmentation

**Teaching Style**: Visual examples, practical applications

---

### Linguist 📝
**Specialty**: Natural Language Processing

**When to choose**:
- Text processing
- Transformers
- BERT, GPT models
- Attention mechanisms

**Teaching Style**: Language-focused, transformer deep dives

---

## 💬 How to Chat with Tutors

1. **Select Your Tutor** from the sidebar
2. **Type Your Question** in the chat input
3. **Receive Personalized Response** with:
   - Detailed explanation
   - Code examples
   - Visual descriptions
   - Follow-up questions
   - Next learning steps

### Example Questions

**For Prof. Data:**
- "What is supervised learning?"
- "How does linear regression work?"
- "Explain the bias-variance tradeoff"

**For Neural:**
- "How do neural networks learn?"
- "Explain backpropagation step by step"
- "What are activation functions?"

**For Vision:**
- "How do CNNs process images?"
- "Explain convolutional layers"
- "What is transfer learning?"

**For Linguist:**
- "How do transformers work?"
- "Explain the attention mechanism"
- "What's the difference between BERT and GPT?"

---

## 🎮 Gamification Features

### Points System

Earn points by:
- Completing lessons: **50 points**
- Passing quizzes: **30 points**
- Perfect quiz scores: **100 points**
- Code exercises: **40 points**
- Completing projects: **500 points**
- Daily challenges: **Variable**

### Levels

Progress through 10 levels:
1. **AI Novice** (0 pts)
2. **Data Apprentice** (500 pts) - Unlock Neural
3. **ML Practitioner** (1,500 pts) - Unlock Vision
4. **Neural Architect** (3,000 pts) - Unlock Linguist
5. **AI Engineer** (5,000 pts)
6. **ML Expert** (8,000 pts)
7. **AI Master** (12,000 pts)
8. **Data Scientist** (17,000 pts)
9. **AI Researcher** (23,000 pts)
10. **AI Guru** (30,000 pts)

### Streaks

Learn daily to maintain your streak:
- 🔥 3 days: "Getting Started" badge
- ⚡ 7 days: "Week Warrior" badge
- 💎 30 days: "Monthly Master" badge
- 🏆 100 days: "Unstoppable" badge

---

## 📚 Learning Dashboard

Your dashboard shows:

- **Daily Challenge**: Complete for bonus points
- **Weekly Progress**: Lessons and topics completed
- **Latest Badges**: Recent achievements
- **Recommended Topics**: AI-suggested next steps
- **Knowledge Map**: Visual progress tracker

---

## 🗺️ Knowledge Map

The Knowledge Map visualizes:

- **All AI/ML Topics**: 40+ topics
- **Prerequisites**: What to learn first
- **Your Progress**: Mastered, in-progress, locked
- **Categories**: Fundamentals, ML, DL, CV, NLP, RL, MLOps
- **Learning Paths**: Shortest route to your goal

---

## 🏆 Achievements Page

Track your:

- **Current Level & Title**
- **Progress to Next Level**
- **Unlocked Badges**
- **Available Badges** to earn
- **Leaderboard Rank** (coming soon)

---

## 🔧 Troubleshooting

### Issue: "API Key Not Found"

**Solution**:
1. Check `.env` file exists
2. Verify `GOOGLE_API_KEY` is set
3. Ensure no quotes around the key
4. Restart the application

### Issue: "Import Error: pydantic-ai"

**Solution**:
```bash
pip install pydantic-ai
```

If still failing, the app will fallback to LangChain (already installed).

### Issue: Slow Responses

**Solution**:
- Google API has rate limits
- Wait a few seconds between questions
- Consider upgrading to Gemini Pro API

### Issue: Chat History Not Showing

**Solution**:
- Refresh the page
- Chat history is session-based (not persisted)
- Clear and restart if issues persist

---

## 💡 Pro Tips

1. **Start with Beginner Topics**: Even if experienced, review fundamentals
2. **Ask Follow-up Questions**: Tutors love deeper discussions
3. **Complete Daily Challenges**: Great for consistency
4. **Try Different Tutors**: Each has unique teaching style
5. **Use the Knowledge Map**: Visualize your journey
6. **Maintain Your Streak**: Daily practice is key
7. **Request Code Examples**: Ask "show me code for X"
8. **Share Your Goals**: Tutors adapt to your objectives

---

## 🚀 Next Steps

1. **Create Your Profile** (5 minutes)
2. **Chat with Prof. Data** (Learn ML basics)
3. **Complete First Lesson** (Earn your first badge!)
4. **Try Daily Challenge** (Bonus points)
5. **Explore Knowledge Map** (See the journey ahead)

---

## 📞 Need Help?

- **Documentation**: See README.md
- **Issues**: Report on GitHub
- **Email**: patricka.azuma@gmail.com

---

## 🎉 Ready to Start?

```bash
streamlit run azuma_app.py
```

**Happy Learning! 🤖🎓**

---

*Built with ❤️ for the AI/ML learning community*
