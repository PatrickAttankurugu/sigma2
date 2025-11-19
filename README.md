# 🤖 Azuma AI - Your Intelligent AI/ML Learning Companion

**Transform your AI/ML learning journey with intelligent, adaptive tutors powered by agentic AI!**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![PydanticAI](https://img.shields.io/badge/PydanticAI-Latest-purple.svg)](https://ai.pydantic.dev)

---

## 🌟 What is Azuma AI?

Azuma AI is an **innovative agentic AI tutor platform** that makes learning AI and Machine Learning engaging, personalized, and fun! Unlike traditional learning platforms, Azuma AI features **specialized AI tutors** with unique personalities that proactively guide you through your learning journey.

### 🎯 Key Features

#### 🧠 **Multiple Specialized AI Tutors**
- **Prof. Data** 📊 - Your friendly ML fundamentals expert
- **Neural** 🧠 - Deep learning enthusiast with visual teaching style
- **Vision** 👁️ - Computer vision specialist
- **Linguist** 📝 - NLP expert who demystifies transformers

Each tutor has a unique personality, teaching style, and expertise!

#### 🎓 **Adaptive Learning System**
- **Personalized Learning Paths** - AI generates custom curricula based on your goals
- **Real-time Difficulty Adjustment** - Content adapts to your performance
- **Knowledge Graph Tracking** - Visualize your learning progress
- **Spaced Repetition** - Intelligent review recommendations

#### 🎮 **Gamification & Engagement**
- **Points & Leveling System** - Earn XP as you learn
- **Badges & Achievements** - 30+ unique badges to unlock
- **Daily Challenges** - Keep your learning streak alive
- **Leaderboards** - Compete with other learners
- **Progress Visualization** - Beautiful knowledge maps

#### 💬 **Interactive Learning**
- **Real-time Chat** - Conversation with AI tutors via WebSocket
- **Socratic Method** - Tutors ask guiding questions
- **Code Examples** - Executable Python code snippets
- **Visual Explanations** - Diagrams and visualizations
- **Instant Feedback** - Immediate assessment and suggestions

#### 📚 **Comprehensive Curriculum**
- Python for ML (NumPy, Pandas, Matplotlib)
- Math Foundations (Linear Algebra, Calculus, Statistics)
- Machine Learning (Supervised, Unsupervised, Evaluation)
- Deep Learning (Neural Networks, Backprop, Optimization)
- Computer Vision (CNNs, Object Detection, Segmentation)
- NLP (Transformers, BERT, GPT, Attention)
- Reinforcement Learning (Q-Learning, DQN, Policy Gradients)
- MLOps (Deployment, Monitoring, MLflow)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PatrickAttankurugu/sigma2.git
   cd sigma2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run azuma_app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

---

## 🏗️ Architecture

### Tech Stack

- **Frontend**: Streamlit (Interactive UI)
- **Backend**: FastAPI (REST API + WebSocket)
- **AI Framework**: PydanticAI (Agentic AI)
- **LLM**: Google Gemini 2.0 Flash
- **Orchestration**: LangGraph (Complex workflows)
- **Visualization**: Plotly, NetworkX

### System Components

```
azuma/
├── agents/              # AI Tutors (PydanticAI)
│   ├── base_tutor.py   # Base tutor class
│   └── teaching_agents.py  # Specialized tutors
├── learning/            # Adaptive Learning System
│   └── adaptive_path.py    # Personalized curriculum
├── gamification/        # Engagement System
│   └── achievement_system.py  # Badges, points, levels
├── knowledge/           # Knowledge Graph
│   └── knowledge_graph.py     # Topic relationships
├── backend/             # FastAPI Server
│   └── api_server.py          # REST + WebSocket APIs
└── models/              # Pydantic Models
    ├── user.py         # User profiles
    └── lesson.py       # Lessons, quizzes, exercises
```

---

## 📖 How to Use

### 1. Create Your Profile

When you first launch Azuma AI, you'll create a personalized profile:
- Choose your experience level (Beginner → Expert)
- Select your learning style (Visual, Hands-on, Theoretical, etc.)
- Set your learning goals
- Pick topics of interest

### 2. Choose Your Tutor

Select from 4 specialized AI tutors based on what you want to learn:
- **ML Fundamentals** → Prof. Data
- **Deep Learning** → Neural
- **Computer Vision** → Vision
- **NLP** → Linguist

### 3. Start Learning!

- **Chat Interface**: Ask questions, get explanations, receive code examples
- **Learning Dashboard**: See recommended topics, daily challenges, progress
- **Knowledge Map**: Visualize your learning journey
- **Achievements**: Track badges, points, and level progress

### 4. Complete Challenges

- Daily challenges for bonus points
- Quiz questions to test understanding
- Code exercises for hands-on practice
- Projects to apply your skills

---

## 🎓 Learning Paths

Azuma AI offers pre-built and AI-generated learning paths:

### Beginner Track
1. Python Basics → NumPy → Pandas
2. Math Foundations → Statistics
3. Introduction to ML → Supervised Learning
4. First ML Project

### Intermediate Track
1. Neural Networks → Deep Learning
2. CNNs → Computer Vision
3. RNNs → NLP Basics
4. Advanced Projects

### Advanced Track
1. Transformers → BERT/GPT
2. Object Detection → Segmentation
3. Reinforcement Learning
4. MLOps & Deployment

---

## 🏆 Gamification

### Levels (1-10)
- Level 1: AI Novice (0 points)
- Level 5: AI Engineer (5,000 points)
- Level 10: AI Guru (30,000 points)

### Badge Categories
- **Milestone Badges**: First lesson, 10 lessons, 50 lessons
- **Streak Badges**: 3, 7, 30, 100 day streaks
- **Mastery Badges**: Topic mastery (ML, DL, CV, NLP)
- **Special Badges**: Early bird, night owl, perfectionist
- **Challenge Badges**: Speed learner, marathon learner

### Points System
- Lesson completed: 50 points
- Quiz passed: 30 points
- Perfect quiz: 100 points
- Code exercise: 40 points
- Project completed: 500 points
- Daily challenge: Variable points

---

## 🔧 Advanced Features

### FastAPI Backend

Run the API server separately for advanced features:

```bash
python -m azuma.backend.api_server
```

The API provides:
- User management endpoints
- Progress tracking
- Learning path generation
- Leaderboards
- Knowledge graph data
- WebSocket chat

### API Endpoints

- `POST /api/users/register` - Register new user
- `GET /api/users/{user_id}/stats` - Get user statistics
- `POST /api/learning-paths/generate` - Generate personalized path
- `GET /api/gamification/leaderboard` - Global leaderboard
- `WS /ws/chat/{user_id}` - WebSocket chat with tutors

---

## 🎨 Customization

### Adding Custom Topics

Edit `azuma/knowledge/knowledge_graph.py` to add new topics to the knowledge graph.

### Creating New Tutors

Extend `BaseTutor` in `azuma/agents/base_tutor.py` and add your tutor to `teaching_agents.py`.

### Custom Badges

Add badges in `azuma/gamification/achievement_system.py`.

---

## 📊 Analytics & Insights

Azuma AI tracks:
- Learning velocity (lessons per week)
- Best time of day for learning
- Topic mastery scores
- Engagement metrics
- Streak consistency
- Performance trends

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PydanticAI** - For the amazing agentic AI framework
- **Google Gemini** - For powerful LLM capabilities
- **Streamlit** - For rapid UI development
- **FastAPI** - For high-performance APIs

---

## 👨‍💻 Author

**Patrick Attankurugu**
- Email: patricka.azuma@gmail.com
- GitHub: [@PatrickAttankurugu](https://github.com/PatrickAttankurugu)

---

## 🔮 Future Enhancements

- [ ] Voice interaction with tutors
- [ ] Mobile app (React Native)
- [ ] Collaborative learning (study groups)
- [ ] Integration with Jupyter notebooks
- [ ] Automatic code review for exercises
- [ ] Video explanations generation
- [ ] Research paper discussions
- [ ] Industry mentorship matching

---

## ⭐ Show Your Support

If you find Azuma AI helpful, please:
- ⭐ Star this repository
- 🐛 Report bugs
- 💡 Suggest features
- 📢 Share with others learning AI/ML

---

<div align="center">

**Made with ❤️ for the AI/ML learning community**

[Get Started](#-quick-start) • [Documentation](#-how-to-use) • [Contribute](#-contributing)

</div>
