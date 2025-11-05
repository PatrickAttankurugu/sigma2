# SIGMA Multi-Agent AI Actions Co-pilot

**Advanced Business Advisory System with Collaborative AI Agents**

A sophisticated prototype demonstrating a multi-agent AI system that collaboratively analyzes business actions, updates business model canvases, and provides strategic guidance through specialized agents working together.

## 🚀 New Features - Multi-Agent System

### 4 Specialized AI Agents
1. **Strategy Agent** 🎯
   - Analyzes Business Model Canvas coherence
   - Detects business stage (validation/growth/scale)
   - Provides strategic context and identifies gaps

2. **Market Research Agent** 📊
   - Validates customer segments
   - Analyzes market opportunities
   - Performs competitive research (with web search)

3. **Product Agent** 🎨
   - Evaluates value propositions
   - Assesses competitive positioning
   - Analyzes product-market fit

4. **Execution Agent** ⚡
   - Synthesizes insights from all agents
   - Proposes specific BMC changes
   - Generates actionable next steps

### Real-Time Streaming
- **WebSocket Integration** - See agents working together in real-time
- **Live Updates** - Watch analysis progress as each agent contributes
- **Collaborative Intelligence** - Agents build on each other's insights

### Dual Operating Modes
1. **Streaming Mode** (with WebSocket server)
   - Real-time agent collaboration
   - Live progress updates
   - Enhanced user experience

2. **Non-Streaming Mode** (standalone)
   - Direct multi-agent execution
   - Works without WebSocket server
   - Full agent insights available

## Overview

This application implements a comprehensive entrepreneur journey:

1. **Business Design Phase** - Define business model across 4 key sections
2. **Multi-Agent Analysis** - 4 specialized agents collaboratively analyze actions
3. **Strategic Recommendations** - Comprehensive BMC updates and next steps

## Key Features

### Core Capabilities
- ✅ **Multi-Agent Collaboration** - 4 specialized agents working together
- ✅ **Real-Time Streaming** - WebSocket-based live updates
- ✅ **Preview of BMC Changes** - Before/after view with confidence scores
- ✅ **Auto-mode Toggle** - Automatically applies high-confidence changes (>80%)
- ✅ **Agent Insights** - See what each agent discovered
- ✅ **LangChain Integration** - Built with LangChain and LangGraph

### Enhanced Intelligence
- **Context-Aware Analysis** - Each agent brings domain expertise
- **Collaborative Reasoning** - Agents build on each other's insights
- **Comprehensive Tools** - BMC analysis, market research, competitive positioning
- **Stage-Based Strategy** - Recommendations tailored to business maturity

## Technical Architecture

### Multi-Agent System
```
┌─────────────────────────────────────────────┐
│         WebSocket Server (Optional)         │
│         Real-time Streaming                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         LangGraph Orchestration             │
│                                             │
│  ┌──────────┐  ┌──────────┐                │
│  │ Strategy │→ │  Market  │                │
│  │  Agent   │  │ Research │                │
│  └──────────┘  └──────────┘                │
│       ↓             ↓                       │
│  ┌──────────┐  ┌──────────┐                │
│  │ Product  │→ │Execution │                │
│  │  Agent   │  │  Agent   │                │
│  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        Streamlit Frontend                   │
│        Real-time Updates                    │
└─────────────────────────────────────────────┘
```

### Key Components
- **Multi-Agent System** (`multi_agent_system.py`) - LangGraph-based agent orchestration
- **Agent Tools** (`agent_tools.py`) - Specialized tools for each agent
- **WebSocket Server** (`websocket_server.py`) - Real-time streaming backend
- **WebSocket Client** (`websocket_client.py`) - Streamlit integration
- **Business Model Canvas** (`bmc_canvas.py`) - Core business model management
- **UI Components** (`ui_components.py`) - Enhanced interface elements

### Technologies
- **LangChain** - AI agent framework
- **LangGraph** - Multi-agent orchestration
- **Google Gemini 2.0 Flash** - Large language model
- **WebSockets** - Real-time communication
- **Streamlit** - Web interface

## Installation & Setup

### 1. Clone Repository
```bash
git clone [repository-url]
cd sigma2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Google API key
GOOGLE_API_KEY=your_actual_google_api_key_here

# Optional: For market research web search
GOOGLE_CSE_ID=your_custom_search_engine_id
```

Get API keys:
- Google API Key: https://makersuite.google.com/app/apikey
- Google Custom Search: https://programmablesearchengine.google.com/

### 4. Run Application

**Option A: With Streaming (Recommended)**
```bash
./start_app.sh
```
This starts both the WebSocket server and Streamlit app.

**Option B: Without Streaming**
```bash
./start_app_no_streaming.sh
```
This starts only the Streamlit app (multi-agent still works, but no live updates).

**Option C: Manual Start**
```bash
# Terminal 1: Start WebSocket server
python websocket_server.py

# Terminal 2: Start Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### Phase 1: Business Design
1. Define **Customer Segments** (1-5 items)
   - Be specific: include demographics, geography, psychographics
2. Specify **Value Propositions**
   - Highlight quantifiable benefits and differentiation
3. Describe **Business Models**
   - Include pricing, revenue streams, channels
4. Outline **Market Opportunities**
   - Define market size, trends, and problems being solved

### Phase 2: Multi-Agent Analysis

#### Enable Streaming (Optional)
- Toggle **"Multi-Agent Streaming"** to see real-time agent collaboration
- Watch as each agent analyzes the action sequentially
- See live updates as insights are generated

#### Analyze Actions
1. Select a sample action or create custom experiment
2. Log detailed outcome and metrics
3. Click **"Analyze Action & Update Business Model"**
4. Watch the multi-agent analysis:
   - 🎯 **Strategy Agent** assesses overall coherence
   - 📊 **Market Research Agent** validates customer/market assumptions
   - 🎨 **Product Agent** evaluates value propositions
   - ⚡ **Execution Agent** synthesizes and recommends changes

#### Review Results
- **Agent Insights** - See what each agent discovered
- **Proposed Changes** - Review BMC updates with confidence scores
- **Next Steps** - Get actionable recommendations with timelines

#### Apply Changes
- **Auto-mode ON**: High-confidence changes applied automatically
- **Auto-mode OFF**: Manual review and approval
- View before/after comparison for each change

## Multi-Agent Workflow Example

**Action**: "Conducted 20 customer interviews with small business owners"
**Outcome**: Successful - Validated pain points around manual processes

### Agent Collaboration:

**🎯 Strategy Agent**
- "Business is in validation stage"
- "Customer segments need more specificity based on interviews"
- "Strong coherence between value props and target customers"

**📊 Market Research Agent**
- "Interview data validates small business segment"
- "Suggests adding specific demographics: 25-45, tech-comfortable"
- "Market opportunity shows strong demand for automation"

**🎨 Product Agent**
- "Value proposition should emphasize time-savings (quantified in interviews)"
- "Competitive positioning: automation + ease-of-use"
- "Product-market fit indicators are positive"

**⚡ Execution Agent**
- *Synthesizes insights*
- *Proposes Changes*:
  - Add: "Tech-comfortable small business owners (25-45, 10-50 employees)"
  - Modify: "Save 70% time on manual processes (validated by interviews)"
- *Next Steps*:
  1. Build MVP focusing on automation features (2-3 weeks)
  2. Test with 5 interview participants (1 week)
  3. Measure time-savings metrics (ongoing)

## Architecture Benefits

### Specialized Expertise
Each agent focuses on its domain, providing deeper analysis than a single AI.

### Collaborative Intelligence
Agents build on each other's insights, creating comprehensive recommendations.

### Modular & Extensible
Easy to add new agents or tools without changing core architecture.

### Real-Time Transparency
Streaming mode lets users see the "thinking process" of the AI system.

## Configuration Options

### Auto-mode
- **Enabled**: Applies changes with >80% confidence automatically
- **Disabled**: Requires manual approval for all changes

### Streaming Mode
- **Enabled**: Real-time updates via WebSocket
- **Disabled**: Batch results after completion
- **Requirement**: WebSocket server must be running

### Agent Tools
Agents have access to:
- BMC coherence analysis
- Business stage detection
- Customer segment validation
- Market trend search (if configured)
- Value proposition assessment
- Competitive positioning analysis
- Action impact evaluation
- Next step generation

## Project Structure

```
sigma2/
├── app.py                           # Main Streamlit application
├── websocket_server.py              # WebSocket streaming server
├── start_app.sh                     # Startup script (with streaming)
├── start_app_no_streaming.sh        # Startup script (without streaming)
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── modules/
│   ├── multi_agent_system.py        # LangGraph multi-agent orchestration
│   ├── agent_tools.py               # Tools for each specialized agent
│   ├── websocket_client.py          # WebSocket client for Streamlit
│   ├── ai_engine.py                 # Original single AI (fallback)
│   ├── bmc_canvas.py                # Business Model Canvas
│   ├── business_design.py           # Business design phase
│   ├── ui_components.py             # Streamlit UI components
│   └── utils.py                     # Logging and utilities
└── logs/                            # Application logs
```

## Troubleshooting

### WebSocket Connection Issues
- Ensure WebSocket server is running: `python websocket_server.py`
- Check port 8765 is not in use
- Fallback: Disable streaming mode (multi-agent still works)

### API Key Issues
- Verify `GOOGLE_API_KEY` in `.env` file
- Test key at: https://makersuite.google.com/app/apikey
- Check API quota limits

### Agent Timeout
- LangGraph may take 30-60 seconds for complex analyses
- Be patient - multiple agents are working together
- Check logs in `logs/` directory for details

## Development

### Adding New Agents
1. Create agent node in `multi_agent_system.py`
2. Add agent tools in `agent_tools.py`
3. Update LangGraph workflow
4. Add to UI display

### Customizing Agent Behavior
- Modify agent prompts in `multi_agent_system.py`
- Adjust tool implementations in `agent_tools.py`
- Fine-tune LLM parameters (temperature, tokens)

## Performance

- **Multi-Agent Analysis**: 30-60 seconds
- **Streaming Updates**: Real-time (<100ms latency)
- **Agent Coordination**: Sequential with context passing
- **Fallback Mode**: Available if streaming fails

## Future Enhancements

- [ ] Parallel agent execution (currently sequential)
- [ ] Additional agents (Finance, Legal, Operations)
- [ ] Agent voting/consensus mechanism
- [ ] Historical action tracking
- [ ] BMC versioning and rollback
- [ ] Export analysis reports
- [ ] Integration with external data sources

## Credits

Built with:
- [LangChain](https://www.langchain.com/) - AI agent framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Multi-agent orchestration
- [Streamlit](https://streamlit.io/) - Web framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM

## License

[Your License Here]

---

**Seedstars Senior AI Engineer Assignment - Option 2**
*Enhanced with Multi-Agent Architecture & Real-Time Streaming*
