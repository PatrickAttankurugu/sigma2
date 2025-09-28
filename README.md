# SIGMA Agentic AI Actions Co-pilot

**Seedstars Senior AI Engineer Assignment - Option 2**

A functional prototype demonstrating an agentic AI system that automatically updates business model canvases based on completed actions and suggests strategic next steps.

## Overview

This application implements a two-phase entrepreneur journey:

1. **Business Design Phase** - Users define their business model across 4 key sections
2. **Actions Phase** - Users log completed experiments/actions, and AI automatically analyzes outcomes and updates the business model

## Key Features

### Core Requirements (Assignment)
- ✅ **Preview of BMC Changes** - Shows before/after view with Apply button
- ✅ **Auto-mode Toggle** - Automatically applies high-confidence changes (>80%)
- ✅ **Idempotent Behavior** - Prevents duplicate updates for same results
- ✅ **Quality Validation** - AI response quality scoring with retry logic

### Enhanced Features
- **Business Intelligence** - Stage-aware recommendations (validation/growth/scale)
- **Strategic Next Steps** - Detailed implementation guidance with timelines, resources, and success metrics
- **Session Management** - Comprehensive logging and metrics tracking
- **Sample Actions** - Context-aware sample experiments based on business stage

## Technical Architecture

### Components
- **Business Model Canvas** (`bmc_canvas.py`) - Core business model management
- **AI Engine** (`ai_engine.py`) - Quality-validated LLM integration with Google Gemini
- **Business Design Manager** (`business_design.py`) - Initial setup flow
- **UI Components** (`ui_components.py`) - Reusable interface elements
- **Session Metrics** (`utils.py`) - Logging and analytics

### AI Quality System
- **Specificity Scoring** - Reduces generic business language
- **Evidence Alignment** - Ensures recommendations match action outcomes
- **Actionability Assessment** - Validates practical implementation feasibility
- **Retry Logic** - Automatically improves low-quality responses

## Installation & Setup

1. **Clone Repository**
   ```bash
   git clone [repository-url]
   cd sigma-actions-copilot
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

## Usage Flow

### Phase 1: Business Design
1. Define Customer Segments (1-5 items)
2. Specify Value Propositions
3. Describe Business Models
4. Outline Market Opportunities

### Phase 2: Actions & Analysis
1. Toggle Auto-mode (optional)
2. Select sample action or create custom experiment
3. Log action outcome (Successful/Failed/Inconclusive)
4. AI analyzes and proposes BMC updates
5. Review and apply changes
6. Follow strategic next steps

## Sample Workflow

**Example: Customer Interview Action**
- **Input**: "Customer discovery interviews with 20 potential small business owners"
- **Outcome**: Successful
- **AI Analysis**: Updates Customer Segments with validated insights
- **Next Steps**: Generates specific MVP testing recommendations with timelines

## Auto-mode Behavior

- **ON**: Changes with >80% confidence applied immediately
- **OFF**: Manual approval required for all changes
- **Safety**: Only applies changes with confidence >60%

## Quality Validation

The AI system includes multi-layer quality checks:
- **Response Scoring**: 0-100% quality assessment
- **Issue Detection**: Identifies generic language, missing evidence
- **Automatic Retry**: Improves responses below quality threshold
- **Fallback Handling**: Graceful degradation for system failures

## File Structure

```
sigma-actions-copilot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── modules/
│   ├── __init__.py
│   ├── ai_engine.py      # AI analysis and quality validation
│   ├── bmc_canvas.py     # Business Model Canvas logic
│   ├── business_design.py # Initial business setup
│   ├── ui_components.py  # Reusable UI elements
│   └── utils.py          # Logging and session management
└── logs/                 # Application logs (auto-generated)
```

## Technical Decisions

### LLM Integration
- **Model**: Google Gemini 2.0 Flash for balance of quality and speed
- **Framework**: LangChain for structured prompt management
- **Quality Control**: Custom validation layer with retry logic

### State Management
- **Streamlit Session State** for persistent user data
- **Structured Logging** for debugging and analytics
- **Change History** tracking for audit trail

### Business Logic
- **Stage Detection** based on content analysis (validation/growth/scale)
- **Risk Assessment** automated from business model content
- **Sample Actions** dynamically generated based on user's business context

## Assignment Compliance

This implementation fully addresses Option 2 requirements:

✅ **Functional Prototype** - Complete working application
✅ **BMC Update Preview** - Before/after change visualization
✅ **Apply Button** - Manual change approval
✅ **Auto-mode Toggle** - Automatic vs manual change application
✅ **Idempotent Behavior** - Prevents duplicate updates
✅ **Version History** - Session logging and change tracking

## Future Enhancements

- Multi-agent architecture for specialized domain expertise
- Integration with real SIGMA platform APIs
- Advanced analytics dashboard
- Team collaboration features
- Export capabilities for business plans

---

**Contact**: Patrick Attankurugu  
**Email**: patricka.azuma@gmail.com  
**GitHub**: https://github.com/PatrickAttankurugu