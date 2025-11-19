"""
Configuration constants for SIGMA Agentic AI Co-pilot
Centralized location for all configuration values and thresholds
"""

# ============================================================================
# LLM Configuration
# ============================================================================

# Google Gemini Model Settings
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3  # Lower temperature for more consistent, focused responses
LLM_MAX_TOKENS = 2000  # Maximum tokens in LLM response
LLM_TIMEOUT = 30  # Timeout in seconds for LLM API calls

# Quality Retry Configuration
MAX_QUALITY_RETRIES = 2  # Maximum number of retry attempts (total 3 attempts)
RETRY_DELAY = 1.0  # Delay in seconds between retries

# ============================================================================
# Quality Scoring Thresholds
# ============================================================================

# Overall quality score thresholds
QUALITY_THRESHOLD_EXCELLENT = 0.8  # Score >= 80% is excellent
QUALITY_THRESHOLD_GOOD = 0.6       # Score >= 60% is good
QUALITY_THRESHOLD_ACCEPTABLE = 0.4 # Score >= 40% is acceptable
QUALITY_THRESHOLD_MIN = 0.4        # Minimum acceptable quality score (triggers retry)

# Individual dimension thresholds
SPECIFICITY_MIN_THRESHOLD = 0.3    # Minimum specificity score (triggers retry)
EVIDENCE_MIN_THRESHOLD = 0.3       # Minimum evidence alignment score
ACTIONABILITY_MIN_THRESHOLD = 0.3  # Minimum actionability score
CONSISTENCY_MIN_THRESHOLD = 0.3    # Minimum consistency score

# Quality issue threshold
MAX_QUALITY_ISSUES = 3  # Maximum number of quality issues before retry

# ============================================================================
# Confidence Scoring Thresholds
# ============================================================================

# Confidence levels for BMC changes
CONFIDENCE_HIGH = 0.8      # High confidence threshold (auto-apply eligible)
CONFIDENCE_MEDIUM = 0.7    # Medium confidence threshold
CONFIDENCE_LOW = 0.6       # Low confidence threshold (manual review recommended)

# Auto-apply settings
AUTO_APPLY_MIN_CONFIDENCE = 0.8  # Minimum confidence for auto-apply mode

# ============================================================================
# Quality Scoring Weights and Parameters
# ============================================================================

# Specificity scoring
SPECIFICITY_NUMBER_BONUS = 0.1         # Bonus for including numbers
SPECIFICITY_METRIC_BONUS = 0.1         # Bonus for including metrics
SPECIFICITY_TIMEFRAME_BONUS = 0.05     # Bonus for including timeframes
SPECIFICITY_GENERIC_PENALTY = 0.2      # Penalty for generic words
SPECIFICITY_SHORT_PENALTY = 0.1        # Penalty for short descriptions

# Generic words that reduce specificity (lowercase for matching)
GENERIC_WORDS = [
    "improve", "enhance", "optimize", "better", "good", "great",
    "increase", "decrease", "monitor", "track", "analyze", "evaluate",
    "consider", "explore", "investigate", "review", "assess"
]

# Metric-related keywords for specificity scoring
METRIC_KEYWORDS = [
    "%", "percent", "rate", "ratio", "score", "count", "total",
    "average", "median", "conversion", "retention", "churn"
]

# Timeframe keywords for specificity scoring
TIMEFRAME_KEYWORDS = [
    "day", "week", "month", "quarter", "year", "daily", "weekly",
    "monthly", "quarterly", "annually", "sprint", "deadline"
]

# Evidence alignment
EVIDENCE_POSITIVE_BOOST = 0.2   # Boost for positive outcome alignment
EVIDENCE_NEGATIVE_BOOST = 0.15  # Boost for negative outcome alignment

# Actionability scoring
ACTIONABILITY_VAGUE_PENALTY = 0.3  # Penalty for vague words

# Vague words that reduce actionability (lowercase for matching)
VAGUE_WORDS = [
    "maybe", "possibly", "perhaps", "might", "could", "should",
    "try", "attempt", "consider", "think about", "look into"
]

# ============================================================================
# Business Model Canvas Configuration
# ============================================================================

# BMC Stage Detection
BMC_STAGE_VALIDATION = "validation"
BMC_STAGE_GROWTH = "growth"
BMC_STAGE_SCALE = "scale"

# Stage detection thresholds
STAGE_GROWTH_MIN_INDICATORS = 2   # Minimum indicators needed for growth stage
STAGE_SCALE_MIN_INDICATORS = 2    # Minimum indicators needed for scale stage
STAGE_SCALE_MIN_COMPLETION = 1.0  # Minimum completion ratio for scale stage (100%)

# Growth stage keywords
GROWTH_INDICATORS = [
    "scale", "growth", "expansion", "channels", "acquisition",
    "retention", "revenue", "profit", "market share"
]

# Scale stage keywords
SCALE_INDICATORS = [
    "international", "global", "automation", "enterprise",
    "partnership", "franchise", "licensing", "exit"
]

# Risk levels
RISK_LEVEL_LOW = "Low"
RISK_LEVEL_MEDIUM = "Medium"
RISK_LEVEL_HIGH = "High"

# ============================================================================
# UI Configuration
# ============================================================================

# Progress bar colors
CONFIDENCE_COLOR_HIGH = "#28a745"    # Green
CONFIDENCE_COLOR_MEDIUM = "#ffc107"  # Yellow
CONFIDENCE_COLOR_LOW = "#dc3545"     # Red

# Priority levels
PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

# Difficulty levels
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"

# ============================================================================
# Validation Configuration
# ============================================================================

# Input validation limits
MAX_ACTION_TITLE_LENGTH = 200        # Maximum characters for action title
MAX_ACTION_DESCRIPTION_LENGTH = 2000 # Maximum characters for action description
MAX_OUTCOME_LENGTH = 5000            # Maximum characters for outcome details
MAX_BUSINESS_NAME_LENGTH = 100       # Maximum characters for business name
MAX_SECTION_ITEM_LENGTH = 500        # Maximum characters per BMC section item

# Minimum required lengths
MIN_ACTION_TITLE_LENGTH = 5          # Minimum characters for action title
MIN_ACTION_DESCRIPTION_LENGTH = 10   # Minimum characters for action description
MIN_OUTCOME_LENGTH = 20              # Minimum characters for outcome details

# ============================================================================
# Logging Configuration
# ============================================================================

# Log file settings
LOG_DIRECTORY = "logs"
LOG_FILE_PREFIX = "sigma_copilot"
LOG_DATE_FORMAT = "%Y%m%d"
LOG_MAX_SIZE_MB = 10  # Maximum size of log file before rotation

# Log levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

# ============================================================================
# Session and Metrics Configuration
# ============================================================================

# Engagement scoring
ENGAGEMENT_THRESHOLD_HIGH = 5     # Number of actions for high engagement
ENGAGEMENT_THRESHOLD_MEDIUM = 2   # Number of actions for medium engagement

# Metrics tracking
METRICS_TRACK_ACTIONS = True
METRICS_TRACK_CHANGES = True
METRICS_TRACK_QUALITY = True
METRICS_TRACK_TIMING = True

# ============================================================================
# Feature Flags
# ============================================================================

# Enable/disable features
FEATURE_AUTO_APPLY_MODE = True       # Enable auto-apply mode
FEATURE_QUALITY_VALIDATION = True    # Enable quality validation
FEATURE_NEXT_STEPS_GENERATION = True # Enable next steps generation
FEATURE_SAMPLE_ACTIONS = True        # Enable sample action suggestions
FEATURE_SESSION_METRICS = True       # Enable session metrics tracking

# ============================================================================
# Error Messages
# ============================================================================

ERROR_INVALID_API_KEY = "Invalid or missing API key. Please set GOOGLE_API_KEY in your .env file."
ERROR_JSON_PARSE_FAILED = "Unable to parse AI response. Please try again."
ERROR_LLM_TIMEOUT = "AI analysis timed out. Please try again with a shorter action description."
ERROR_NETWORK_ERROR = "Network error occurred. Please check your connection and try again."
ERROR_QUALITY_TOO_LOW = "AI response quality is below threshold. Manual review recommended."

# Success messages
SUCCESS_CHANGE_APPLIED = "Change applied successfully to Business Model Canvas."
SUCCESS_ACTION_ANALYZED = "Action analyzed successfully."
SUCCESS_BMC_UPDATED = "Business Model Canvas updated successfully."

# ============================================================================
# Help Text and Tooltips
# ============================================================================

HELP_AUTO_APPLY_MODE = """
Auto-apply mode automatically applies high-confidence changes (≥80%) to your Business Model Canvas.
Lower confidence changes will still require manual review.
"""

HELP_QUALITY_SCORE = """
Quality score measures the specificity, evidence alignment, actionability, and consistency of AI analysis.
Higher scores indicate more reliable and actionable recommendations.
"""

HELP_CONFIDENCE_SCORE = """
Confidence score indicates how certain the AI is about the recommended change.
High confidence (≥80%) suggests strong evidence for the change.
"""
