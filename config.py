"""
Comprehensive configuration for the Agentic AI Actions Co-pilot system.

This module centralizes all configuration settings for document processing,
vector search, AI models, file handling, and regional context.
"""

import os
from pathlib import Path
from typing import Dict, List, Set
from enum import Enum

# ==================== BASE CONFIGURATION ====================

# Application metadata
APP_NAME = "Agentic AI Actions Co-pilot"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "AI-powered business model canvas updates from document analysis"

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_store"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, UPLOADS_DIR, VECTOR_DB_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================== FILE UPLOAD CONFIGURATION ====================

class FileType(str, Enum):
    """Supported file types for document upload."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    MD = "md"

# File upload limits
MAX_FILE_SIZE_MB = 10
MAX_FILES_PER_UPLOAD = 5
MAX_TOTAL_FILES_PER_USER = 25

# Allowed file types and extensions
ALLOWED_FILE_TYPES = {
    FileType.PDF: [".pdf"],
    FileType.DOCX: [".docx"],
    FileType.DOC: [".doc"],
    FileType.TXT: [".txt"],
    FileType.XLSX: [".xlsx"],
    FileType.XLS: [".xls"],
    FileType.CSV: [".csv"],
    FileType.MD: [".md", ".markdown"]
}

# Flatten for easy validation
ALLOWED_EXTENSIONS = set()
for extensions in ALLOWED_FILE_TYPES.values():
    ALLOWED_EXTENSIONS.update(extensions)

# File validation settings
SCAN_FOR_MALWARE = False  # Set to True in production
QUARANTINE_SUSPICIOUS_FILES = True
FILE_HASH_CHECK = True

# Document types we can extract business model information from
BUSINESS_DOCUMENT_KEYWORDS = {
    "business_model": ["business model", "bmc", "canvas", "value proposition"],
    "action_outcome": ["experiment", "test", "pilot", "mvp", "outcome", "result"],
    "market_research": ["survey", "interview", "focus group", "customer feedback"],
    "financial": ["revenue", "cost", "pricing", "financial model", "budget"],
    "strategy": ["strategy", "roadmap", "plan", "objectives", "goals"]
}


# ==================== DOCUMENT PROCESSING CONFIGURATION ====================

# Text extraction settings
MIN_TEXT_LENGTH = 50  # Minimum characters for meaningful text
MAX_TEXT_LENGTH = 50000  # Maximum characters to process
CHUNK_SIZE = 1000  # Text chunk size for processing
CHUNK_OVERLAP = 200  # Overlap between chunks

# Language detection and processing
PRIMARY_LANGUAGE = "en"  # English
SUPPORTED_LANGUAGES = ["en", "fr"]  # English and French for West Africa
MIN_CONFIDENCE_LANGUAGE_DETECTION = 0.7

# Business model canvas extraction
BMC_SECTIONS = [
    "customer_segments", "value_propositions", "channels", "customer_relationships",
    "revenue_streams", "key_resources", "key_activities", "key_partnerships", "cost_structure"
]

BMC_SECTION_KEYWORDS = {
    "customer_segments": ["customer", "segment", "target", "user", "client", "buyer"],
    "value_propositions": ["value", "proposition", "benefit", "solution", "offering"],
    "channels": ["channel", "distribution", "sales", "reach", "delivery"],
    "customer_relationships": ["relationship", "support", "service", "engagement"],
    "revenue_streams": ["revenue", "income", "monetization", "pricing", "payment"],
    "key_resources": ["resource", "asset", "infrastructure", "capability"],
    "key_activities": ["activity", "process", "operation", "function"],
    "key_partnerships": ["partner", "alliance", "supplier", "collaboration"],
    "cost_structure": ["cost", "expense", "budget", "spending", "investment"]
}

# Action outcome detection patterns
ACTION_OUTCOME_PATTERNS = {
    "successful": ["success", "achieved", "completed", "positive", "effective", "worked"],
    "failed": ["failed", "unsuccessful", "negative", "ineffective", "didn't work", "poor"],
    "inconclusive": ["unclear", "mixed", "inconclusive", "uncertain", "partial", "ongoing"]
}


# ==================== GOOGLE API CONFIGURATION ====================

# Primary API settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_REGION = os.getenv("GOOGLE_REGION", "us-central1")

# Model configurations
GEMINI_CONFIG = {
    "default_model": "gemini-1.5-flash",
    "fallback_model": "gemini-1.5-pro",
    "temperature": 0.2,
    "max_output_tokens": 2000,
    "timeout": 60,
    "max_retries": 3
}

EMBEDDING_CONFIG = {
    "model": "textembedding-gecko@003",
    "dimensions": 768,
    "batch_size": 50,
    "max_retries": 3,
    "timeout": 30
}

# Rate limiting
API_RATE_LIMIT = {
    "requests_per_minute": 100,
    "requests_per_hour": 1000,
    "backoff_factor": 2,
    "max_backoff": 300
}


# ==================== VECTOR STORE CONFIGURATION ====================

# FAISS settings
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",  # Inner Product (cosine similarity)
    "dimension": 768,  # Google embedding dimension
    "metric": "cosine",
    "nprobe": 10,  # Number of probes for search
    "ef_search": 50,  # Search parameter for HNSW
    "ef_construction": 200  # Construction parameter for HNSW
}

# Vector search parameters
VECTOR_SEARCH_CONFIG = {
    "default_top_k": 5,
    "max_top_k": 20,
    "similarity_threshold": 0.7,
    "rerank_results": True,
    "diversity_lambda": 0.5,  # Balance between relevance and diversity
    "enable_query_expansion": True
}

# Embedding cache settings
EMBEDDING_CACHE_CONFIG = {
    "enabled": True,
    "cache_size": 10000,  # Maximum cached embeddings
    "ttl_hours": 24 * 7,  # Cache for 1 week
    "persist_to_disk": True,
    "cache_file": CACHE_DIR / "embeddings_cache.pkl"
}

# Pattern database settings
PATTERN_DB_CONFIG = {
    "min_pattern_length": 100,
    "max_pattern_length": 5000,
    "quality_threshold": 0.6,
    "freshness_decay_days": 365,  # Patterns get less relevant over time
    "regional_boost": 1.2,  # Boost for African patterns
    "cultural_boost": 1.1   # Boost for cultural context
}


# ==================== BUSINESS INTELLIGENCE CONFIGURATION ====================

# African market context
AFRICAN_CONTEXT = {
    "primary_regions": [
        "West Africa", "East Africa", "Southern Africa", "North Africa"
    ],
    "key_countries": [
        "Ghana", "Nigeria", "Kenya", "South Africa", "Rwanda", "Tanzania",
        "Senegal", "Ivory Coast", "Uganda", "Ethiopia"
    ],
    "focus_country": "Ghana",  # Primary focus
    "languages": ["English", "French", "Swahili", "Hausa", "Yoruba", "Twi"],
    "currencies": ["GHS", "NGN", "KES", "ZAR", "RWF", "TZS", "XOF"],
    "business_environments": ["formal", "informal", "hybrid"]
}

# Industry categorization
AFRICAN_INDUSTRIES = {
    "fintech": {
        "keywords": ["mobile money", "payment", "banking", "credit", "savings"],
        "context_weight": 1.5
    },
    "agritech": {
        "keywords": ["agriculture", "farming", "crop", "livestock", "rural"],
        "context_weight": 1.3
    },
    "healthtech": {
        "keywords": ["health", "medical", "telemedicine", "diagnostics"],
        "context_weight": 1.2
    },
    "edtech": {
        "keywords": ["education", "learning", "school", "training"],
        "context_weight": 1.2
    },
    "logistics": {
        "keywords": ["delivery", "transport", "logistics", "supply chain"],
        "context_weight": 1.1
    },
    "retail": {
        "keywords": ["retail", "commerce", "marketplace", "shopping"],
        "context_weight": 1.0
    }
}

# Business stage classification
BUSINESS_STAGES = {
    "idea": {"keywords": ["concept", "idea", "planning"], "confidence_adjustment": 0.8},
    "mvp": {"keywords": ["prototype", "mvp", "pilot"], "confidence_adjustment": 0.9},
    "early": {"keywords": ["launch", "early", "startup"], "confidence_adjustment": 1.0},
    "growth": {"keywords": ["scaling", "expansion", "growth"], "confidence_adjustment": 1.1},
    "mature": {"keywords": ["established", "mature", "stable"], "confidence_adjustment": 1.0}
}


# ==================== AGENT CONFIGURATION ====================

# Agent-specific settings
AGENT_CONFIG = {
    "action_detection": {
        "confidence_threshold": 0.6,
        "max_processing_time": 30,
        "require_validation": True
    },
    "outcome_analysis": {
        "min_context_length": 200,
        "pattern_search_top_k": 7,
        "cultural_context_required": True
    },
    "canvas_update": {
        "max_changes_per_session": 10,
        "confidence_threshold": 0.7,
        "safety_check_required": True
    },
    "next_step": {
        "max_suggestions": 5,
        "prioritization_enabled": True,
        "cultural_adaptation": True
    },
    "vector_search": {
        "query_expansion": True,
        "semantic_search": True,
        "metadata_filtering": True
    }
}

# Auto-mode safety settings
AUTO_MODE_CONFIG = {
    "enabled": True,
    "confidence_threshold": 0.8,
    "max_auto_changes": 5,
    "safety_check_required": True,
    "excluded_sections": ["revenue_streams"],  # Never auto-change revenue
    "require_human_approval": ["key_partnerships", "cost_structure"]
}


# ==================== UI CONFIGURATION ====================

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": MAX_FILE_SIZE_MB
}

# Chat interface settings
CHAT_CONFIG = {
    "max_message_length": 5000,
    "max_chat_history": 100,
    "enable_typing_indicator": True,
    "message_delay_ms": 500,
    "auto_scroll": True
}

# Display settings
UI_CONFIG = {
    "theme": "clean",  # Simplified from SIGMA styling
    "show_debug_info": os.getenv("DEBUG", "false").lower() == "true",
    "animate_transitions": True,
    "show_confidence_scores": True,
    "display_retrieved_patterns": True,
    "max_patterns_shown": 3
}


# ==================== LOGGING AND MONITORING ====================

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / "app.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "console_output": True
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    "track_processing_time": True,
    "track_api_calls": True,
    "track_vector_search": True,
    "alert_slow_queries": True,
    "slow_query_threshold_ms": 5000
}

# Error handling
ERROR_HANDLING = {
    "max_retries": 3,
    "retry_delay": 1,
    "exponential_backoff": True,
    "graceful_degradation": True,
    "fallback_responses": True
}


# ==================== SECURITY CONFIGURATION ====================

# File security
SECURITY_CONFIG = {
    "scan_uploads": True,
    "sanitize_filenames": True,
    "restrict_file_types": True,
    "max_file_age_days": 30,
    "auto_cleanup": True,
    "encrypt_storage": False  # Enable in production
}

# API security
API_SECURITY = {
    "require_api_key": True,
    "rate_limiting": True,
    "request_validation": True,
    "sanitize_inputs": True,
    "log_suspicious_activity": True
}


# ==================== DEVELOPMENT AND TESTING ====================

# Development settings
DEV_CONFIG = {
    "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
    "mock_api_calls": os.getenv("MOCK_API", "false").lower() == "true",
    "sample_data_enabled": True,
    "performance_profiling": False,
    "verbose_logging": os.getenv("VERBOSE", "false").lower() == "true"
}

# Testing configuration
TEST_CONFIG = {
    "test_data_dir": BASE_DIR / "tests" / "data",
    "sample_documents": [
        "sample_business_plan.pdf",
        "market_research_report.docx",
        "customer_feedback.txt"
    ],
    "mock_embedding_dimension": 768,
    "test_patterns_count": 10
}


# ==================== VALIDATION FUNCTIONS ====================

def validate_config() -> List[str]:
    """Validate configuration settings and return list of issues."""
    issues = []
    
    # Check required API keys
    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY environment variable not set")
    
    # Check file size limits
    if MAX_FILE_SIZE_MB > 100:
        issues.append("MAX_FILE_SIZE_MB seems too large (>100MB)")
    
    # Check directory permissions
    for directory in [UPLOADS_DIR, VECTOR_DB_DIR, CACHE_DIR]:
        if not directory.exists():
            issues.append(f"Directory does not exist: {directory}")
        elif not os.access(directory, os.W_OK):
            issues.append(f"No write permission for: {directory}")
    
    # Validate vector search settings
    if VECTOR_SEARCH_CONFIG["similarity_threshold"] > 1.0:
        issues.append("similarity_threshold cannot be greater than 1.0")
    
    return issues


def get_file_type_from_extension(filename: str) -> FileType:
    """Get file type enum from filename extension."""
    ext = Path(filename).suffix.lower()
    
    for file_type, extensions in ALLOWED_FILE_TYPES.items():
        if ext in extensions:
            return file_type
    
    raise ValueError(f"Unsupported file extension: {ext}")


def is_business_document(text: str) -> Dict[str, float]:
    """Score how likely a document contains business model information."""
    text_lower = text.lower()
    scores = {}
    
    for doc_type, keywords in BUSINESS_DOCUMENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[doc_type] = score / len(keywords)  # Normalize
    
    return scores


def get_african_context_boost(text: str) -> float:
    """Calculate boost factor for African business context."""
    text_lower = text.lower()
    
    # Check for African countries and regions
    country_mentions = sum(1 for country in AFRICAN_CONTEXT["key_countries"] 
                         if country.lower() in text_lower)
    
    # Check for African business terms
    african_terms = ["mobile money", "informal sector", "agent network", 
                    "village", "community", "tribal", "local language"]
    term_mentions = sum(1 for term in african_terms if term in text_lower)
    
    # Calculate boost (1.0 = no boost, >1.0 = boost)
    base_boost = 1.0
    country_boost = min(country_mentions * 0.1, 0.3)  # Max 30% boost
    term_boost = min(term_mentions * 0.05, 0.2)       # Max 20% boost
    
    return base_boost + country_boost + term_boost


# ==================== EXPORT CONFIGURATION ====================

# Make key configurations easily importable
__all__ = [
    # Core settings
    "APP_NAME", "APP_VERSION", "BASE_DIR", "DATA_DIR",
    
    # File handling
    "ALLOWED_EXTENSIONS", "ALLOWED_FILE_TYPES", "MAX_FILE_SIZE_MB",
    "FileType", "get_file_type_from_extension",
    
    # Document processing
    "BMC_SECTIONS", "BMC_SECTION_KEYWORDS", "ACTION_OUTCOME_PATTERNS",
    "is_business_document",
    
    # API configuration
    "GOOGLE_API_KEY", "GEMINI_CONFIG", "EMBEDDING_CONFIG",
    
    # Vector search
    "FAISS_CONFIG", "VECTOR_SEARCH_CONFIG", "PATTERN_DB_CONFIG",
    
    # African context
    "AFRICAN_CONTEXT", "AFRICAN_INDUSTRIES", "get_african_context_boost",
    
    # Agents and UI
    "AGENT_CONFIG", "AUTO_MODE_CONFIG", "UI_CONFIG", "CHAT_CONFIG",
    
    # Validation
    "validate_config"
]

# Initialize configuration validation on import
if __name__ == "__main__":
    # Run validation when script is executed directly
    issues = validate_config()
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ Configuration validation passed!")