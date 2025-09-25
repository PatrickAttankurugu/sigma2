"""
SIGMA-Inspired Professional Styling Module

This module provides comprehensive CSS styling and UI components that match
the professional design language of the SIGMA platform. Includes color schemes,
typography, component styles, and responsive design patterns.
"""

import streamlit as st
from typing import Dict, List, Optional


class SigmaColors:
    """SIGMA color palette constants"""
    
    # Primary Colors
    PRIMARY_BLUE = "#1E40AF"
    LIGHT_BLUE = "#3B82F6"
    DARK_BLUE = "#1E3A8A"
    
    # Background Colors  
    BG_LIGHT = "#F8FAFC"
    BG_WHITE = "#FFFFFF"
    BG_CARD = "#FFFFFF"
    
    # Text Colors
    TEXT_PRIMARY = "#1F2937"
    TEXT_SECONDARY = "#6B7280" 
    TEXT_MUTED = "#9CA3AF"
    
    # Border and Divider Colors
    BORDER_LIGHT = "#E5E7EB"
    BORDER_MEDIUM = "#D1D5DB"
    BORDER_DARK = "#9CA3AF"
    
    # Status Colors
    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    DANGER = "#EF4444"
    INFO = "#3B82F6"
    
    # Confidence Level Colors
    CONFIDENCE_HIGH = "#D1FAE5"
    CONFIDENCE_MEDIUM = "#FEF3C7"
    CONFIDENCE_LOW = "#FEE2E2"
    
    # Gradient Colors
    GRADIENT_START = "#1E3A8A"
    GRADIENT_END = "#3B82F6"


class SigmaTypography:
    """Typography scale and font definitions"""
    
    FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif"
    FONT_FAMILY_MONO = "'SF Mono', Monaco, Inconsolata, 'Roboto Mono', 'Source Code Pro', monospace"
    
    # Font Sizes
    TEXT_XS = "0.75rem"    # 12px
    TEXT_SM = "0.875rem"   # 14px  
    TEXT_BASE = "1rem"     # 16px
    TEXT_LG = "1.125rem"   # 18px
    TEXT_XL = "1.25rem"    # 20px
    TEXT_2XL = "1.5rem"    # 24px
    TEXT_3XL = "1.875rem"  # 30px
    TEXT_4XL = "2.25rem"   # 36px
    
    # Font Weights
    FONT_LIGHT = "300"
    FONT_NORMAL = "400" 
    FONT_MEDIUM = "500"
    FONT_SEMIBOLD = "600"
    FONT_BOLD = "700"
    FONT_EXTRABOLD = "800"


class SigmaSpacing:
    """Consistent spacing scale"""
    
    XS = "0.25rem"   # 4px
    SM = "0.5rem"    # 8px
    BASE = "0.75rem" # 12px
    MD = "1rem"      # 16px
    LG = "1.5rem"    # 24px
    XL = "2rem"      # 32px
    XXL = "3rem"     # 48px
    XXXL = "4rem"    # 64px


def load_sigma_base_css() -> str:
    """Load base SIGMA CSS styles"""
    
    return f"""
    <style>
    /* ===== SIGMA BASE STYLES ===== */
    
    /* CSS Variables for Color Palette */
    :root {{
        --sigma-primary: {SigmaColors.PRIMARY_BLUE};
        --sigma-light-blue: {SigmaColors.LIGHT_BLUE};
        --sigma-dark-blue: {SigmaColors.DARK_BLUE};
        --sigma-bg-light: {SigmaColors.BG_LIGHT};
        --sigma-bg-white: {SigmaColors.BG_WHITE};
        --sigma-bg-card: {SigmaColors.BG_CARD};
        --sigma-text-primary: {SigmaColors.TEXT_PRIMARY};
        --sigma-text-secondary: {SigmaColors.TEXT_SECONDARY};
        --sigma-text-muted: {SigmaColors.TEXT_MUTED};
        --sigma-border-light: {SigmaColors.BORDER_LIGHT};
        --sigma-border-medium: {SigmaColors.BORDER_MEDIUM};
        --sigma-success: {SigmaColors.SUCCESS};
        --sigma-warning: {SigmaColors.WARNING};
        --sigma-danger: {SigmaColors.DANGER};
        --sigma-info: {SigmaColors.INFO};
        
        /* Typography */
        --sigma-font-family: {SigmaTypography.FONT_FAMILY};
        --sigma-font-mono: {SigmaTypography.FONT_FAMILY_MONO};
        
        /* Spacing */
        --sigma-spacing-xs: {SigmaSpacing.XS};
        --sigma-spacing-sm: {SigmaSpacing.SM};
        --sigma-spacing-base: {SigmaSpacing.BASE};
        --sigma-spacing-md: {SigmaSpacing.MD};
        --sigma-spacing-lg: {SigmaSpacing.LG};
        --sigma-spacing-xl: {SigmaSpacing.XL};
        --sigma-spacing-xxl: {SigmaSpacing.XXL};
        
        /* Shadows */
        --sigma-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --sigma-shadow-base: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --sigma-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --sigma-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --sigma-shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        
        /* Border Radius */
        --sigma-radius-sm: 0.25rem;
        --sigma-radius-base: 0.375rem;
        --sigma-radius-md: 0.5rem;
        --sigma-radius-lg: 0.75rem;
        --sigma-radius-xl: 1rem;
        --sigma-radius-full: 9999px;
        
        /* Transitions */
        --sigma-transition-fast: 150ms ease-in-out;
        --sigma-transition-base: 250ms ease-in-out;
        --sigma-transition-slow: 400ms ease-in-out;
    }}
    
    /* Global Streamlit Overrides */
    .stApp {{
        background-color: var(--sigma-bg-light);
        font-family: var(--sigma-font-family);
    }}
    
    .main .block-container {{
        max-width: 1200px;
        padding-top: var(--sigma-spacing-lg);
        padding-left: var(--sigma-spacing-lg);
        padding-right: var(--sigma-spacing-lg);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    </style>
    """


def load_sigma_component_css() -> str:
    """Load SIGMA component-specific styles"""
    
    return f"""
    <style>
    /* ===== SIGMA COMPONENT STYLES ===== */
    
    /* Header Styles */
    .sigma-header {{
        background: linear-gradient(135deg, var(--sigma-dark-blue) 0%, var(--sigma-light-blue) 100%);
        color: white;
        padding: var(--sigma-spacing-xxl) var(--sigma-spacing-lg);
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 var(--sigma-radius-xl) var(--sigma-radius-xl);
        box-shadow: var(--sigma-shadow-xl);
        position: relative;
        overflow: hidden;
    }}
    
    .sigma-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }}
    
    .sigma-header h1 {{
        font-size: {SigmaTypography.TEXT_4XL};
        font-weight: {SigmaTypography.FONT_BOLD};
        margin: 0 0 var(--sigma-spacing-md) 0;
        text-align: center;
        position: relative;
        z-index: 1;
    }}
    
    .sigma-header .subtitle {{
        font-size: {SigmaTypography.TEXT_LG};
        opacity: 0.9;
        text-align: center;
        margin-bottom: var(--sigma-spacing-lg);
        font-weight: {SigmaTypography.FONT_NORMAL};
        position: relative;
        z-index: 1;
    }}
    
    .sigma-header .features {{
        display: flex;
        justify-content: center;
        gap: var(--sigma-spacing-lg);
        flex-wrap: wrap;
        margin-top: var(--sigma-spacing-lg);
        position: relative;
        z-index: 1;
    }}
    
    .feature-badge {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: var(--sigma-spacing-base) var(--sigma-spacing-lg);
        border-radius: var(--sigma-radius-full);
        font-size: {SigmaTypography.TEXT_SM};
        font-weight: {SigmaTypography.FONT_MEDIUM};
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: var(--sigma-transition-base);
    }}
    
    .feature-badge:hover {{
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
    }}
    
    /* Card Styles */
    .sigma-card {{
        background: var(--sigma-bg-card);
        border-radius: var(--sigma-radius-xl);
        padding: var(--sigma-spacing-xl);
        margin-bottom: var(--sigma-spacing-lg);
        box-shadow: var(--sigma-shadow-base);
        border: 1px solid var(--sigma-border-light);
        transition: var(--sigma-transition-base);
        position: relative;
    }}
    
    .sigma-card:hover {{
        box-shadow: var(--sigma-shadow-lg);
        transform: translateY(-2px);
        border-color: var(--sigma-light-blue);
    }}
    
    .sigma-card h3 {{
        color: var(--sigma-dark-blue);
        margin: 0 0 var(--sigma-spacing-lg) 0;
        font-weight: {SigmaTypography.FONT_SEMIBOLD};
        font-size: {SigmaTypography.TEXT_XL};
        display: flex;
        align-items: center;
        gap: var(--sigma-spacing-base);
    }}
    
    .sigma-card h4 {{
        color: var(--sigma-text-primary);
        margin: 0 0 var(--sigma-spacing-base) 0;
        font-weight: {SigmaTypography.FONT_MEDIUM};
        font-size: {SigmaTypography.TEXT_LG};
    }}
    
    /* Auto-mode Toggle Container */
    .auto-mode-container {{
        background: var(--sigma-bg-card);
        border: 2px solid var(--sigma-light-blue);
        border-radius: var(--sigma-radius-xl);
        padding: var(--sigma-spacing-xl);
        margin: var(--sigma-spacing-lg) 0;
        text-align: center;
        transition: var(--sigma-transition-base);
        position: relative;
        overflow: hidden;
    }}
    
    .auto-mode-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.5s ease;
    }}
    
    .auto-mode-active {{
        background: linear-gradient(135deg, #DBEAFE, #BFDBFE);
        border-color: var(--sigma-success);
        box-shadow: var(--sigma-shadow-md);
    }}
    
    .auto-mode-active::before {{
        left: 100%;
    }}
    
    </style>
    """


def load_sigma_agent_pipeline_css() -> str:
    """Load agent pipeline visualization styles"""
    
    return f"""
    <style>
    /* ===== AGENT PIPELINE STYLES ===== */
    
    .agent-pipeline {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--sigma-bg-card);
        padding: var(--sigma-spacing-xl);
        border-radius: var(--sigma-radius-xl);
        box-shadow: var(--sigma-shadow-base);
        margin: var(--sigma-spacing-lg) 0;
        border: 1px solid var(--sigma-border-light);
        position: relative;
    }}
    
    .agent-step {{
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
        z-index: 2;
    }}
    
    .agent-step::after {{
        content: '';
        position: absolute;
        top: 25px;
        left: 60%;
        width: 80%;
        height: 2px;
        background: var(--sigma-border-medium);
        z-index: 1;
        transition: var(--sigma-transition-base);
    }}
    
    .agent-step:last-child::after {{
        display: none;
    }}
    
    .agent-step.completed::after {{
        background: var(--sigma-success);
    }}
    
    .agent-icon {{
        width: 50px;
        height: 50px;
        border-radius: var(--sigma-radius-full);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: {SigmaTypography.FONT_BOLD};
        color: white;
        margin-bottom: var(--sigma-spacing-base);
        z-index: 2;
        position: relative;
        font-size: {SigmaTypography.TEXT_LG};
        box-shadow: var(--sigma-shadow-md);
        transition: var(--sigma-transition-base);
    }}
    
    .agent-completed {{ 
        background: linear-gradient(135deg, var(--sigma-success), #059669);
    }}
    
    .agent-running {{ 
        background: linear-gradient(135deg, var(--sigma-light-blue), var(--sigma-primary));
        animation: pulse 2s infinite;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }}
    
    .agent-pending {{ 
        background: linear-gradient(135deg, var(--sigma-text-secondary), #4B5563);
    }}
    
    .agent-failed {{ 
        background: linear-gradient(135deg, var(--sigma-danger), #DC2626);
        animation: shake 0.5s;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ 
            opacity: 1; 
            transform: scale(1);
        }}
        50% {{ 
            opacity: 0.8; 
            transform: scale(1.05);
        }}
    }}
    
    @keyframes shake {{
        0%, 100% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-5px); }}
        75% {{ transform: translateX(5px); }}
    }}
    
    .agent-label {{
        font-size: {SigmaTypography.TEXT_SM};
        font-weight: {SigmaTypography.FONT_SEMIBOLD};
        color: var(--sigma-text-primary);
        text-align: center;
        line-height: 1.4;
        min-height: 2.5em;
        display: flex;
        align-items: center;
    }}
    
    </style>
    """


def load_sigma_chat_css() -> str:
    """Load chat interface styles"""
    
    return f"""
    <style>
    /* ===== CHAT INTERFACE STYLES ===== */
    
    .chat-container {{
        max-height: 600px;
        overflow-y: auto;
        padding: var(--sigma-spacing-lg);
        background: var(--sigma-bg-card);
        border-radius: var(--sigma-radius-xl);
        border: 1px solid var(--sigma-border-light);
        position: relative;
    }}
    
    .chat-container::-webkit-scrollbar {{
        width: 6px;
    }}
    
    .chat-container::-webkit-scrollbar-track {{
        background: var(--sigma-bg-light);
        border-radius: var(--sigma-radius-base);
    }}
    
    .chat-container::-webkit-scrollbar-thumb {{
        background: var(--sigma-border-medium);
        border-radius: var(--sigma-radius-base);
    }}
    
    .chat-container::-webkit-scrollbar-thumb:hover {{
        background: var(--sigma-text-secondary);
    }}
    
    .chat-message {{
        margin-bottom: var(--sigma-spacing-lg);
        padding: var(--sigma-spacing-lg);
        border-radius: var(--sigma-radius-lg);
        max-width: 85%;
        position: relative;
        word-wrap: break-word;
        font-size: {SigmaTypography.TEXT_BASE};
        line-height: 1.5;
        box-shadow: var(--sigma-shadow-sm);
        transition: var(--sigma-transition-base);
    }}
    
    .chat-message:hover {{
        box-shadow: var(--sigma-shadow-base);
    }}
    
    .chat-message.user {{
        background: linear-gradient(135deg, var(--sigma-light-blue), var(--sigma-primary));
        color: white;
        margin-left: auto;
        border-bottom-right-radius: var(--sigma-radius-sm);
    }}
    
    .chat-message.user::after {{
        content: '';
        position: absolute;
        bottom: 0;
        right: -8px;
        width: 0;
        height: 0;
        border-left: 8px solid var(--sigma-primary);
        border-bottom: 8px solid transparent;
    }}
    
    .chat-message.assistant {{
        background: var(--sigma-bg-light);
        border: 1px solid var(--sigma-border-light);
        border-bottom-left-radius: var(--sigma-radius-sm);
        color: var(--sigma-text-primary);
    }}
    
    .chat-message.assistant::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: -8px;
        width: 0;
        height: 0;
        border-right: 8px solid var(--sigma-bg-light);
        border-bottom: 8px solid transparent;
    }}
    
    .chat-message.system {{
        background: linear-gradient(135deg, var(--sigma-success), #059669);
        color: white;
        text-align: center;
        margin: 0 auto;
        max-width: 70%;
        font-size: {SigmaTypography.TEXT_SM};
        font-weight: {SigmaTypography.FONT_MEDIUM};
        border-radius: var(--sigma-radius-full);
    }}
    
    .chat-message.agent {{
        background: linear-gradient(135deg, #F3F4F6, #E5E7EB);
        color: var(--sigma-text-primary);
        border: 1px solid var(--sigma-border-light);
        font-style: italic;
        max-width: 70%;
        margin: 0 auto;
    }}
    
    .message-time {{
        font-size: {SigmaTypography.TEXT_XS};
        opacity: 0.7;
        margin-top: var(--sigma-spacing-sm);
        text-align: right;
    }}
    
    .typing-indicator {{
        display: flex;
        align-items: center;
        padding: var(--sigma-spacing-lg);
        color: var(--sigma-text-secondary);
        font-style: italic;
        background: var(--sigma-bg-light);
        border-radius: var(--sigma-radius-lg);
        margin: var(--sigma-spacing-base) 0;
        max-width: 70%;
    }}
    
    .typing-dots {{
        display: inline-flex;
        margin-left: var(--sigma-spacing-base);
        gap: var(--sigma-spacing-xs);
    }}
    
    .typing-dots span {{
        height: 8px;
        width: 8px;
        background: var(--sigma-text-secondary);
        border-radius: var(--sigma-radius-full);
        animation: typing 1.4s infinite ease-in-out;
        opacity: 0.4;
    }}
    
    .typing-dots span:nth-child(1) {{ animation-delay: 0s; }}
    .typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
    
    @keyframes typing {{
        0%, 60%, 100% {{ 
            transform: translateY(0);
            opacity: 0.4;
        }}
        30% {{ 
            transform: translateY(-15px);
            opacity: 1;
        }}
    }}
    
    </style>
    """


def load_sigma_confidence_css() -> str:
    """Load confidence indicator styles"""
    
    return f"""
    <style>
    /* ===== CONFIDENCE INDICATOR STYLES ===== */
    
    .confidence-high {{ 
        background: linear-gradient(135deg, {SigmaColors.CONFIDENCE_HIGH}, #A7F3D0);
        border-left: 4px solid var(--sigma-success);
        color: #047857;
    }}
    
    .confidence-medium {{ 
        background: linear-gradient(135deg, {SigmaColors.CONFIDENCE_MEDIUM}, #FDE68A);
        border-left: 4px solid var(--sigma-warning);
        color: #92400E;
    }}
    
    .confidence-low {{ 
        background: linear-gradient(135deg, {SigmaColors.CONFIDENCE_LOW}, #FECACA);
        border-left: 4px solid var(--sigma-danger);
        color: #991B1B;
    }}
    
    .confidence-high, .confidence-medium, .confidence-low {{
        padding: var(--sigma-spacing-lg);
        border-radius: var(--sigma-radius-md);
        margin: var(--sigma-spacing-base) 0;
        font-weight: {SigmaTypography.FONT_MEDIUM};
        box-shadow: var(--sigma-shadow-sm);
        transition: var(--sigma-transition-base);
        position: relative;
        overflow: hidden;
    }}
    
    .confidence-high:hover, .confidence-medium:hover, .confidence-low:hover {{
        box-shadow: var(--sigma-shadow-md);
        transform: translateX(4px);
    }}
    
    .confidence-badge {{
        display: inline-flex;
        align-items: center;
        gap: var(--sigma-spacing-xs);
        padding: var(--sigma-spacing-xs) var(--sigma-spacing-base);
        border-radius: var(--sigma-radius-full);
        font-size: {SigmaTypography.TEXT_XS};
        font-weight: {SigmaTypography.FONT_SEMIBOLD};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .confidence-badge.high {{
        background: var(--sigma-success);
        color: white;
    }}
    
    .confidence-badge.medium {{
        background: var(--sigma-warning);
        color: white;
    }}
    
    .confidence-badge.low {{
        background: var(--sigma-danger);
        color: white;
    }}
    
    </style>
    """


def load_sigma_bmc_css() -> str:
    """Load Business Model Canvas styles"""
    
    return f"""
    <style>
    /* ===== BUSINESS MODEL CANVAS STYLES ===== */
    
    .bmc-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: var(--sigma-spacing-lg);
        margin: var(--sigma-spacing-lg) 0;
    }}
    
    .bmc-section {{
        background: var(--sigma-bg-card);
        border: 1px solid var(--sigma-border-light);
        border-radius: var(--sigma-radius-lg);
        padding: var(--sigma-spacing-lg);
        min-height: 200px;
        position: relative;
        transition: var(--sigma-transition-base);
    }}
    
    .bmc-section:hover {{
        border-color: var(--sigma-light-blue);
        box-shadow: var(--sigma-shadow-md);
        transform: translateY(-2px);
    }}
    
    .bmc-section h4 {{
        color: var(--sigma-dark-blue);
        font-weight: {SigmaTypography.FONT_SEMIBOLD};
        margin: 0 0 var(--sigma-spacing-lg) 0;
        font-size: {SigmaTypography.TEXT_BASE};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: var(--sigma-spacing-base);
        padding-bottom: var(--sigma-spacing-base);
        border-bottom: 2px solid var(--sigma-border-light);
    }}
    
    .bmc-item {{
        background: var(--sigma-bg-light);
        padding: var(--sigma-spacing-base);
        border-radius: var(--sigma-radius-md);
        margin-bottom: var(--sigma-spacing-base);
        font-size: {SigmaTypography.TEXT_SM};
        border-left: 3px solid var(--sigma-light-blue);
        transition: var(--sigma-transition-fast);
        position: relative;
    }}
    
    .bmc-item:hover {{
        background: #E0F2FE;
        border-left-color: var(--sigma-primary);
        transform: translateX(2px);
    }}
    
    .bmc-item:last-child {{
        margin-bottom: 0;
    }}
    
    .bmc-empty-state {{
        color: var(--sigma-text-muted);
        font-style: italic;
        text-align: center;
        padding: var(--sigma-spacing-xl);
        border: 2px dashed var(--sigma-border-light);
        border-radius: var(--sigma-radius-md);
        background: linear-gradient(45deg, transparent 40%, rgba(59, 130, 246, 0.02) 50%, transparent 60%);
    }}
    
    </style>
    """


def load_sigma_metric_css() -> str:
    """Load metrics and dashboard styles"""
    
    return f"""
    <style>
    /* ===== METRICS AND DASHBOARD STYLES ===== */
    
    .metric-card {{
        background: var(--sigma-bg-card);
        border: 1px solid var(--sigma-border-light);
        border-radius: var(--sigma-radius-lg);
        padding: var(--sigma-spacing-lg);
        text-align: center;
        transition: var(--sigma-transition-base);
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--sigma-primary), var(--sigma-light-blue));
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: var(--sigma-shadow-lg);
        border-color: var(--sigma-light-blue);
    }}
    
    .metric-value {{
        font-size: {SigmaTypography.TEXT_3XL};
        font-weight: {SigmaTypography.FONT_BOLD};
        color: var(--sigma-dark-blue);
        margin: 0;
        line-height: 1;
    }}
    
    .metric-label {{
        color: var(--sigma-text-secondary);
        font-size: {SigmaTypography.TEXT_SM};
        font-weight: {SigmaTypography.FONT_MEDIUM};
        margin-top: var(--sigma-spacing-base);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-trend {{
        font-size: {SigmaTypography.TEXT_XS};
        margin-top: var(--sigma-spacing-xs);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--sigma-spacing-xs);
    }}
    
    .trend-up {{
        color: var(--sigma-success);
    }}
    
    .trend-down {{
        color: var(--sigma-danger);
    }}
    
    .trend-neutral {{
        color: var(--sigma-text-muted);
    }}
    
    </style>
    """


def load_sigma_button_css() -> str:
    """Load button and interactive element styles"""
    
    return f"""
    <style>
    /* ===== BUTTON AND INTERACTIVE STYLES ===== */
    
    .stButton > button {{
        background: linear-gradient(135deg, var(--sigma-light-blue), var(--sigma-dark-blue));
        color: white;
        border: none;
        border-radius: var(--sigma-radius-lg);
        padding: var(--sigma-spacing-base) var(--sigma-spacing-xl);
        font-weight: {SigmaTypography.FONT_SEMIBOLD};
        font-size: {SigmaTypography.TEXT_BASE};
        transition: var(--sigma-transition-base);
        box-shadow: var(--sigma-shadow-md);
        position: relative;
        overflow: hidden;
        font-family: var(--sigma-font-family);
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: var(--sigma-shadow-lg);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: var(--sigma-shadow-base);
    }}
    
    /* Secondary Button Style */
    .stButton.secondary > button {{
        background: var(--sigma-bg-card);
        color: var(--sigma-primary);
        border: 2px solid var(--sigma-primary);
    }}
    
    .stButton.secondary > button:hover {{
        background: var(--sigma-primary);
        color: white;
    }}
    
    /* Success Button Style */
    .stButton.success > button {{
        background: linear-gradient(135deg, var(--sigma-success), #059669);
    }}
    
    /* Warning Button Style */
    .stButton.warning > button {{
        background: linear-gradient(135deg, var(--sigma-warning), #D97706);
    }}
    
    /* Danger Button Style */
    .stButton.danger > button {{
        background: linear-gradient(135deg, var(--sigma-danger), #DC2626);
    }}
    
    /* Toggle Styles */
    .stCheckbox > label {{
        font-size: {SigmaTypography.TEXT_BASE};
        font-weight: {SigmaTypography.FONT_MEDIUM};
        color: var(--sigma-text-primary);
    }}
    
    /* Selectbox Styles */
    .stSelectbox > div > div > div {{
        border-radius: var(--sigma-radius-md);
        border-color: var(--sigma-border-medium);
        transition: var(--sigma-transition-fast);
    }}
    
    .stSelectbox > div > div > div:focus-within {{
        border-color: var(--sigma-primary);
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }}
    
    </style>
    """


def load_sigma_responsive_css() -> str:
    """Load responsive design styles"""
    
    return f"""
    <style>
    /* ===== RESPONSIVE DESIGN STYLES ===== */
    
    @media (max-width: 768px) {{
        .sigma-header {{
            padding: var(--sigma-spacing-lg) var(--sigma-spacing-base);
        }}
        
        .sigma-header h1 {{
            font-size: {SigmaTypography.TEXT_2XL};
        }}
        
        .sigma-header .subtitle {{
            font-size: {SigmaTypography.TEXT_BASE};
        }}
        
        .sigma-header .features {{
            gap: var(--sigma-spacing-base);
        }}
        
        .feature-badge {{
            font-size: {SigmaTypography.TEXT_XS};
            padding: var(--sigma-spacing-sm) var(--sigma-spacing-base);
        }}
        
        .bmc-grid {{
            grid-template-columns: 1fr;
        }}
        
        .agent-pipeline {{
            flex-direction: column;
            gap: var(--sigma-spacing-lg);
        }}
        
        .agent-step::after {{
            display: none;
        }}
        
        .chat-message {{
            max-width: 95%;
        }}
        
        .sigma-card {{
            padding: var(--sigma-spacing-lg);
        }}
        
        .metric-card {{
            margin-bottom: var(--sigma-spacing-base);
        }}
        
        .main .block-container {{
            padding-left: var(--sigma-spacing-base);
            padding-right: var(--sigma-spacing-base);
        }}
    }}
    
    @media (max-width: 480px) {{
        .sigma-header h1 {{
            font-size: {SigmaTypography.TEXT_XL};
        }}
        
        .sigma-header .features {{
            flex-direction: column;
            align-items: center;
        }}
        
        .agent-pipeline {{
            padding: var(--sigma-spacing-base);
        }}
        
        .agent-icon {{
            width: 40px;
            height: 40px;
            font-size: {SigmaTypography.TEXT_BASE};
        }}
        
        .sigma-card {{
            padding: var(--sigma-spacing-base);
        }}
        
        .bmc-section {{
            min-height: 150px;
        }}
    }}
    
    @media (min-width: 1200px) {{
        .bmc-grid {{
            grid-template-columns: repeat(3, 1fr);
        }}
        
        .main .block-container {{
            max-width: 1400px;
        }}
    }}
    
    </style>
    """


def apply_sigma_styling():
    """Apply complete SIGMA styling to Streamlit app"""
    
    # Load all CSS components
    css_components = [
        load_sigma_base_css(),
        load_sigma_component_css(),
        load_sigma_agent_pipeline_css(),
        load_sigma_chat_css(),
        load_sigma_confidence_css(),
        load_sigma_bmc_css(),
        load_sigma_metric_css(),
        load_sigma_button_css(),
        load_sigma_responsive_css()
    ]
    
    # Combine and apply all styles
    combined_css = '\n'.join(css_components)
    st.markdown(combined_css, unsafe_allow_html=True)


def create_status_badge(status: str, text: str) -> str:
    """Create a styled status badge"""
    
    badge_classes = {
        'success': 'confidence-high',
        'warning': 'confidence-medium', 
        'danger': 'confidence-low',
        'info': 'confidence-medium'
    }
    
    badge_class = badge_classes.get(status, 'confidence-medium')
    
    return f'<span class="confidence-badge {status}">{text}</span>'


def create_metric_card(value: str, label: str, trend: Optional[str] = None) -> str:
    """Create a styled metric card"""
    
    trend_html = ""
    if trend:
        trend_class = "trend-up" if trend.startswith("+") else "trend-down" if trend.startswith("-") else "trend-neutral"
        trend_html = f'<div class="metric-trend {trend_class}">{trend}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {trend_html}
    </div>
    """


def create_progress_indicator(current: int, total: int, label: str) -> str:
    """Create a progress indicator"""
    
    percentage = (current / total) * 100 if total > 0 else 0
    
    return f"""
    <div style="margin: var(--sigma-spacing-base) 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: var(--sigma-spacing-xs);">
            <span style="font-size: {SigmaTypography.TEXT_SM}; font-weight: {SigmaTypography.FONT_MEDIUM};">{label}</span>
            <span style="font-size: {SigmaTypography.TEXT_SM}; color: var(--sigma-text-secondary);">{current}/{total}</span>
        </div>
        <div style="width: 100%; height: 8px; background: var(--sigma-border-light); border-radius: var(--sigma-radius-full);">
            <div style="width: {percentage}%; height: 100%; background: linear-gradient(90deg, var(--sigma-primary), var(--sigma-light-blue)); border-radius: var(--sigma-radius-full); transition: var(--sigma-transition-base);"></div>
        </div>
    </div>
    """


# Export main functions
__all__ = [
    'SigmaColors',
    'SigmaTypography', 
    'SigmaSpacing',
    'apply_sigma_styling',
    'create_status_badge',
    'create_metric_card',
    'create_progress_indicator'
]