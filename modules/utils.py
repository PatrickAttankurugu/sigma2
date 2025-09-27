"""
Utilities and Logging for SIGMA Agentic AI Actions Co-pilot
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Dict, Any


def setup_logging():
    """Configure structured logging for the application"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/sigma_copilot_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create specialized loggers
    app_logger = logging.getLogger("sigma.app")
    ai_logger = logging.getLogger("sigma.ai")
    bmc_logger = logging.getLogger("sigma.bmc")
    metrics_logger = logging.getLogger("sigma.metrics")
    quality_logger = logging.getLogger("sigma.quality")
    
    return app_logger, ai_logger, bmc_logger, metrics_logger, quality_logger


class LoggingMixin:
    """Mixin to add structured logging capabilities"""
    
    @staticmethod
    def log_user_action(action_type: str, details: Dict[str, Any]):
        """Log user interactions"""
        app_logger = logging.getLogger("sigma.app")
        app_logger.info(f"USER_ACTION: {action_type}", extra={
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
    
    @staticmethod 
    def log_ai_performance(operation: str, duration_ms: int, success: bool, details: Dict[str, Any]):
        """Log AI operation performance"""
        ai_logger = logging.getLogger("sigma.ai")
        ai_logger.info(f"AI_PERFORMANCE: {operation}", extra={
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
    
    @staticmethod
    def log_bmc_change(section: str, change_type: str, confidence: float, auto_applied: bool):
        """Log BMC modifications"""
        bmc_logger = logging.getLogger("sigma.bmc")
        bmc_logger.info(f"BMC_CHANGE: {section}.{change_type}", extra={
            "section": section,
            "change_type": change_type,
            "confidence": confidence,
            "auto_applied": auto_applied,
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    def log_session_metrics(session_id: str, metrics: Dict[str, Any]):
        """Log session-level metrics"""
        metrics_logger = logging.getLogger("sigma.metrics")
        metrics_logger.info(f"SESSION_METRICS: {session_id}", extra={
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })


class SessionMetrics(LoggingMixin):
    """Track session-level metrics and usage patterns"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.actions_analyzed = 0
        self.changes_proposed = 0
        self.changes_applied = 0
        self.auto_mode_usage = 0
        self.custom_actions_used = 0
        self.quality_retries = 0
        self.business_design_completed = False
        
        # Log session start
        self.log_session_metrics(self.session_id, {
            "event": "session_started",
            "start_time": self.start_time.isoformat()
        })
    
    def record_business_design_completed(self, sections_count: int):
        """Record business design phase completion"""
        self.business_design_completed = True
        self.log_user_action("business_design_completed", {
            "session_id": self.session_id,
            "sections_count": sections_count
        })
    
    def record_action_analyzed(self, outcome: str, quality_score: float = None, retries_used: int = 0):
        """Record an action analysis"""
        self.actions_analyzed += 1
        self.custom_actions_used += 1
        
        if retries_used > 0:
            self.quality_retries += 1
        
        self.log_user_action("action_analyzed", {
            "session_id": self.session_id,
            "outcome": outcome,
            "total_analyzed": self.actions_analyzed,
            "quality_score": quality_score,
            "retries_used": retries_used
        })
    
    def record_changes_proposed(self, count: int, avg_confidence: float):
        """Record proposed changes"""
        self.changes_proposed += count
        
        self.log_user_action("changes_proposed", {
            "session_id": self.session_id,
            "count": count,
            "avg_confidence": avg_confidence,
            "total_proposed": self.changes_proposed
        })
    
    def record_changes_applied(self, count: int, auto_applied: bool):
        """Record applied changes"""
        self.changes_applied += count
        if auto_applied:
            self.auto_mode_usage += 1
        
        self.log_bmc_change("multiple", "apply", 1.0, auto_applied)
        self.log_user_action("changes_applied", {
            "session_id": self.session_id,
            "count": count,
            "auto_applied": auto_applied,
            "total_applied": self.changes_applied
        })
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "business_design_completed": self.business_design_completed,
            "actions_analyzed": self.actions_analyzed,
            "changes_proposed": self.changes_proposed,
            "changes_applied": self.changes_applied,
            "auto_mode_usage": self.auto_mode_usage,
            "quality_retries": self.quality_retries,
            "custom_actions": self.custom_actions_used,
            "engagement_score": min(self.actions_analyzed * 2 + self.changes_applied, 10)
        }