"""
Business Model Canvas for SIGMA - Business Design Phase
"""

from typing import List, Dict, Any
from .utils import LoggingMixin


class BusinessModelCanvas(LoggingMixin):
    """Simplified BMC with 4 sections from Business Design Phase"""
    
    def __init__(self):
        self.customer_segments = []
        self.value_propositions = []
        self.business_models = []
        self.market_opportunities = []
        
        self._is_complete = False
        
        self.log_user_action("bmc_initialized", {
            "sections": 4,
            "initial_state": "empty"
        })

    def get_section_names(self) -> List[str]:
        """Get all BMC section names"""
        return ['customer_segments', 'value_propositions', 'business_models', 'market_opportunities']

    def get_section(self, section_name: str) -> List[str]:
        """Get items from a BMC section"""
        return getattr(self, section_name, [])

    def update_section(self, section_name: str, items: List[str]):
        """Update a BMC section with new items and log the change"""
        old_items = getattr(self, section_name, [])
        setattr(self, section_name, items)
        
        self.log_user_action("bmc_section_updated", {
            "section": section_name,
            "old_count": len(old_items),
            "new_count": len(items),
            "items_added": len(items) - len(old_items)
        })
        
        self._update_completion_status()

    def set_initial_business_design(self, customer_segments: List[str], value_propositions: List[str], 
                                  business_models: List[str], market_opportunities: List[str]):
        """Set initial business design from user input"""
        self.customer_segments = customer_segments
        self.value_propositions = value_propositions
        self.business_models = business_models
        self.market_opportunities = market_opportunities
        
        self._is_complete = True
        
        self.log_user_action("business_design_set", {
            "total_items": sum([
                len(customer_segments), len(value_propositions), 
                len(business_models), len(market_opportunities)
            ]),
            "sections_filled": sum([
                len(customer_segments) > 0, len(value_propositions) > 0,
                len(business_models) > 0, len(market_opportunities) > 0
            ])
        })

    def get_all_sections(self) -> Dict[str, List[str]]:
        """Get all BMC sections as dictionary"""
        return {section: getattr(self, section) for section in self.get_section_names()}

    def is_complete(self) -> bool:
        """Check if business design phase is complete"""
        return self._is_complete and all(
            len(getattr(self, section)) > 0 for section in self.get_section_names()
        )

    def get_completeness_score(self) -> float:
        """Calculate BMC completeness percentage"""
        filled_sections = sum(1 for section in self.get_section_names() if getattr(self, section))
        return filled_sections / len(self.get_section_names())

    def get_business_summary(self) -> str:
        """Get a concise summary of the business for AI context"""
        if not self.is_complete():
            return "Business design not yet complete"
        
        summary = []
        
        if self.customer_segments:
            summary.append(f"Target Customers: {', '.join(self.customer_segments[:2])}")
        
        if self.value_propositions:
            summary.append(f"Value Proposition: {self.value_propositions[0]}")
        
        if self.business_models:
            summary.append(f"Business Model: {self.business_models[0]}")
        
        if self.market_opportunities:
            summary.append(f"Market Opportunity: {self.market_opportunities[0]}")
        
        return " | ".join(summary)

    def get_business_stage(self) -> str:
        """Detect current business stage based on BMC content and maturity"""
        if not self.is_complete():
            return "validation"
        
        all_content = " ".join([
            " ".join(self.customer_segments),
            " ".join(self.value_propositions),
            " ".join(self.business_models),
            " ".join(self.market_opportunities)
        ]).lower()
        
        growth_indicators = [
            'scale', 'scaling', 'growth', 'expand', 'expansion', 'market share',
            'revenue', 'profit', 'optimization', 'retention', 'churn',
            'channels', 'distribution', 'partnerships'
        ]
        
        scale_indicators = [
            'international', 'global', 'enterprise', 'automation', 'efficiency',
            'competitive advantage', 'market leader', 'dominance', 'margins',
            'operational', 'systems', 'processes'
        ]
        
        validation_indicators = [
            'test', 'validate', 'prototype', 'mvp', 'pilot', 'experiment',
            'interview', 'survey', 'feedback', 'assumption', 'hypothesis'
        ]
        
        growth_score = sum(1 for indicator in growth_indicators if indicator in all_content)
        scale_score = sum(1 for indicator in scale_indicators if indicator in all_content)
        validation_score = sum(1 for indicator in validation_indicators if indicator in all_content)
        
        if scale_score >= 2 and self.get_completeness_score() == 1.0:
            return "scale"
        elif growth_score >= 3 or (growth_score >= 2 and self.get_completeness_score() >= 0.75):
            return "growth"
        else:
            return "validation"

    def get_risk_assessment(self) -> str:
        """Identify potential risks in current business model"""
        if not self.is_complete():
            return "Complete business design to assess risks"
        
        risks = []
        
        if len(self.customer_segments) == 1:
            risks.append("Single customer segment dependency")
        
        vp_content = " ".join(self.value_propositions).lower()
        if not any(word in vp_content for word in ['unique', 'different', 'advantage', 'better']):
            risks.append("Value proposition lacks differentiation")
        
        bm_content = " ".join(self.business_models).lower()
        if len(self.business_models) == 1 and not any(word in bm_content for word in ['recurring', 'subscription', 'repeat']):
            risks.append("Single revenue stream without recurring income")
        
        mo_content = " ".join(self.market_opportunities).lower()
        if not any(word in mo_content for word in ['large', 'growing', 'billion', 'million', 'market size']):
            risks.append("Market size unclear or potentially limited")
        
        if not risks:
            return "Low risk profile - well-balanced business model"
        elif len(risks) <= 2:
            return f"Medium risk: {'; '.join(risks[:2])}"
        else:
            return f"High risk: {'; '.join(risks[:3])}"

    def get_enhanced_business_context(self) -> str:
        """Get comprehensive business context for AI analysis"""
        if not self.is_complete():
            return "Business design phase not completed"
        
        stage = self.get_business_stage()
        risk_assessment = self.get_risk_assessment()
        completion = self.get_completeness_score()
        
        context = f"""
BUSINESS PROFILE:
Stage: {stage.title()}
Risk Assessment: {risk_assessment}
Completion: {completion:.0%}

BUSINESS MODEL DETAILS:
Customer Segments ({len(self.customer_segments)}):
{chr(10).join(f"• {item}" for item in self.customer_segments)}

Value Propositions ({len(self.value_propositions)}):
{chr(10).join(f"• {item}" for item in self.value_propositions)}

Business Models ({len(self.business_models)}):
{chr(10).join(f"• {item}" for item in self.business_models)}

Market Opportunities ({len(self.market_opportunities)}):
{chr(10).join(f"• {item}" for item in self.market_opportunities)}

STRATEGIC CONTEXT:
Based on the current {stage} stage, focus recommendations on {self._get_stage_focus(stage)}.
Risk mitigation should address: {risk_assessment}
        """
        
        return context.strip()

    def _get_stage_focus(self, stage: str) -> str:
        """Get stage-specific strategic focus"""
        if stage == "validation":
            return "customer validation, assumption testing, MVP development, and market fit validation"
        elif stage == "growth":
            return "scaling customer acquisition, optimizing conversion rates, expanding market reach, and improving retention"
        else:
            return "operational efficiency, market expansion, competitive differentiation, and sustainable growth systems"

    def _update_completion_status(self):
        """Update completion status based on current sections"""
        self._is_complete = all(
            len(getattr(self, section)) > 0 for section in self.get_section_names()
        )

    def get_section_display_name(self, section_name: str) -> str:
        """Get user-friendly display name for sections"""
        display_names = {
            'customer_segments': 'Customer Segments',
            'value_propositions': 'Value Propositions', 
            'business_models': 'Business Models',
            'market_opportunities': 'Market Opportunities'
        }
        return display_names.get(section_name, section_name.replace('_', ' ').title())

    def get_section_description(self, section_name: str) -> str:
        """Get description for each section to help users"""
        descriptions = {
            'customer_segments': 'Who are your target customers? Define specific groups of people or organizations you aim to reach.',
            'value_propositions': 'What unique value do you provide? Describe the benefits, products or services that create value for customers.',
            'business_models': 'How do you create, deliver and capture value? Describe your approach to generating revenue.',
            'market_opportunities': 'What market opportunities are you pursuing? Describe the problems you solve or needs you address.'
        }
        return descriptions.get(section_name, '')

    def validate_section_input(self, section_name: str, items: List[str]) -> tuple[bool, str]:
        """Validate section input"""
        if not items or len(items) == 0:
            return False, f"{self.get_section_display_name(section_name)} cannot be empty"
        
        if any(len(item.strip()) < 10 for item in items):
            return False, f"Each {self.get_section_display_name(section_name)} item should be at least 10 characters"
        
        if len(items) > 5:
            return False, f"Maximum 5 items allowed per section"
        
        return True, "Valid input"