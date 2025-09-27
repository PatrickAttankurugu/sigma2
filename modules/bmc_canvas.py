"""
Business Model Canvas for SIGMA - Business Design Phase
Focused on the 4 core sections that entrepreneurs define first
"""

from typing import List, Dict, Any
from .utils import LoggingMixin


class BusinessModelCanvas(LoggingMixin):
    """Simplified BMC with 4 sections from Business Design Phase"""
    
    def __init__(self):
        # Initialize empty sections - entrepreneurs will fill these
        self.customer_segments = []
        self.value_propositions = []
        self.business_models = []
        self.market_opportunities = []
        
        # Track if business design is complete
        self._is_complete = False
        
        # Log BMC initialization
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
        
        # Log the section update
        self.log_user_action("bmc_section_updated", {
            "section": section_name,
            "old_count": len(old_items),
            "new_count": len(items),
            "items_added": len(items) - len(old_items)
        })
        
        # Update completion status
        self._update_completion_status()

    def set_initial_business_design(self, customer_segments: List[str], value_propositions: List[str], 
                                  business_models: List[str], market_opportunities: List[str]):
        """Set initial business design from user input"""
        self.customer_segments = customer_segments
        self.value_propositions = value_propositions
        self.business_models = business_models
        self.market_opportunities = market_opportunities
        
        self._is_complete = True
        
        # Log business design completion
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