"""
Business Design Phase Handler for SIGMA
Manages the initial business design form and validation
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
from .bmc_canvas import BusinessModelCanvas
from .utils import LoggingMixin


class BusinessDesignManager(LoggingMixin):
    """Manages the business design phase of the entrepreneur journey"""
    
    def __init__(self):
        self.sections = [
            'customer_segments',
            'value_propositions', 
            'business_models',
            'market_opportunities'
        ]
    
    def render_business_design_form(self, bmc: BusinessModelCanvas) -> bool:
        """Render the business design form and return True if completed"""
        
        st.title("🚀 SIGMA Business Design")
        st.markdown("""
        Welcome to SIGMA! Let's start by defining the core elements of your business.
        This will help our AI provide personalized recommendations for your experiments and actions.
        """)
        
        # Progress indicator
        self._render_progress_indicator()
        
        # Form container
        with st.form("business_design_form", clear_on_submit=False):
            st.subheader("Define Your Business")
            st.markdown("Fill in each section with 1-5 clear, specific items:")
            
            form_data = {}
            
            # Customer Segments
            form_data['customer_segments'] = self._render_section_input(
                'customer_segments', bmc
            )
            
            # Value Propositions
            form_data['value_propositions'] = self._render_section_input(
                'value_propositions', bmc
            )
            
            # Business Models
            form_data['business_models'] = self._render_section_input(
                'business_models', bmc
            )
            
            # Market Opportunities
            form_data['market_opportunities'] = self._render_section_input(
                'market_opportunities', bmc
            )
            
            # Submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button(
                    "Complete Business Design", 
                    use_container_width=True,
                    type="primary"
                )
            
            if submitted:
                return self._process_form_submission(form_data, bmc)
        
        return False
    
    def _render_section_input(self, section_name: str, bmc: BusinessModelCanvas) -> List[str]:
        """Render input for a specific section"""
        display_name = bmc.get_section_display_name(section_name)
        description = bmc.get_section_description(section_name)
        
        st.markdown(f"**{display_name}**")
        st.caption(description)
        
        # Get existing values if any
        existing_values = bmc.get_section(section_name)
        
        # Create text areas for up to 5 items
        items = []
        for i in range(5):
            default_value = existing_values[i] if i < len(existing_values) else ""
            placeholder = self._get_placeholder_text(section_name, i)
            
            value = st.text_area(
                f"{display_name} {i+1}",
                value=default_value,
                placeholder=placeholder,
                height=80,
                key=f"{section_name}_{i}",
                label_visibility="collapsed" if i > 0 else "visible"
            )
            
            if value.strip():
                items.append(value.strip())
        
        st.markdown("---")
        return items
    
    def _get_placeholder_text(self, section_name: str, index: int) -> str:
        """Get placeholder text for form fields"""
        placeholders = {
            'customer_segments': [
                "e.g., Small business owners in urban areas (25-45 years old)",
                "e.g., Tech-savvy entrepreneurs seeking growth solutions", 
                "e.g., Service-based businesses with 5-50 employees",
                "e.g., B2B companies looking for digital transformation",
                "e.g., Startups in the fintech or e-commerce space"
            ],
            'value_propositions': [
                "e.g., AI-powered analytics that reduce decision-making time by 70%",
                "e.g., Automated workflow that saves 20 hours per week",
                "e.g., Real-time insights that increase revenue by 30%",
                "e.g., Simple integration that works with existing tools",
                "e.g., 24/7 support with guaranteed 2-hour response time"
            ],
            'business_models': [
                "e.g., SaaS subscription with tiered pricing ($50-500/month)",
                "e.g., Freemium model with premium features for $30/month",
                "e.g., Marketplace taking 10% commission on transactions",
                "e.g., One-time software license with annual support fees",
                "e.g., Consulting services at $150/hour + software tools"
            ],
            'market_opportunities': [
                "e.g., Growing demand for digital transformation in SMEs",
                "e.g., $50B market for business automation tools by 2026",
                "e.g., 40% of businesses still use manual processes",
                "e.g., Remote work driving need for better collaboration tools",
                "e.g., Regulatory changes requiring better data management"
            ]
        }
        
        section_placeholders = placeholders.get(section_name, [""])
        return section_placeholders[index] if index < len(section_placeholders) else ""
    
    def _process_form_submission(self, form_data: Dict[str, List[str]], bmc: BusinessModelCanvas) -> bool:
        """Process form submission and validate data"""
        
        # Validate all sections
        errors = []
        for section_name, items in form_data.items():
            is_valid, error_msg = bmc.validate_section_input(section_name, items)
            if not is_valid:
                errors.append(error_msg)
        
        # Show errors if any
        if errors:
            for error in errors:
                st.error(error)
            return False
        
        # All valid - save to BMC
        bmc.set_initial_business_design(
            customer_segments=form_data['customer_segments'],
            value_propositions=form_data['value_propositions'],
            business_models=form_data['business_models'],
            market_opportunities=form_data['market_opportunities']
        )
        
        # Log completion
        self.log_user_action("business_design_completed", {
            "total_items": sum(len(items) for items in form_data.values()),
            "sections_completed": len([k for k, v in form_data.items() if v])
        })
        
        st.success("🎉 Business design completed! You can now start running experiments and actions.")
        st.balloons()
        
        return True
    
    def _render_progress_indicator(self):
        """Render progress indicator for the business design phase"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <div style="background: #f0f2f6; border-radius: 10px; padding: 15px;">
                    <strong>Step 1 of 2: Business Design Phase</strong><br>
                    <small>Next: Actions & Experiments</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_business_summary(self, bmc: BusinessModelCanvas):
        """Render a summary of the completed business design"""
        if not bmc.is_complete():
            return
        
        st.sidebar.subheader("Your Business")
        
        sections = bmc.get_all_sections()
        for section_name, items in sections.items():
            display_name = bmc.get_section_display_name(section_name)
            
            with st.sidebar.expander(f"{display_name} ({len(items)})"):
                for i, item in enumerate(items, 1):
                    st.write(f"{i}. {item}")
        
        # Business summary
        st.sidebar.markdown("---")
        st.sidebar.caption(f"**Summary:** {bmc.get_business_summary()}")
    
    def get_sample_actions(self, bmc: BusinessModelCanvas) -> Dict[str, Dict[str, str]]:
        """Generate sample actions based on user's business design"""
        if not bmc.is_complete():
            return {}
        
        # Get business context
        primary_customer = bmc.customer_segments[0] if bmc.customer_segments else "target customers"
        primary_value_prop = bmc.value_propositions[0] if bmc.value_propositions else "main value proposition"
        
        return {
            f"Customer Validation with {primary_customer.split()[0]} Segment": {
                "title": f"Customer interviews with 20 potential {primary_customer.lower()}",
                "outcome": "Successful", 
                "description": f"Conducted in-depth interviews with potential customers in the {primary_customer.lower()} segment to validate our assumptions about their pain points and willingness to pay.",
                "results": f"""
VALIDATION RESULTS:

Customer Pain Points Confirmed:
- 85% confirmed the exact problem we're solving exists
- 78% currently use manual/inefficient solutions  
- 92% expressed frustration with current alternatives
- Average time wasted per week: 12 hours on this problem

Value Proposition Validation:
- 89% found our solution concept "very valuable" 
- 76% would recommend to peers based on description
- Willingness to pay confirmed at proposed price point
- {primary_value_prop.lower()} resonated strongly with 91% of interviewees

Key Insights:
- Implementation concerns around learning curve (addressed)
- Strong preference for integration with existing tools
- Mobile access ranked as "critical" by 67%
- Customer support during onboarding is essential
"""
            },
            
            f"MVP Testing - {primary_value_prop.split()[0]} Feature": {
                "title": f"2-week MVP test of core {primary_value_prop.lower().split()[0]} functionality",
                "outcome": "Inconclusive",
                "description": f"Built and tested minimum viable version of our core feature that delivers {primary_value_prop.lower()} to early adopters.",
                "results": f"""
MIXED MVP RESULTS:

Positive Signals:
- 67% user activation rate (industry average: 40%)
- Daily active usage by 45% of registered users
- Average session time: 23 minutes (target: 15 minutes)
- 89% completion rate for onboarding flow

Challenges Identified:
- Feature discovery issues - users missed key functionality
- Mobile experience significantly weaker than desktop
- Integration setup took average 3.2 hours (target: 30 minutes) 
- 34% churn rate after week 1 (too high)

Inconclusive Elements:
- Limited sample size (47 users across 2 weeks)
- Seasonal factors may have affected usage patterns
- Technical issues in days 3-5 skewed engagement data
- Need longer testing period to assess retention properly

Next Steps Needed:
- Extend test to 30 days with larger user group
- Focus on mobile experience improvements
- Simplify integration process based on user feedback
"""
            },
            
            "Pricing Strategy Validation": {
                "title": "A/B testing of pricing models with different customer segments",
                "outcome": "Failed",
                "description": "Tested three different pricing strategies across different customer segments to optimize revenue and adoption.",
                "results": f"""
PRICING TEST FAILURE:

Models Tested:
- Premium Model: $99/month with full features
- Freemium Model: Free basic + $49/month premium  
- Usage-Based: $2 per transaction/use

Results by Model:
- Premium: 12% conversion rate (too low)
- Freemium: 67% stayed on free forever (95% never upgraded)
- Usage-Based: Unpredictable costs scared away 78% of prospects

Key Learnings:
- Our target customers are extremely price-sensitive
- Current value proposition doesn't justify premium pricing
- Free users consume support resources without revenue
- Usage-based model creates adoption barriers

Market Reality Check:
- Competitors pricing 40-60% lower than our premium model
- Customer acquisition cost ($180) too high for $49/month model
- {primary_customer.lower()} segment has limited budget flexibility
- Enterprise features needed to justify higher pricing

Failed Assumptions:
- Overestimated willingness to pay for {primary_value_prop.lower()}
- Underestimated price comparison shopping behavior
- Value demonstration insufficient for premium positioning
"""
            }
        }