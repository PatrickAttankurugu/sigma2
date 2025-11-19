"""
Business Design Phase Handler for SIGMA
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
        
        st.title("SIGMA Business Design")
        st.markdown("""
        Welcome to SIGMA! Let's start by defining the core elements of your business.
        This will help our AI provide personalized recommendations for your experiments and actions.
        """)
        
        self._render_progress_indicator()
        
        with st.form("business_design_form", clear_on_submit=False):
            st.subheader("Define Your Business")
            st.markdown("Fill in each section with 1-5 clear, specific items:")
            
            form_data = {}
            
            form_data['customer_segments'] = self._render_section_input(
                'customer_segments', bmc
            )
            
            form_data['value_propositions'] = self._render_section_input(
                'value_propositions', bmc
            )
            
            form_data['business_models'] = self._render_section_input(
                'business_models', bmc
            )
            
            form_data['market_opportunities'] = self._render_section_input(
                'market_opportunities', bmc
            )
            
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
        
        existing_values = bmc.get_section(section_name)
        
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
        
        errors = []
        for section_name, items in form_data.items():
            is_valid, error_msg = bmc.validate_section_input(section_name, items)
            if not is_valid:
                errors.append(error_msg)
        
        if errors:
            for error in errors:
                st.error(error)
            return False
        
        bmc.set_initial_business_design(
            customer_segments=form_data['customer_segments'],
            value_propositions=form_data['value_propositions'],
            business_models=form_data['business_models'],
            market_opportunities=form_data['market_opportunities']
        )
        
        self.log_user_action("business_design_completed", {
            "total_items": sum(len(items) for items in form_data.values()),
            "sections_completed": len([k for k, v in form_data.items() if v])
        })
        
        st.success("Business design completed! You can now start running experiments and actions.")
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
        
        st.sidebar.markdown("---")
        st.sidebar.caption(f"**Summary:** {bmc.get_business_summary()}")
    
    def get_sample_actions(self, bmc: BusinessModelCanvas) -> Dict[str, Dict[str, str]]:
        """Generate stage-aware sample actions based on user's business design"""
        if not bmc.is_complete():
            return {}
        
        primary_customer = bmc.customer_segments[0] if bmc.customer_segments else "target customers"
        primary_value_prop = bmc.value_propositions[0] if bmc.value_propositions else "main value proposition"
        business_stage = bmc.get_business_stage()
        
        if business_stage == "validation":
            return self._get_validation_stage_samples(primary_customer, primary_value_prop)
        elif business_stage == "growth":
            return self._get_growth_stage_samples(primary_customer, primary_value_prop)
        else:
            return self._get_scale_stage_samples(primary_customer, primary_value_prop)
    
    def _get_validation_stage_samples(self, primary_customer: str, primary_value_prop: str) -> Dict[str, Dict[str, str]]:
        """Get validation stage sample actions"""
        return {
            f"Customer Validation - {primary_customer.split()[0]} Interviews": {
                "title": f"Customer discovery interviews with 20 potential {primary_customer.lower()}",
                "outcome": "Successful", 
                "description": f"Conducted in-depth interviews with potential customers in the {primary_customer.lower()} segment to validate our assumptions about their pain points and willingness to pay.",
                "results": f"""
VALIDATION SUCCESS:

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

Key Insights for Next Steps:
- Implementation concerns around learning curve (need onboarding focus)
- Strong preference for integration with existing tools
- Mobile access ranked as "critical" by 67%
- Customer support during onboarding is essential for adoption
- Demo request rate: 73% (strong validation signal)
"""
            },
            
            f"MVP Testing - Core {primary_value_prop.split()[0]} Feature": {
                "title": f"2-week MVP test of core feature delivering {primary_value_prop.lower().split()[0]} functionality",
                "outcome": "Inconclusive",
                "description": f"Built and tested minimum viable version of our core feature with early adopters to validate product-market fit.",
                "results": f"""
MIXED MVP SIGNALS:

Positive Indicators:
- 67% user activation rate (industry average: 40%)
- Daily active usage by 45% of registered users
- Average session time: 23 minutes (target: 15 minutes)
- 89% completion rate for onboarding flow
- Net Promoter Score: 42 (above average for early-stage)

Concerning Patterns:
- Feature discovery issues - users missed key functionality
- Mobile experience significantly weaker than desktop (58% vs 89% satisfaction)
- Integration setup took average 3.2 hours (target: 30 minutes) 
- 34% churn rate after week 1 (too high for sustainable growth)

Inconclusive Elements:
- Limited sample size (47 users across 2 weeks)
- Seasonal factors may have affected usage patterns
- Technical issues in days 3-5 skewed engagement data
- Mixed feedback on core value proposition delivery

Critical Next Steps Needed:
- Extend test to 30 days with larger user group (200+ users)
- Focus on mobile experience improvements and faster integration
- Conduct user interviews to understand churn reasons
"""
            },
            
            "Pricing Model Validation Test": {
                "title": "A/B testing of pricing strategies with early customer segment",
                "outcome": "Failed",
                "description": "Tested three different pricing models to find optimal revenue approach and validate customer willingness to pay.",
                "results": f"""
PRICING VALIDATION FAILURE:

Models Tested (3 weeks each):
- Premium Model: $99/month with full features (50 prospects)
- Freemium Model: Free basic + $49/month premium (50 prospects)
- Usage-Based: $2 per transaction/use (50 prospects)

Disappointing Results:
- Premium: 12% conversion rate (industry: 15-20%)
- Freemium: 67% stayed on free forever, only 3% upgraded
- Usage-Based: Unpredictable costs scared away 78% of prospects

Key Learnings About Market Reality:
- Our target customers are more price-sensitive than assumed
- Current value proposition doesn't justify premium pricing positioning
- Free users consume significant support resources without revenue
- Usage-based model creates adoption barriers due to cost uncertainty
- {primary_customer.lower()} segment has tighter budget constraints than researched

Failed Assumptions:
- Overestimated willingness to pay for {primary_value_prop.lower()}
- Underestimated price comparison shopping behavior in this market
- Value demonstration insufficient for justifying premium pricing
- Competition pricing 40-60% lower than our premium model
"""
            }
        }
    
    def _get_growth_stage_samples(self, primary_customer: str, primary_value_prop: str) -> Dict[str, Dict[str, str]]:
        """Get growth stage sample actions"""
        return {
            f"Channel Optimization - {primary_customer.split()[0]} Acquisition": {
                "title": f"3-month multi-channel acquisition test for {primary_customer.lower()} segment",
                "outcome": "Successful",
                "description": f"Optimized customer acquisition channels to scale growth efficiently while maintaining quality of {primary_customer.lower()} customers.",
                "results": f"""
CHANNEL OPTIMIZATION SUCCESS:

Channel Performance Results:
- Content Marketing: $45 CAC, 89% quality score, 3.2x LTV/CAC
- LinkedIn Ads: $67 CAC, 92% quality score, 2.8x LTV/CAC  
- Partner Referrals: $23 CAC, 95% quality score, 4.1x LTV/CAC
- Google Ads: $89 CAC, 76% quality score, 2.1x LTV/CAC

Scaling Success Metrics:
- Overall CAC reduced from $120 to $58 (52% improvement)
- Monthly new customer growth: 145% increase
- Customer quality score improved by 23%
- {primary_value_prop.lower()} adoption rate: 87% within first month

Strategic Channel Focus:
- Partner referrals show highest efficiency and quality
- Content marketing provides best long-term brand building
- LinkedIn ads effective for enterprise segment expansion
- Google ads useful for competitive conquest campaigns

Growth Opportunities Identified:
- Scale partner program to reach 50+ active referrers
- Invest heavily in content marketing system and automation
- Test expansion into adjacent customer segments using proven channels
"""
            },
            
            f"Retention Optimization - {primary_value_prop.split()[0]} Delivery": {
                "title": f"Customer success program to improve {primary_value_prop.lower()} delivery and retention",
                "outcome": "Inconclusive", 
                "description": f"Implemented customer success initiatives to reduce churn and increase expansion revenue from existing {primary_customer.lower()}.",
                "results": f"""
MIXED RETENTION RESULTS:

Positive Retention Signals:
- Monthly churn decreased from 8.5% to 6.2% (27% improvement)
- Net Revenue Retention improved to 112% (from 95%)
- Customer satisfaction (CSAT) increased to 4.2/5 (from 3.7/5)
- Support ticket volume decreased by 34%

Concerning Patterns:
- Improvement concentrated in first 6 months of customer lifecycle
- Long-term customers (12+ months) showed minimal retention improvement
- Enterprise segment responded better than SMB segment
- High-touch success program not economically viable for smaller accounts

Inconclusive Elements:
- Expansion revenue growth rate slowing after initial boost
- Unclear which specific success tactics drive retention vs satisfaction
- Customer success ROI varies significantly by segment and use case
- Long-term sustainability of current success model questionable

Critical Questions for Next Phase:
- How to scale customer success efficiently across all segments?
- What drives long-term customer expansion beyond initial satisfaction?
- Can we automate success processes without losing effectiveness?
"""
            }
        }
    
    def _get_scale_stage_samples(self, primary_customer: str, primary_value_prop: str) -> Dict[str, Dict[str, str]]:
        """Get scale stage sample actions"""
        return {
            f"Market Expansion - International {primary_customer.split()[0]} Segment": {
                "title": f"6-month international expansion test targeting {primary_customer.lower()} in European markets",
                "outcome": "Failed",
                "description": f"Attempted to replicate our successful {primary_value_prop.lower()} model in UK, Germany, and France markets.",
                "results": f"""
INTERNATIONAL EXPANSION FAILURE:

Market Entry Results by Country:
- UK: 23% of projected customer acquisition, high CAC ($340 vs $67 domestic)
- Germany: 31% of projected acquisition, regulatory compliance issues
- France: 18% of projected acquisition, language/cultural barriers significant

Failed Assumptions:
- Product-market fit doesn't translate directly across markets
- {primary_value_prop.lower()} messaging requires significant localization
- Competitive landscape much more crowded in European markets
- Local partnerships essential but difficult to establish remotely

Operational Challenges:
- Customer support in multiple time zones strained resources
- Legal/compliance requirements added 40% operational overhead
- Payment processing complexities reduced conversion by 28%
- Brand recognition near zero, requiring massive marketing investment

Strategic Learnings:
- International expansion requires dedicated in-market teams
- Product features need localization beyond just language translation
- Partnership strategy must be country-specific and relationship-driven
- Timing may be premature - domestic market still has growth potential

Next Steps Required:
- Reassess domestic market saturation before international retry
- Develop partnerships-first international strategy vs direct expansion
- Consider acquisition of local competitors as expansion method
"""
            }
        }