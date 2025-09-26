"""
SEMA Business Data for Agentic AI Actions Co-pilot Demo

This module provides real business model canvas data and action scenarios
from SEMA, an AI-powered predictive surveillance startup in Ghana.
This demonstrates authentic African startup intelligence for the Seedstars assignment.
"""

from datetime import datetime, timedelta
from typing import Dict, List

from business_models import (
    BusinessModelCanvas,
    CompletedAction,
    ActionOutcome
)


def get_sample_business_model_canvas() -> BusinessModelCanvas:
    """Return SEMA's actual business model canvas for predictive surveillance."""
    return BusinessModelCanvas(
        customer_segments=[
            "Gated communities in urban Accra with approximately 1500 homes",
            "Tech-savvy homeowners aged 35-60 seeking proactive security solutions",
            "Well-educated professionals who value advanced technology for family safety",
            "Property management companies overseeing multiple residential complexes",
            "High-net-worth individuals with existing security infrastructure"
        ],
        value_propositions=[
            "Enhanced security through predictive surveillance using AI/ML algorithms",
            "Proactive crime prevention with AI-driven real-time alerts and notifications",
            "User-friendly dashboard for easy monitoring and control of security systems",
            "Significant reduction in security incidents compared to reactive systems",
            "Integration with existing CCTV infrastructure to add predictive capabilities"
        ],
        channels=[
            "Direct sales through dedicated sales team targeting gated communities",
            "Online marketing including social media campaigns and LinkedIn outreach",
            "Partnerships with property developers and security companies",
            "Referral programs from existing satisfied customers",
            "Industry events and security trade shows in Ghana"
        ],
        customer_relationships=[
            "Personal assistance through dedicated account managers for enterprise clients",
            "24/7 technical support for system monitoring and maintenance",
            "Regular training sessions for security personnel and homeowners",
            "Community-based customer success programs within gated communities",
            "Proactive system health monitoring and preventive maintenance"
        ],
        revenue_streams=[
            "Monthly SaaS subscriptions - Basic tier at $4 per month per camera",
            "Premium subscription tier at $10 per month with advanced analytics",
            "One-time installation and setup services ranging from $200-500",
            "Hardware sales commission from camera and equipment partnerships",
            "Custom integration services for complex security infrastructure"
        ],
        key_resources=[
            "Skilled AI developers with computer vision and machine learning expertise",
            "Cloud computing infrastructure on AWS for real-time data processing",
            "Security cameras and IoT hardware from strategic partnerships",
            "Proprietary AI algorithms for predictive crime detection and analysis",
            "Customer database and behavioral pattern recognition systems"
        ],
        key_activities=[
            "Software development for predictive crime detection algorithms",
            "Real-time data analysis from CCTV feeds using computer vision",
            "Customer support and technical training for security systems",
            "Marketing and sales activities targeting property developers",
            "Continuous improvement of AI models based on incident data"
        ],
        key_partnerships=[
            "Ghana Digital Centres Limited for co-working space and business incubation",
            "Amazon Web Services (AWS) for cloud computing infrastructure",
            "Hikvision Ghana for security camera hardware and technical support",
            "Queens University for research collaboration and student internships",
            "Local security companies for installation and maintenance services"
        ],
        cost_structure=[
            "Cloud hosting and infrastructure costs - approximately $6,000 per month",
            "Team salaries and benefits for developers and support staff - $20,000 monthly",
            "Data storage and processing costs for video analytics - $10,000 monthly",
            "Marketing and customer acquisition expenses - $8,000 per month",
            "Hardware procurement and inventory management costs"
        ],
        version="1.3.2",
        tags=["security-tech", "predictive-surveillance", "ghana-startup", "b2b-saas", "ai-ml"]
    )


def get_sample_completed_actions() -> List[CompletedAction]:
    """Return realistic SEMA action scenarios for agentic intelligence demonstration."""

    actions = []

    # Scenario 1: Highly Successful Customer Pilot (SUCCESSFUL)
    actions.append(CompletedAction(
        title="3-Month Predictive Security Pilot at Trasacco Estates",
        description="Deployed SEMA's predictive surveillance system across Trasacco Estates' Phase 4, covering 200 homes with 150 existing CCTV cameras. Implemented AI-driven threat detection, real-time alerts, and behavioral pattern analysis to prevent security incidents before they occur.",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        EXCEPTIONAL PILOT PERFORMANCE RESULTS:
        
        === SECURITY IMPACT METRICS ===
        • Crime prediction accuracy: 89% (exceeded 75% target)
        • Security incidents prevented: 23 verified cases in 3 months
        • False positive rate: Only 12% (industry standard 35%)
        • Average alert response time: 2.3 minutes (target was <5 minutes)
        • System uptime: 99.7% availability across all camera feeds
        
        === CUSTOMER SATISFACTION ===
        • Resident satisfaction score: 9.2/10 (surveyed 180 households)
        • Property management enthusiasm: "Game-changing technology"
        • Security guard feedback: 94% report system makes job more effective
        • Homeowner adoption: 87% activated mobile alerts within first month
        • Renewal intent: 91% of residents want permanent installation
        
        === TECHNICAL PERFORMANCE ===
        • Successfully integrated with existing Hikvision camera infrastructure
        • Real-time processing handled 150 concurrent video feeds efficiently
        • Mobile app usage: 78% monthly active users among residents
        • Data processing latency: Average 1.2 seconds from detection to alert
        • AI model accuracy improved 15% during pilot from learning
        
        === BUSINESS VALIDATION ===
        • Monthly recurring revenue potential: $1,800 from this community alone
        • Installation cost recovery: 3.2 months (better than 6-month target)
        • Customer acquisition cost: $45 per household (20% below budget)
        • Trasacco management requesting expansion to 3 additional phases
        • Generated 5 qualified leads from word-of-mouth referrals
        
        === COMPETITIVE ADVANTAGE DEMONSTRATED ===
        • Traditional security firms reactive approach vs our predictive prevention
        • 67% reduction in actual security incidents compared to previous year
        • Cost savings for residents: $200/month reduced private security expenses
        • Property value increase: 8% appreciation cited by real estate agents
        """,
        completion_date=datetime.now() - timedelta(days=2),
        success_metrics={
            "prediction_accuracy": 0.89,
            "incident_prevention": 23,
            "customer_satisfaction": 9.2,
            "system_uptime": 0.997,
            "renewal_intent": 0.91,
            "roi_months": 3.2
        },
        action_category="Customer Pilot",
        stakeholders_involved=["SEMA Tech Team", "Trasacco Estates Management", "Residents", "Security Personnel"],
        budget_spent=8500.0,
        duration_days=90
    ))

    # Scenario 2: Technical Integration Failure (FAILED)
    actions.append(CompletedAction(
        title="CCTV Integration Testing with Local Security Companies",
        description="Attempted to integrate SEMA's AI algorithms with existing CCTV systems used by 5 major Ghanaian security companies. Goal was to demonstrate plug-and-play compatibility to expand market reach beyond direct installations.",
        outcome=ActionOutcome.FAILED,
        results_data="""
        INTEGRATION TESTING FAILURE ANALYSIS:
        
        === TECHNICAL COMPATIBILITY ISSUES ===
        • Successfully integrated: Only 2 out of 5 security company systems (40%)
        • Major incompatibility: 3 companies using proprietary Chinese camera protocols
        • Video format conflicts: 60% of existing cameras output in incompatible formats
        • Network architecture barriers: Legacy DVR systems cannot support cloud integration
        • Bandwidth limitations: Existing infrastructure insufficient for real-time AI processing
        
        === SPECIFIC INTEGRATION FAILURES ===
        • SecureGuard Ghana: Cameras too old (2015 models) for AI integration
        • Protector Services: Proprietary video management system blocks third-party access
        • Ghana Security Solutions: Network security policies prevent cloud data transmission
        • SafeHome Ltd: Successfully integrated but performance degraded by 45%
        • Elite Protection: Integration successful with minor modifications needed
        
        === ROOT CAUSE ANALYSIS ===
        • Underestimated diversity of existing security infrastructure in Ghana
        • Legacy systems dominate market (78% of installations over 5 years old)
        • Security companies resistant to cloud-based solutions due to privacy concerns
        • Lack of standardized protocols in Ghanaian security industry
        • Our system designed for modern IP cameras, not analog CCTV systems
        
        === BUSINESS IMPACT ===
        • Partnership expansion strategy significantly delayed by 6-8 months
        • Additional development costs: $15,000 for compatibility layer development
        • Market size reduced: 65% of potential B2B customers now incompatible
        • Customer acquisition timeline extended: Must focus on new installations only
        • Competitive positioning weakened against companies with legacy support
        
        === CUSTOMER FEEDBACK ===
        • "System works great but we can't afford to replace all our cameras"
        • "Too complex for our current technical capabilities"
        • "Concerns about data security with cloud-based AI processing"
        • "Would need 6-month ROI guarantee to justify infrastructure upgrade"
        
        === STRATEGIC IMPLICATIONS ===
        • Need to pivot from retrofit market to new installation market
        • Requires partnership with camera manufacturers for integrated solutions
        • Must develop offline AI processing capabilities for security-sensitive clients
        • Market entry strategy needs significant revision
        """,
        completion_date=datetime.now() - timedelta(days=5),
        success_metrics={
            "integration_success_rate": 0.40,
            "compatible_systems": 2,
            "market_accessibility": 0.35,
            "additional_dev_cost": 15000,
            "timeline_delay_months": 7
        },
        action_category="Technical Integration",
        stakeholders_involved=["SEMA Engineering Team", "5 Security Companies", "System Integrators"],
        budget_spent=12000.0,
        duration_days=45
    ))

    # Scenario 3: Strategic Partnership Negotiations (INCONCLUSIVE)
    actions.append(CompletedAction(
        title="Ghana Police Service Partnership Discussions for Smart City Initiative",
        description="Engaged in 4-month negotiations with Ghana Police Service to integrate SEMA's predictive surveillance technology into their Smart City crime prevention initiative. Discussions involved multiple government departments and international development partners.",
        outcome=ActionOutcome.INCONCLUSIVE,
        results_data="""
        PARTNERSHIP NEGOTIATION STATUS - COMPLEX MIXED SIGNALS:
        
        === POSITIVE INDICATORS ===
        • Strong technical endorsement from GPS Technology Division
        • Successful proof-of-concept demonstration at Police Headquarters
        • Written support from 3 Regional Police Commanders
        • Integration approved by National Security technical committee
        • World Bank Smart Cities fund shows interest in co-financing
        • Minister of Interior expressed public support at tech conference
        
        === SIGNIFICANT CHALLENGES ===
        • Procurement process requires 18-month government tender process
        • Data privacy concerns raised by Attorney General's office
        • Budget allocation pending National Assembly approval (uncertain timeline)
        • Competition from 2 international security firms with government relationships
        • Regulatory framework for AI surveillance still under development
        • Change in Police Service leadership during negotiation period
        
        === CURRENT NEGOTIATION STATUS ===
        • Technical requirements: 95% agreement reached
        • Commercial terms: 60% consensus (pricing still under discussion)
        • Legal framework: 40% complete (data governance unresolved)
        • Implementation timeline: Disputed (GPS wants 6 months, we need 12)
        • Pilot scope: Agreement on 2 Accra suburbs for initial deployment
        
        === STAKEHOLDER POSITIONS ===
        • Ghana Police Service: Enthusiastic but constrained by procurement rules
        • Ministry of Interior: Supportive but cautious about budget implications
        • World Bank: Interested in funding but requires competitive bidding
        • Local competitors: Actively lobbying against foreign AI technology
        • Civil society groups: Raising privacy and surveillance concerns
        
        === FINANCIAL IMPLICATIONS ===
        • Potential contract value: $2.4M over 3 years (transformational for SEMA)
        • Required investment for compliance: $180K in legal and regulatory costs
        • Opportunity cost: 6 months of business development focus
        • Risk assessment: 45% probability of successful partnership
        • Alternative strategy cost: $50K to pivot to private sector focus
        
        === NEXT STEPS UNCERTAINTY ===
        • Awaiting cabinet-level decision on Smart Cities budget allocation
        • Legal review of data sharing agreements ongoing
        • Technical pilot approval stuck in bureaucratic approval chain
        • International development partner engagement required
        • Political election cycle may reset government priorities
        
        === STRATEGIC DECISION REQUIRED ===
        • Continue government partnership pursuit vs focus on private market
        • Resource allocation for long-term regulatory compliance
        • Risk tolerance for extended negotiation timeline
        • Alternative market entry strategies if partnership fails
        """,
        completion_date=datetime.now() - timedelta(days=8),
        success_metrics={
            "negotiation_progress": 0.65,
            "technical_approval": 0.95,
            "commercial_agreement": 0.60,
            "timeline_uncertainty_months": 18,
            "success_probability": 0.45,
            "potential_contract_value": 2400000
        },
        action_category="Strategic Partnership",
        stakeholders_involved=["Ghana Police Service", "Ministry of Interior", "World Bank", "Legal Team"],
        budget_spent=25000.0,
        duration_days=120
    ))

    # Scenario 4: Market Expansion Research (SUCCESSFUL)
    actions.append(CompletedAction(
        title="West African Market Expansion Feasibility Study",
        description="Comprehensive market research across Nigeria, Ivory Coast, and Senegal to assess expansion opportunities for predictive surveillance technology. Included regulatory analysis, customer interviews, and competitive landscape assessment.",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        WEST AFRICAN EXPANSION RESEARCH - STRONG MARKET OPPORTUNITY:
        
        === MARKET SIZE ANALYSIS ===
        • Nigeria: 2.3M high-income households, 45% with existing security systems
        • Ivory Coast: 180K target households, 23% security system penetration
        • Senegal: 95K potential customers, 31% current security adoption
        • Combined market opportunity: $67M annually across 3 countries
        • Growth rate: 23% annually in security technology adoption
        
        === CUSTOMER VALIDATION INSIGHTS ===
        • 340 customer interviews across 3 countries completed
        • Willingness to pay: 78% would pay premium for predictive security
        • Price sensitivity: Optimal pricing $8-12/month (20% higher than Ghana)
        • Feature preferences: Mobile alerts (94%), dashboard analytics (67%)
        • Trust factors: Local partnerships essential for credibility
        
        === REGULATORY LANDSCAPE ===
        • Nigeria: Favorable AI technology policies, streamlined business registration
        • Ivory Coast: Government digitization initiative supports security tech
        • Senegal: Strong data protection laws require careful compliance
        • All countries: Need local partnerships for government sector access
        • Timeline: 6-8 months for regulatory approval and business setup
        
        === COMPETITIVE ANALYSIS ===
        • Limited predictive surveillance competitors in target markets
        • Traditional security companies dominate but with reactive solutions
        • International firms present but focused on enterprise, not residential
        • Opportunity for first-mover advantage in predictive technology
        • Local technical talent available for country-specific development
        
        === BUSINESS MODEL VALIDATION ===
        • SaaS subscription model widely accepted across all 3 markets
        • Local payment preferences: Mobile money (Nigeria), bank transfers (others)
        • Customer acquisition: Referrals most effective (67% conversion)
        • Market entry: Franchise/partnership model preferred by customers
        • Break-even timeline: 18 months per country with proper capitalization
        
        === STRATEGIC RECOMMENDATIONS ===
        • Nigeria highest priority: Largest market, best infrastructure
        • Phased expansion: Nigeria first (12 months), Ivory Coast (18 months), Senegal (24 months)
        • Required investment: $450K for Nigeria market entry
        • Key partnerships identified: 3 security companies per country
        • Revenue projection: $1.2M by Year 2 across all markets
        """,
        completion_date=datetime.now() - timedelta(days=12),
        success_metrics={
            "market_size_usd": 67000000,
            "customer_interviews": 340,
            "willingness_to_pay": 0.78,
            "market_growth_rate": 0.23,
            "break_even_months": 18,
            "required_investment": 450000
        },
        action_category="Market Research",
        stakeholders_involved=["Market Research Team", "Local Partners", "Regulatory Consultants"],
        budget_spent=18000.0,
        duration_days=75
    ))

    # Scenario 5: Product Development Challenge (FAILED)
    actions.append(CompletedAction(
        title="AI Model Performance Optimization for Low-Light Conditions",
        description="Invested 4 months in improving SEMA's AI algorithms for better threat detection in low-light and nighttime conditions, which account for 70% of security incidents in Ghana.",
        outcome=ActionOutcome.FAILED,
        results_data="""
        AI OPTIMIZATION PROJECT - SIGNIFICANT TECHNICAL SETBACK:
        
        === PERFORMANCE TARGETS VS RESULTS ===
        • Target: 85% accuracy in low-light conditions
        • Achieved: 52% accuracy (barely above baseline 45%)
        • Nighttime detection: 34% accuracy (far below 75% target)
        • Processing speed: 40% slower than original model
        • False positive rate: Increased to 28% (from 12% baseline)
        
        === TECHNICAL CHALLENGES ENCOUNTERED ===
        • Infrared camera integration more complex than anticipated
        • Training data insufficient: Only 2,000 low-light incident samples
        • Model overfitting to specific lighting conditions
        • Hardware requirements increased 3x for real-time processing
        • Existing cloud infrastructure insufficient for new model demands
        
        === DEVELOPMENT RESOURCE IMPACT ===
        • 2 senior ML engineers dedicated full-time for 4 months
        • Development costs: $45,000 in additional compute resources
        • Timeline delay: 6 months behind product roadmap
        • Technical debt: Need to refactor core detection algorithms
        • Customer impact: 3 pilot deployments postponed indefinitely
        
        === ROOT CAUSE ANALYSIS ===
        • Underestimated complexity of low-light computer vision
        • Insufficient investment in specialized training data collection
        • Wrong algorithm approach: Tried to modify existing model vs building specialized one
        • Lack of hardware optimization for edge computing requirements
        • Team expertise gap in specialized computer vision techniques
        
        === BUSINESS CONSEQUENCES ===
        • Competitive disadvantage: Competitors launched similar features successfully
        • Customer complaints: 23% of users report poor nighttime performance
        • Sales impact: Lost 2 major contracts due to performance limitations
        • Investment loss: $75K in development with no usable outcome
        • Market positioning: Cannot claim "24/7 security coverage"
        
        === CUSTOMER FEEDBACK ===
        • "System is great during day but useless at night when we need it most"
        • "Too many false alarms at night - had to disable notifications"
        • "Competitors offer better night vision capabilities"
        • "Would switch providers if nighttime performance doesn't improve"
        
        === STRATEGIC IMPLICATIONS ===
        • Need to partner with computer vision specialists
        • Consider hardware-software co-development approach
        • May require significant additional funding for proper solution
        • Timeline for competitive feature parity extended to 12+ months
        • Product positioning must acknowledge current limitations
        """,
        completion_date=datetime.now() - timedelta(days=15),
        success_metrics={
            "accuracy_achieved": 0.52,
            "target_accuracy": 0.85,
            "performance_gap": -0.33,
            "development_cost": 75000,
            "timeline_delay_months": 6,
            "customer_complaints": 23
        },
        action_category="Product Development",
        stakeholders_involved=["ML Engineering Team", "Product Management", "Customer Success"],
        budget_spent=75000.0,
        duration_days=120
    ))

    return actions


def get_sample_action_by_title(title: str) -> CompletedAction:
    """Get a specific sample action by its title."""
    actions = get_sample_completed_actions()
    action_dict = {action.title: action for action in actions}
    return action_dict.get(title, actions[0])  # Default to first action if not found


def get_action_titles() -> List[str]:
    """Get list of all sample action titles for UI dropdowns."""
    return [action.title for action in get_sample_completed_actions()]


def get_sema_market_context() -> Dict:
    """Return SEMA-specific market context for enhanced agent intelligence."""
    return {
        "company_profile": {
            "name": "SEMA",
            "industry": "Security Technology / PropTech",
            "stage": "Early Growth Phase",
            "location": "Accra, Ghana",
            "founding_year": 2022,
            "team_size": 12,
            "funding_stage": "Seed/Pre-Series A"
        },
        "market_context": {
            "primary_market": "Ghana (Urban Areas)",
            "expansion_targets": ["Nigeria", "Ivory Coast", "Senegal"],
            "market_size": "67M USD (West Africa)",
            "growth_rate": "23% annually",
            "penetration": "15% in target segments"
        },
        "competitive_landscape": {
            "direct_competitors": ["Traditional Security Companies", "International Security Tech"],
            "competitive_advantage": "AI-powered predictive vs reactive security",
            "market_position": "Innovation leader, early-stage revenue"
        },
        "technology_focus": [
            "Computer Vision & AI",
            "Predictive Analytics", 
            "Real-time Video Processing",
            "Mobile & Dashboard Interfaces",
            "Cloud Infrastructure"
        ],
        "target_customers": [
            "Gated Communities",
            "High-Net-Worth Individuals", 
            "Property Management Companies",
            "Corporate Facilities",
            "Government Institutions"
        ],
        "key_metrics": {
            "monthly_recurring_revenue": 45000,
            "customer_acquisition_cost": 180,
            "customer_lifetime_value": 2400,
            "churn_rate": 0.08,
            "system_uptime": 0.997
        },
        "challenges": [
            "Legacy CCTV system integration",
            "Data privacy and security concerns",
            "Customer education on predictive security",
            "Regulatory compliance across markets",
            "Technical talent acquisition"
        ],
        "opportunities": [
            "West African market expansion",
            "Government Smart City initiatives", 
            "Corporate security market penetration",
            "Partnership with property developers",
            "International development funding"
        ]
    }


def get_business_intelligence_scenarios() -> List[Dict]:
    """Return scenarios designed to showcase sophisticated agentic intelligence."""
    return [
        {
            "scenario_name": "Customer Pilot Success Analysis",
            "complexity_level": "High", 
            "key_insight": "Exceptional pilot results require scaling strategy and resource allocation",
            "cross_bmc_impact": ["customer_segments", "value_propositions", "revenue_streams", "key_activities"],
            "confidence_factors": ["Strong quantitative results", "High customer satisfaction", "Clear business metrics"],
            "strategic_implications": "Accelerated growth strategy with focus on similar customer segments"
        },
        {
            "scenario_name": "Technical Integration Failure Recovery",
            "complexity_level": "Very High",
            "key_insight": "Legacy system incompatibility requires fundamental strategy pivot",
            "cross_bmc_impact": ["customer_segments", "key_partnerships", "channels", "cost_structure"],
            "confidence_factors": ["Clear failure data", "Root cause analysis", "Market impact assessment"],
            "strategic_implications": "Pivot from retrofit to new installation market strategy"
        },
        {
            "scenario_name": "Government Partnership Navigation",
            "complexity_level": "High",
            "key_insight": "Complex stakeholder dynamics require long-term strategic patience",
            "cross_bmc_impact": ["key_partnerships", "customer_segments", "revenue_streams"],
            "confidence_factors": ["Mixed signals", "High complexity", "Political uncertainties"],
            "strategic_implications": "Balanced approach between government and private sector focus"
        },
        {
            "scenario_name": "Market Expansion Validation",
            "complexity_level": "Medium",
            "key_insight": "Strong regional expansion opportunity with clear implementation roadmap",
            "cross_bmc_impact": ["customer_segments", "channels", "key_partnerships", "revenue_streams"],
            "confidence_factors": ["Comprehensive research", "Customer validation", "Market data"],
            "strategic_implications": "Phased expansion strategy starting with Nigeria"
        },
        {
            "scenario_name": "Product Development Setback Management",
            "complexity_level": "High",
            "key_insight": "Technical limitations require strategic pivots and resource reallocation",
            "cross_bmc_impact": ["value_propositions", "key_resources", "key_activities", "cost_structure"],
            "confidence_factors": ["Clear failure metrics", "Customer feedback", "Technical analysis"],
            "strategic_implications": "Need for specialized partnerships and adjusted market positioning"
        }
    ]


def get_confidence_calibration_examples() -> List[Dict]:
    """Return examples for confidence score calibration in SEMA context."""
    return [
        {
            "scenario": "Trasacco Estates pilot with 89% accuracy and 9.2/10 satisfaction",
            "confidence_range": "0.90-0.95",
            "reasoning": "Exceptional quantitative results with strong customer validation"
        },
        {
            "scenario": "CCTV integration failure with clear technical analysis",
            "confidence_range": "0.85-0.92", 
            "reasoning": "Clear failure data provides high confidence in needed strategic changes"
        },
        {
            "scenario": "Government partnership with mixed signals and uncertainties",
            "confidence_range": "0.45-0.65",
            "reasoning": "Political complexity and regulatory uncertainty limit confidence"
        },
        {
            "scenario": "Market research with 340 interviews across 3 countries",
            "confidence_range": "0.82-0.88",
            "reasoning": "Strong primary research but execution risks remain"
        },
        {
            "scenario": "Product development failure with clear performance metrics",
            "confidence_range": "0.88-0.94",
            "reasoning": "Technical failure data provides high confidence in strategic implications"
        }
    ]


# Export key functions for clean imports
__all__ = [
    'get_sample_business_model_canvas',
    'get_sample_completed_actions', 
    'get_sample_action_by_title',
    'get_action_titles',
    'get_sema_market_context',
    'get_business_intelligence_scenarios',
    'get_confidence_calibration_examples'
]