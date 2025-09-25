"""
Enhanced mock data for the Agentic AI Actions Co-pilot system.

This module provides sophisticated sample data representing emerging market fintech 
startups with realistic business model canvas and completed action scenarios designed 
to showcase advanced agentic intelligence capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List

from business_models import (
    BusinessModelCanvas,
    ValuePropositionCanvas,
    CompletedAction,
    ActionOutcome
)


def get_sample_business_model_canvas() -> BusinessModelCanvas:
    """Return an enhanced BMC for an emerging market fintech startup."""
    return BusinessModelCanvas(
        customer_segments=[
            "Informal traders in major markets (Makola, Kejetia, Tamale Central)",
            "Rural smallholder farmers in Northern and Upper regions", 
            "Unbanked youth aged 18-35 in peri-urban areas",
            "Small-scale entrepreneurs and micro-businesses (1-10 employees)",
            "Cross-border traders between Ghana, Burkina Faso, and Togo",
            "Women-led businesses in informal economy sectors",
            "Artisans and craftspeople in traditional sectors"
        ],
        value_propositions=[
            "USSD-based mobile money accessible on basic feature phones",
            "Transaction fees 50% lower than traditional banking services",
            "Multi-language support (Twi, Hausa, Ewe, Ga, French)",
            "24/7 agent network for cash-in/cash-out in rural areas",
            "Instant peer-to-peer transfers across West African countries", 
            "Integration with traditional savings groups (susu/tontines)",
            "Micro-credit scoring based on transaction history",
            "Offline transaction capability for areas with poor connectivity"
        ],
        channels=[
            "Mobile network operator partnerships (MTN, Vodafone Ghana, AirtelTigo)",
            "Community agent network in markets and rural villages",
            "USSD shortcode (*712*456#) accessible on all phone types",
            "Word-of-mouth referrals through trusted community leaders",
            "Local radio advertisements in native languages",
            "SMS marketing campaigns with transaction insights",
            "WhatsApp Business integration for customer support",
            "Partnership distribution through microfinance institutions"
        ],
        customer_relationships=[
            "Personal assistance through trained local agents",
            "Community-based financial literacy workshops",
            "24/7 multilingual customer service via USSD and voice",
            "Trust-building through local religious and community partnerships",
            "Loyalty programs with cashback for frequent users",
            "Peer-to-peer referral incentive programs",
            "Women entrepreneur support groups and training"
        ],
        revenue_streams=[
            "Transaction fees (2% on transfers, 1.5% on bill payments)",
            "Agent commission structure (35% of transaction fees)",
            "Bill payment service charges (utilities, schools, healthcare)",
            "Cross-border transfer fees (3% with currency conversion)",
            "Interest income on customer float funds",
            "Premium service subscriptions for business accounts",
            "Micro-loan origination fees and interest",
            "Data monetization through anonymized transaction insights"
        ],
        key_resources=[
            "USSD platform technology and telecom integration infrastructure",
            "Extensive agent network across Ghana (1,200+ agents)",
            "Payment service provider license from Bank of Ghana",
            "Customer database with transaction history (500K+ users)",
            "Multilingual customer support team and training materials",
            "Strategic partnerships with telcos, banks, and MFIs",
            "Proprietary credit scoring algorithms and risk models",
            "Brand reputation and community trust built over 3 years"
        ],
        key_activities=[
            "Mobile platform development, maintenance, and security",
            "Agent recruitment, training, and performance management",
            "Regulatory compliance reporting and risk management",
            "Customer acquisition through community outreach programs",
            "Fraud detection and prevention system operations",
            "Financial literacy education and community engagement",
            "API integration with banks, utilities, and service providers",
            "Data analytics for business intelligence and credit decisions"
        ],
        key_partnerships=[
            "Rural and community banks (ARB Apex Bank network)",
            "Mobile network operators (MTN Mobile Money, Vodafone Cash)",
            "Central bank partnerships (Bank of Ghana regulatory sandbox)",
            "Local market associations and trader cooperatives",
            "International remittance providers (Western Union, MoneyGram)",
            "Technology infrastructure providers (Huawei, Ericsson)",
            "Microfinance institutions for last-mile distribution",
            "Government agencies for digital ID and KYC services"
        ],
        cost_structure=[
            "Technology infrastructure and platform maintenance (35%)",
            "Agent network operations and commissions (25%)",
            "Regulatory compliance and licensing costs (10%)",
            "Customer acquisition and community marketing (15%)",
            "Staff salaries and operational overhead (10%)",
            "Fraud prevention and security measures (3%)",
            "Third-party integrations and API costs (2%)"
        ]
    )


def get_sample_value_proposition_canvas() -> ValuePropositionCanvas:
    """Return an enhanced VPC aligned with the emerging market fintech BMC."""
    return ValuePropositionCanvas(
        jobs_to_be_done=[
            "Send money to family members in rural areas quickly and securely",
            "Pay for goods and services without carrying cash",
            "Save money securely without traditional bank account requirements",
            "Receive payments from customers instantly during business hours",
            "Access credit for inventory and business expansion needs",
            "Pay utility bills and school fees from remote locations",
            "Transfer money across borders for trade and remittances",
            "Build financial history for future loan applications"
        ],
        pains=[
            "High transaction fees from traditional banks (5-8% on transfers)",
            "Long travel distances to bank branches (average 15km in rural areas)",
            "Complex account opening requirements and documentation",
            "Language barriers with financial services (English-only interfaces)",
            "Security risks of carrying cash in markets and transport",
            "Unreliable internet connectivity for smartphone banking apps",
            "Long queues and limited banking hours (8am-3pm weekdays only)",
            "Lack of credit history and financial identity"
        ],
        gains=[
            "Quick and convenient money transfers completed in under 2 minutes",
            "Significantly lower costs than traditional banking (50% savings)",
            "Financial inclusion for previously unbanked populations",
            "Enhanced security through digital transactions and PIN protection",
            "Access to credit and savings products based on transaction history",
            "Integration with existing business and social practices",
            "24/7 availability without dependency on banking hours",
            "Building financial identity and creditworthiness over time"
        ],
        products_services=[
            "USSD mobile money platform (*712*456#)",
            "Agent-assisted cash-in/cash-out services",
            "Peer-to-peer money transfers within and across borders",
            "Bill payment services for utilities, schools, and healthcare",
            "Micro-savings accounts with competitive interest rates",
            "Merchant payment solutions for small businesses",
            "Micro-credit products based on transaction history",
            "Financial literacy education and business training"
        ],
        pain_relievers=[
            "Transaction fees 50-60% lower than traditional banks",
            "USSD functionality works on basic phones without internet",
            "Local agents eliminate need to travel to bank branches",
            "Multi-language customer support in 5 local languages", 
            "Secure digital alternative to risky cash transactions",
            "Simplified registration requiring only phone number and ID",
            "24/7 service availability through USSD and agent network",
            "Gradual credit building through consistent transaction history"
        ],
        gain_creators=[
            "Instant money transfers completed within 30 seconds",
            "Complete financial freedom with 24/7/365 availability",
            "Financial empowerment for previously excluded populations",
            "Seamless integration with traditional community savings (susu)",
            "Loyalty rewards and cashback for frequent platform usage",
            "Business growth acceleration through digital payment acceptance",
            "Cross-border trade facilitation with favorable exchange rates",
            "Pathway to formal financial services through transaction history"
        ]
    )


def get_sample_completed_actions() -> List[CompletedAction]:
    """Return sophisticated completed actions representing realistic business scenarios 
    designed to showcase advanced agentic intelligence."""

    actions = []

    # Scenario 1: Complex Multi-Stakeholder Market Research (SUCCESSFUL)
    actions.append(CompletedAction(
        title="Comprehensive Market Penetration Analysis - Northern Region",
        description="Conducted extensive 3-month market research across 12 Northern Region districts, including quantitative surveys (n=2,847), qualitative focus groups (24 sessions), competitive analysis, and regulatory landscape assessment to evaluate expansion opportunities",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        COMPREHENSIVE MARKET ANALYSIS RESULTS:
        
        === QUANTITATIVE FINDINGS ===
        • Total survey respondents: 2,847 across 12 districts
        • Market penetration opportunity: 340,000 potential users
        • Current mobile money usage: 23% (significantly below national 39%)
        • Willingness to switch providers: 67% citing better rates and local language support
        
        === CUSTOMER SEGMENTATION INSIGHTS ===
        • Rural farmers (43%): Seasonal income, need credit during planting season
        • Cross-border traders (28%): Require multi-currency support, higher transaction limits  
        • Young entrepreneurs (18%): Digital-native, want smartphone integration
        • Traditional merchants (11%): Trust-focused, prefer agent relationships
        
        === COMPETITIVE LANDSCAPE ===
        • MTN Mobile Money: 45% market share, limited agent network in rural areas
        • Vodafone Cash: 32% market share, poor local language support
        • AirtelTigo Money: 23% market share, inconsistent service quality
        • Opportunity gap: Premium service tier for business users
        
        === REGULATORY & INFRASTRUCTURE ===
        • Bank of Ghana approval required for Northern Region expansion
        • Network coverage: 89% 2G/3G, 34% 4G in target areas  
        • Agent network gaps: 67% of communities lack financial service agents
        • Local partnership opportunities: 34 community banks, 12 MFIs
        
        === FINANCIAL PROJECTIONS ===
        • Projected customer acquisition: 85,000 in Year 1, 230,000 in Year 3
        • Revenue opportunity: GHS 12.4M annually by Year 3
        • Required investment: GHS 8.7M for infrastructure and agent network
        • Break-even timeline: 18 months with 65,000 active users
        
        === KEY SUCCESS FACTORS ===
        • Local language USSD interfaces (Dagbani, Mampruli, Gonja)
        • Community-based agent network with cultural understanding
        • Integration with agricultural value chain payment systems
        • Flexible KYC requirements adapted to rural documentation challenges
        """,
        completion_date=datetime.now() - timedelta(days=1),
        success_metrics={
            "sample_size": 2847,
            "geographic_coverage": 12,
            "data_quality_score": 0.94,
            "confidence_interval": 0.95,
            "actionable_insights": 23,
            "stakeholder_interviews": 67,
            "competitive_analysis_depth": 0.87
        }
    ))

    # Scenario 2: Product Feature Usability Crisis (FAILED) 
    actions.append(CompletedAction(
        title="Advanced Financial Dashboard Beta Testing Program",
        description="Launched beta testing program for new advanced financial dashboard targeting business users, including analytics, cash flow forecasting, and automated bookkeeping features. Tested with 150 SME customers over 6 weeks",
        outcome=ActionOutcome.FAILED,
        results_data="""
        BETA TESTING PROGRAM - CRITICAL FAILURE ANALYSIS:
        
        === PARTICIPATION & ENGAGEMENT ===
        • Beta testers recruited: 150 SME customers
        • Completed full testing cycle: 47 (31% completion rate)
        • Dropped out due to complexity: 68 users (45%)
        • Requested to revert to basic interface: 89% of active testers
        
        === USABILITY DISASTER METRICS ===
        • Task completion rate: 23% (target was 85%)
        • Average time to complete basic tasks: 12.4 minutes (target: 3 minutes)
        • Error rate: 47% (unacceptably high, target <5%)
        • Customer satisfaction score: 2.1/10 (catastrophic)
        • Support tickets generated: 342 in 6 weeks (previous 6 months: 89)
        
        === SPECIFIC FAILURE POINTS ===
        • Cash flow forecasting: 89% couldn't understand the charts
        • Automated categorization: 76% found it inaccurate and confusing
        • Export functionality: 34% couldn't find it, 45% couldn't use it
        • Multi-currency support: Completely broken for CFA/Naira transactions
        • Mobile responsiveness: 67% said mobile interface unusable
        
        === ROOT CAUSE ANALYSIS ===
        • UI/UX designed for Western SMEs, not emerging market context
        • No user research conducted with actual target demographic
        • Feature complexity exceeded customers' digital literacy levels
        • Failed to integrate with existing business workflow patterns
        • Inadequate testing with low-end Android devices (78% of user base)
        
        === CUSTOMER FEEDBACK THEMES ===
        • "Too complicated, I just want to send money" (recurring theme)
        • "The old version was perfect, why change it?"
        • "Crashed my phone twice, had to restart"
        • "Cannot understand the English technical words"
        • "Takes too long to load on my network"
        
        === BUSINESS IMPACT ===
        • Customer churn: 23 beta testers cancelled accounts
        • Brand reputation: NPS dropped from +45 to +12 among SME segment
        • Support costs: 340% increase in customer service tickets
        • Development costs: GHS 847,000 investment with zero ROI
        • Competitive damage: MTN gained 12% market share during beta period
        
        === LESSONS LEARNED ===
        • Feature sophistication must match customer digital maturity
        • Extensive user research essential before development
        • Mobile-first design critical for emerging market success
        • Gradual feature introduction better than comprehensive overhauls
        • Local context and language absolutely critical
        """,
        completion_date=datetime.now() - timedelta(days=3),
        success_metrics={
            "completion_rate": 0.31,
            "user_satisfaction": 2.1,
            "task_success_rate": 0.23,
            "error_rate": 0.47,
            "support_ticket_increase": 3.84,
            "churn_rate": 0.15,
            "nps_impact": -33
        }
    ))

    # Scenario 3: Strategic Partnership with Mixed Signals (INCONCLUSIVE)
    actions.append(CompletedAction(
        title="Multi-Country West African Expansion Partnership Assessment",
        description="6-month evaluation of strategic partnership opportunities with Orange Money (Senegal/Mali) and Wave (Senegal) to enable cross-border interoperability and market expansion. Included technical integration testing, regulatory compliance analysis, and pilot transaction programs",
        outcome=ActionOutcome.INCONCLUSIVE,
        results_data="""
        WEST AFRICAN PARTNERSHIP ASSESSMENT - MIXED RESULTS:
        
        === PARTNERSHIP OPPORTUNITY ANALYSIS ===
        Orange Money Partnership:
        • Technical integration: 87% successful (minor API compatibility issues)
        • Regulatory approval: Pending in Mali (4+ months), approved in Senegal
        • Market size opportunity: 2.3M potential users across both markets
        • Revenue sharing terms: 65/35 (favorable to us)
        • Cultural fit: Strong alignment with community-based approach
        
        Wave Partnership:
        • Technical integration: 45% successful (major protocol incompatibilities) 
        • Regulatory approval: Fast-tracked in Senegal, blocked in Ghana by BoG
        • Market size opportunity: 890K users but high overlap with Orange
        • Revenue sharing terms: 50/50 (less favorable)
        • Cultural fit: Aggressive growth strategy conflicts with our approach
        
        === PILOT PROGRAM RESULTS ===
        Cross-border Transaction Pilot (90 days):
        • Transaction volume: GHS 234,000 processed
        • Success rate: 78% (target was 95%)
        • Average transaction time: 4.2 minutes (target: <2 minutes)
        • Customer satisfaction: 6.7/10 (mixed feedback)
        • Technical issues: 89 reported, 67 resolved
        
        === POSITIVE INDICATORS ===
        • Strong customer demand: 2,300 pre-registrations for cross-border service
        • Cost advantages: 40% cheaper than Western Union for Ghana-Senegal corridor
        • Regulatory support: Bank of Ghana encouraged expansion initiative
        • Market positioning: First mover advantage in Ghana-West Africa corridor
        • Brand strength: High recognition in target markets (67% brand recall)
        
        === CONCERNING SIGNALS ===
        • Technical complexity: Integration costs 340% higher than projected
        • Competitive response: MTN Mobile Money announced similar partnership
        • Regulatory uncertainty: New ECOWAS payment regulations pending
        • Operational challenges: Customer service in 3 languages, 2 currencies
        • Cash-out network: Insufficient agent coverage in Mali rural areas
        
        === FINANCIAL IMPLICATIONS ===
        Investment Required:
        • Technical infrastructure: GHS 2.8M (originally budgeted GHS 800K)
        • Regulatory compliance: GHS 450K across 3 countries
        • Marketing and customer acquisition: GHS 1.2M
        • Operational setup (agents, support): GHS 950K
        • Total: GHS 5.4M (127% over original budget)
        
        Revenue Projections (3-year):
        • Conservative scenario: GHS 8.9M (65% IRR)
        • Optimistic scenario: GHS 16.7M (142% IRR)
        • Risk scenario: GHS 2.1M (-31% IRR)
        
        === STRATEGIC CONSIDERATIONS ===
        • Market timing: 18-month window before competitors establish dominance
        • Resource allocation: Would require 60% of dev team for 12+ months
        • Risk tolerance: High-reward opportunity with significant execution risks
        • Alternative strategies: Focus on domestic market depth vs. regional breadth
        • Stakeholder alignment: Board split 4-3 on proceeding
        
        RECOMMENDATION: Requires deeper technical due diligence and phased approach
        """,
        completion_date=datetime.now() - timedelta(days=5),
        success_metrics={
            "partnership_agreements": 2,
            "technical_integration_success": 0.66,
            "regulatory_approval_rate": 0.67,
            "pilot_transaction_success": 0.78,
            "customer_satisfaction": 6.7,
            "roi_range_variance": 1.73,
            "budget_variance": 1.27,
            "execution_complexity_score": 0.83
        }
    ))

    # Scenario 4: Regulatory Compliance Achievement (SUCCESSFUL)
    actions.append(CompletedAction(
        title="Bank of Ghana Regulatory Sandbox Graduation and Full License Acquisition",
        description="Successful completion of 18-month Bank of Ghana Regulatory Sandbox program and acquisition of full Payment Service Provider (PSP) license, including comprehensive compliance framework implementation, risk management system deployment, and regulatory audit passage",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        REGULATORY MILESTONE ACHIEVEMENT - FULL COMPLIANCE SUCCESS:
        
        === SANDBOX GRADUATION METRICS ===
        • Sandbox entry date: March 2023
        • Graduation date: September 2024 (6 weeks ahead of schedule)
        • Compliance score: 94/100 (highest in cohort)
        • Risk assessment rating: A- (second highest possible)
        • Customer protection score: 96/100
        • Innovation impact score: 87/100
        
        === FULL PSP LICENSE ACQUISITION ===
        • Application submitted: July 2024
        • License granted: September 2024 (industry average: 12-18 months)
        • License validity: 5 years with annual renewal requirements
        • Authorized services: Payments, transfers, e-money issuance, agent banking
        • Transaction limits: Up to GHS 10,000 per transaction, GHS 50,000 monthly
        • Cross-border authorization: Pending separate application (Q1 2025)
        
        === COMPLIANCE FRAMEWORK IMPLEMENTATION ===
        Anti-Money Laundering (AML):
        • KYC procedures: 3-tier system implemented and tested
        • Transaction monitoring: AI-powered system detecting 97.3% of suspicious patterns
        • Suspicious transaction reports: 47 filed, 100% acknowledged by FIC
        • Staff training: 100% completion rate on AML certification
        
        Data Protection & Privacy:
        • GDPR-equivalent framework implemented
        • Customer data encryption: AES-256 standard
        • Data breach response: <4 hour notification protocol
        • Privacy impact assessments: Completed for all new features
        
        Financial Risk Management:
        • Capital adequacy ratio: 18.7% (requirement: 10%)
        • Liquidity coverage: 145% (requirement: 100%)
        • Operational risk framework: Comprehensive incident management
        • Business continuity: 99.97% uptime achieved
        
        === AUDIT RESULTS ===
        External Audit (PwC Ghana):
        • Financial controls: No material weaknesses identified
        • Operational processes: 3 minor recommendations (all implemented)
        • Technology security: Exceeded industry benchmarks
        • Regulatory compliance: Full compliance certification
        
        Bank of Ghana Examination:
        • Overall rating: Satisfactory (highest possible)
        • Capital adequacy: Well capitalized
        • Management quality: Satisfactory
        • Earnings: Satisfactory with positive trend
        • Liquidity: More than adequate
        
        === BUSINESS IMPACT ===
        Market Credibility:
        • Customer acquisition: 340% increase post-license announcement
        • Partnership inquiries: 23 new institutional partnerships
        • Media coverage: Featured in 12 major publications
        • Investor confidence: Series A funding round oversubscribed by 180%
        
        Operational Capabilities:
        • Transaction limits increased: 5x previous sandbox limits
        • New services enabled: Agent banking, bulk payments, salary disbursements
        • Geographic expansion: Authorized for nationwide operations
        • Customer trust: NPS increased from +34 to +67
        
        === COMPETITIVE ADVANTAGE ===
        • Third fintech in Ghana to achieve full PSP license
        • Fastest sandbox-to-license transition in Bank of Ghana history
        • Regulatory expertise now competitive differentiator
        • Compliance-as-a-service opportunity identified
        
        === NEXT PHASE OPPORTUNITIES ===
        • Cross-border payment license application (Q1 2025)
        • Credit bureau reporting integration
        • Microfinance license evaluation
        • Insurance product distribution partnerships
        • Government payment system integration opportunities
        """,
        completion_date=datetime.now() - timedelta(days=7),
        success_metrics={
            "compliance_score": 0.94,
            "license_approval_speed": 1.67,  # 67% faster than average
            "audit_rating": 1.0,  # Perfect score
            "capital_adequacy_ratio": 1.87,  # 87% above requirement
            "customer_acquisition_boost": 4.4,  # 340% increase
            "nps_improvement": 33,
            "partnership_inquiries": 23,
            "investor_confidence": 2.8  # 180% oversubscription
        }
    ))

    # Scenario 5: Customer Retention Crisis Response (FAILED)
    actions.append(CompletedAction(
        title="Emergency Customer Win-Back Campaign Following Competitive Pressure",
        description="Rapid-response 60-day customer retention campaign launched after MTN Mobile Money's aggressive pricing strategy caused 23% customer churn. Included targeted promotions, personalized outreach, service improvements, and competitive pricing adjustments",
        outcome=ActionOutcome.FAILED,
        results_data="""
        CUSTOMER RETENTION CRISIS RESPONSE - CAMPAIGN FAILURE:
        
        === CRISIS CONTEXT ===
        • MTN Mobile Money launched: Zero-fee transfers for 6 months
        • Customer churn acceleration: 23% in 30 days (normal: 3% monthly)
        • Revenue impact: 34% decline in transaction fee income
        • Market share loss: From 28% to 19% in competitive segment
        • Media coverage: Negative headlines about "fintech struggling"
        
        === CAMPAIGN STRATEGY & EXECUTION ===
        Pricing Response:
        • Reduced transaction fees by 40% (2% to 1.2%)
        • Eliminated fees for transactions under GHS 50
        • Free agent cash-out for transactions >GHS 200
        • Loyalty cashback program: 0.5% on all transactions
        
        Marketing Blitz:
        • Radio advertising: 450 spots across 12 stations
        • SMS campaigns: 2.3M messages sent to customer base
        • Agent incentive programs: Double commissions for customer retention
        • Community outreach: 67 market visits and demonstrations
        
        === CAMPAIGN RESULTS ===
        Customer Metrics:
        • Churn rate during campaign: 18% (target was <5%)
        • New customer acquisition: 4,200 (target: 15,000)
        • Customer reactivation: 890 (target: 8,000)
        • Net customer loss: 38,400 customers
        • Active user engagement: Down 29%
        
        Financial Impact:
        • Campaign cost: GHS 1.47M
        • Revenue recovered: GHS 340K
        • Net campaign ROI: -76%
        • Ongoing revenue impact: -42% due to reduced fees
        • Cash burn acceleration: 340% above sustainable levels
        
        === FAILURE ANALYSIS ===
        Strategic Missteps:
        • Reactive approach: Campaign was defensive, not proactive
        • Price war trap: Unsustainable race to bottom with larger competitor
        • Value proposition confusion: Reduced fees undermined premium positioning
        • Brand messaging: Appeared desperate rather than confident
        • Timing: MTN had first-mover advantage and better execution
        
        Operational Challenges:
        • Agent network confusion: Multiple pricing changes created errors
        • Customer service overload: 340% increase in support tickets
        • System integration issues: Pricing changes caused technical problems
        • Marketing message inconsistency: 3 different campaigns running simultaneously
        
        Competitive Dynamics:
        • MTN's scale advantage: Could sustain losses longer
        • Customer switching costs: Lower than anticipated
        • Brand loyalty: Overestimated customer loyalty to our platform
        • Market maturity: Customers more price-sensitive than expected
        
        === CUSTOMER FEEDBACK ===
        Exit Survey Responses (n=1,247):
        • "MTN is free, why should I pay?" (34% of responses)
        • "Service is the same everywhere" (28% of responses)
        • "Your agents are harder to find" (19% of responses)
        • "App is slower than MTN" (12% of responses)
        • "Friends and family all use MTN" (7% of responses)
        
        === LONG-TERM DAMAGE ASSESSMENT ===
        • Brand perception: "Struggling competitor" narrative established
        • Financial position: 8 months runway reduced to 4.5 months
        • Team morale: 23% turnover in customer-facing roles
        • Investor confidence: Down 45% based on internal surveys
        • Market position: Now considered follower, not leader
        
        === LESSONS LEARNED ===
        • Price competition unsustainable against larger players
        • Customer loyalty requires deeper value than just pricing
        • Defensive strategies rarely succeed in winner-take-all markets
        • Differentiation more valuable than cost leadership
        • Crisis response must be strategic, not just tactical
        
        STRATEGIC PIVOT REQUIRED: Focus on underserved niches where MTN is weak
        """,
        completion_date=datetime.now() - timedelta(days=10),
        success_metrics={
            "churn_reduction_target": 0.28,  # Target was 80% reduction, achieved 28%
            "customer_reactivation": 0.11,  # 11% of target
            "campaign_roi": -0.76,
            "market_share_recovery": 0.0,  # No recovery
            "brand_sentiment_change": -0.31,
            "financial_runway_impact": -0.44,  # 44% reduction in runway
            "team_retention": 0.77
        }
    ))

    # Scenario 6: Innovation Lab Success (SUCCESSFUL)
    actions.append(CompletedAction(
        title="Voice-Based Transaction System for Low-Literacy Users Pilot",
        description="Developed and tested innovative voice-based USSD system allowing customers to conduct transactions using voice commands in local languages. 12-week pilot with 300 low-literacy users across 5 rural communities, focusing on accessibility and financial inclusion",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        VOICE-BASED INNOVATION PILOT - BREAKTHROUGH SUCCESS:
        
        === INNOVATION CONTEXT ===
        • Target demographic: Adults with limited literacy (estimated 2.1M in Ghana)
        • Technology approach: Voice recognition + USSD integration
        • Languages supported: Twi, Hausa, Ga (with 89% accuracy rates)
        • Device compatibility: Works on all phones (no smartphone required)
        • Accessibility focus: Completely audio-driven interface
        
        === PILOT PROGRAM DESIGN ===
        • Participant selection: 300 adults, 67% women, avg age 43
        • Geographic spread: 5 rural communities in Central and Northern regions
        • Literacy levels: 78% primary education or below
        • Training approach: Community-based peer education model
        • Duration: 12 weeks with weekly usage tracking
        
        === OUTSTANDING RESULTS ===
        User Adoption Metrics:
        • Training completion rate: 91% (industry average: 54%)
        • Weekly active usage: 84% by week 12
        • Transaction success rate: 87% (exceeded 75% target)
        • User satisfaction: 9.2/10 (extraordinary)
        • Word-of-mouth referrals: 156 unprompted sign-ups
        
        Transaction Performance:
        • Total transactions: 4,847 over 12 weeks
        • Average transaction time: 2.3 minutes (very competitive)
        • Voice recognition accuracy: 89% (target was 80%)
        • Error resolution rate: 94% self-resolved
        • Failed transaction rate: 4.2% (excellent for pilot)
        
        === BREAKTHROUGH INSIGHTS ===
        User Experience Discoveries:
        • Voice interface more intuitive than expected for target demographic
        • Local language support critical - 67% wouldn't use English-only version
        • Community training model 3x more effective than individual training
        • Women showed 23% higher adoption rates than men
        • Older users (50+) had 91% satisfaction vs 86% for younger users
        
        Financial Behavior Changes:
        • Digital transaction frequency: Increased 340% during pilot
        • Savings behavior: 67% started using mobile savings features
        • Bill payment adoption: 45% began paying school fees digitally
        • Family financial inclusion: 78% taught family members to use system
        • Cash dependency: Reduced by average 52% among participants
        
        === TECHNICAL ACHIEVEMENTS ===
        Voice Recognition Performance:
        • Twi language accuracy: 91% (industry-leading for African languages)
        • Background noise handling: Works in markets, vehicles, outdoor settings
        • Accent adaptation: Self-learning system improved individual accuracy
        • Response time: Average 1.7 seconds from voice to system response
        • Data usage: 67% less than traditional mobile banking apps
        
        Integration Success:
        • USSD backend: Seamless integration with existing infrastructure
        • Agent network: 89% of agents successfully trained on voice support
        • Customer service: Multilingual voice support implemented
        • Security: Voice biometric authentication 94% accurate
        
        === MARKET VALIDATION ===
        Competitive Differentiation:
        • First voice-based mobile money system in West Africa
        • Patent applications filed: 3 (voice USSD integration)
        • Technology licensing interest: 7 international inquiries
        • Academic recognition: Featured in MIT Technology Review
        
        Financial Inclusion Impact:
        • Previously excluded users: 78% had never used digital financial services
        • Women's economic empowerment: 89% reported increased financial autonomy
        • Intergenerational adoption: 45% taught children/grandchildren
        • Community economic impact: 23% increase in local digital commerce
        
        === BUSINESS CASE VALIDATION ===
        Market Opportunity:
        • Addressable market: 2.1M low-literacy adults in Ghana
        • Revenue potential: GHS 8.4M annually if scaled nationally
        • Competitive moat: 18-month technology leadership advantage
        • Social impact alignment: Strong ESG story for investors
        
        Scaling Requirements:
        • Technology investment: GHS 2.1M for national rollout
        • Training program: Community educator network (450 people)
        • Marketing approach: Community-based demonstration model
        • Timeline to scale: 8 months for 50% market penetration
        
        === RECOGNITION & AWARDS ===
        • UNESCO Digital Innovation Award (shortlisted)
        • Ghana Tech Awards - Financial Inclusion Category (winner)
        • World Bank Innovation Challenge (finalist)
        • Local media: Featured in 23 articles and TV segments
        
        === NEXT PHASE OPPORTUNITIES ===
        • National rollout planning initiated
        • Partnership discussions with UNESCO and World Bank
        • White-label licensing opportunities in 12 African countries
        • Voice-based credit scoring system development
        • Integration with government social protection programs
        """,
        completion_date=datetime.now() - timedelta(days=14),
        success_metrics={
            "user_adoption_rate": 0.91,
            "transaction_success_rate": 0.87,
            "user_satisfaction": 9.2,
            "voice_accuracy": 0.89,
            "community_referral_rate": 0.52,  # 156/300 participants
            "financial_behavior_change": 0.52,  # 52% reduction in cash dependency
            "competitive_differentiation": 1.0,  # First in market
            "scalability_score": 0.84
        }
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


# Enhanced market context for emerging markets
EMERGING_MARKET_CONTEXT = {
    "primary_markets": [
        "Ghana - Mobile Money Hub (39% penetration)",
        "Nigeria - Largest Market (15% formal banking)",
        "Kenya - M-Pesa Pioneer (73% mobile money usage)",
        "Senegal - Regional Gateway (Wave expansion)",
        "Rwanda - Digital Transformation Leader",
        "Tanzania - Cross-border Trade Hub"
    ],
    "mobile_operators": [
        "MTN Group (Nigeria, Ghana, Uganda, Rwanda)",
        "Vodacom (Tanzania, DRC, Mozambique)",
        "Orange (Senegal, Mali, Burkina Faso)",
        "Airtel Africa (Multi-country presence)",
        "Safaricom (Kenya, Ethiopia)"
    ],
    "local_languages": [
        "Twi/Akan (Ghana - 9M speakers)",
        "Hausa (Nigeria/Ghana - 70M speakers)", 
        "Swahili (East Africa - 200M speakers)",
        "Yoruba (Nigeria - 45M speakers)",
        "Wolof (Senegal - 12M speakers)",
        "French (West/Central Africa - 120M speakers)",
        "Amharic (Ethiopia - 32M speakers)"
    ],
    "regulatory_bodies": [
        "Bank of Ghana (Payment Systems Oversight)",
        "Central Bank of Nigeria (PSP Licensing)",
        "Bank of Tanzania (Mobile Money Regulations)",
        "Central Bank of Kenya (National Payment System)",
        "BCEAO (West African Economic Union)",
        "National Bank of Rwanda (Cashless Economy)"
    ],
    "key_economic_indicators": {
        "mobile_penetration": "87%",
        "smartphone_penetration": "34%",
        "banking_penetration": "43%",
        "mobile_money_penetration": "56%",
        "rural_population": "57%",
        "informal_economy": "89%",
        "cross_border_remittances": "$48B annually",
        "youth_population": "65% under 35"
    },
    "infrastructure_challenges": [
        "Intermittent electricity (average 14 hours/day)",
        "Limited internet connectivity (23% reliable broadband)",
        "Cash-dominant economy (78% of transactions)",
        "Low financial literacy (34% basic numeracy)",
        "Identity documentation gaps (43% lack formal ID)",
        "Agent network sustainability challenges"
    ],
    "opportunity_areas": [
        "Agricultural value chain payments",
        "Cross-border remittances and trade",
        "Women's financial inclusion initiatives",
        "Youth entrepreneurship funding",
        "Government-to-person (G2P) payments",
        "Insurance and risk management products",
        "Credit scoring for informal sector",
        "Merchant payment systems"
    ]
}


def get_market_context() -> Dict:
    """Return comprehensive emerging market context data for AI agents."""
    return EMERGING_MARKET_CONTEXT


def get_business_intelligence_scenarios() -> List[Dict]:
    """Return scenarios designed to showcase sophisticated business intelligence."""
    return [
        {
            "scenario_name": "Multi-dimensional Market Analysis",
            "complexity_level": "High", 
            "key_insight": "Northern Region expansion requires customer segmentation strategy",
            "cross_bmc_impact": ["customer_segments", "channels", "key_partnerships", "revenue_streams"],
            "confidence_factors": ["Large sample size", "Geographic diversity", "Competitive analysis depth"],
            "strategic_implications": "Market entry strategy with localized approach"
        },
        {
            "scenario_name": "Product-Market Fit Crisis", 
            "complexity_level": "High",
            "key_insight": "Feature complexity misaligned with customer digital maturity",
            "cross_bmc_impact": ["value_propositions", "customer_relationships", "channels"],
            "confidence_factors": ["Clear failure metrics", "Customer feedback themes", "Competitive impact"],
            "strategic_implications": "Product strategy pivot required"
        },
        {
            "scenario_name": "Strategic Partnership Evaluation",
            "complexity_level": "Very High",
            "key_insight": "Cross-border expansion requires careful risk-reward analysis",
            "cross_bmc_impact": ["key_partnerships", "revenue_streams", "key_activities", "cost_structure"],
            "confidence_factors": ["Mixed signals", "High complexity", "Regulatory uncertainty"],
            "strategic_implications": "Phased approach with deep technical due diligence"
        },
        {
            "scenario_name": "Regulatory Achievement",
            "complexity_level": "Medium",
            "key_insight": "Compliance success enables market expansion and credibility",
            "cross_bmc_impact": ["key_resources", "key_activities", "channels", "customer_segments"],
            "confidence_factors": ["Regulatory validation", "Audit success", "Market response"],
            "strategic_implications": "Accelerated growth strategy with regulatory moat"
        },
        {
            "scenario_name": "Competitive Pressure Response",
            "complexity_level": "High",
            "key_insight": "Price war strategy failed against larger competitor",
            "cross_bmc_impact": ["value_propositions", "revenue_streams", "cost_structure"],
            "confidence_factors": ["Clear failure metrics", "Customer behavior data", "Financial impact"],
            "strategic_implications": "Differentiation strategy required, not cost leadership"
        },
        {
            "scenario_name": "Innovation Breakthrough",
            "complexity_level": "Medium",
            "key_insight": "Voice technology creates new market segment for low-literacy users",
            "cross_bmc_impact": ["customer_segments", "value_propositions", "channels", "key_resources"],
            "confidence_factors": ["Strong pilot results", "Market validation", "Technical success"],
            "strategic_implications": "Scale innovation for competitive advantage"
        }
    ]


def get_confidence_calibration_examples() -> List[Dict]:
    """Return examples for confidence score calibration."""
    return [
        {
            "scenario": "Large-scale market research with 2,847 respondents",
            "confidence_range": "0.85-0.95",
            "reasoning": "High sample size, geographic diversity, multiple validation methods"
        },
        {
            "scenario": "Clear product failure with 31% completion rate",
            "confidence_range": "0.90-0.98", 
            "reasoning": "Unambiguous failure metrics, consistent customer feedback themes"
        },
        {
            "scenario": "Mixed partnership results with regulatory uncertainty",
            "confidence_range": "0.45-0.65",
            "reasoning": "Conflicting signals, external dependencies, high complexity"
        },
        {
            "scenario": "Regulatory compliance success with audit validation",
            "confidence_range": "0.88-0.95",
            "reasoning": "Third-party validation, measurable compliance metrics"
        },
        {
            "scenario": "Customer retention campaign with clear ROI failure",
            "confidence_range": "0.85-0.93",
            "reasoning": "Measurable campaign metrics, clear financial impact"
        },
        {
            "scenario": "Innovation pilot with outstanding user satisfaction",
            "confidence_range": "0.82-0.90",
            "reasoning": "Small sample size but strong results, novel technology risk"
        }
    ]