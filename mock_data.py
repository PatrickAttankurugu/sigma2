"""
Mock data for the Agentic AI Actions Co-pilot system.

This module provides sample data representing a Ghanaian fintech startup
with realistic business model canvas and completed action scenarios.
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
    """Return a sample BMC for a Ghanaian fintech startup focused on mobile payments."""
    return BusinessModelCanvas(
        customer_segments=[
            "Informal traders in markets (Makola, Kejetia, Tamale Central)",
            "Rural smallholder farmers in Northern regions",
            "Unbanked youth aged 18-35 in urban areas",
            "Small-scale entrepreneurs and micro-businesses",
            "Cross-border traders between Ghana, Burkina Faso, and Togo"
        ],
        value_propositions=[
            "USSD-based mobile money accessible on basic phones",
            "Transaction fees 50% lower than traditional banks",
            "Local language support (Twi, Hausa, Ewe, Ga)",
            "24/7 agent network for cash-in/cash-out services",
            "Instant peer-to-peer transfers within Ghana",
            "Integration with traditional savings groups (susu)"
        ],
        channels=[
            "Mobile network operator partnerships (MTN, Vodafone Ghana, AirtelTigo)",
            "Community agent network in markets and rural areas",
            "USSD shortcode (*123*456#) accessible on all phones",
            "Word-of-mouth referrals through trusted community members",
            "Radio advertisements in local languages",
            "SMS marketing campaigns"
        ],
        customer_relationships=[
            "Personal assistance through local agents",
            "Community-based customer education workshops",
            "24/7 USSD customer service in local languages",
            "Trust-building through local partnerships",
            "Loyalty programs for frequent users"
        ],
        revenue_streams=[
            "Transaction fees (2% on transfers, 1% on payments)",
            "Agent commission structure (40% of transaction fees)",
            "Bill payment service charges",
            "Cross-border transfer fees",
            "Interest on float funds",
            "Premium service subscriptions"
        ],
        key_resources=[
            "USSD platform technology and mobile integration",
            "Agent network across Ghana (urban and rural)",
            "Banking license and regulatory compliance",
            "Customer database and transaction history",
            "Local language customer support team",
            "Partnership agreements with telcos and banks"
        ],
        key_activities=[
            "Mobile platform development and maintenance",
            "Agent recruitment, training, and management",
            "Regulatory compliance and reporting",
            "Customer acquisition and retention",
            "Risk management and fraud prevention",
            "Community education and financial literacy programs"
        ],
        key_partnerships=[
            "Rural and community banks (ARB Apex Bank)",
            "Mobile network operators (MTN Mobile Money, Vodafone Cash)",
            "Bank of Ghana for regulatory compliance",
            "Local market associations and trader groups",
            "International remittance providers",
            "Technology vendors for USSD infrastructure"
        ],
        cost_structure=[
            "Technology infrastructure and platform maintenance",
            "Agent network setup and ongoing commissions",
            "Regulatory compliance and licensing costs",
            "Customer acquisition and marketing expenses",
            "Staff salaries and operational overhead",
            "Fraud prevention and security measures"
        ]
    )


def get_sample_value_proposition_canvas() -> ValuePropositionCanvas:
    """Return a sample VPC aligned with the Ghanaian fintech BMC."""
    return ValuePropositionCanvas(
        jobs_to_be_done=[
            "Send money to family members in rural areas",
            "Pay for goods and services without cash",
            "Save money securely without a bank account",
            "Receive payments from customers instantly",
            "Access financial services from remote locations"
        ],
        pains=[
            "High transaction fees from traditional banks",
            "Long travel distances to bank branches",
            "Complex account opening requirements",
            "Language barriers with financial services",
            "Risk of carrying cash in markets",
            "Unreliable internet connectivity for apps"
        ],
        gains=[
            "Quick and convenient money transfers",
            "Lower costs than traditional banking",
            "Financial inclusion for unbanked populations",
            "Secure digital transactions",
            "Access to credit and savings products",
            "Integration with existing business practices"
        ],
        products_services=[
            "USSD mobile money platform",
            "Agent-assisted cash services",
            "Peer-to-peer money transfers",
            "Bill payment services",
            "Micro-savings accounts",
            "Merchant payment solutions"
        ],
        pain_relievers=[
            "50% lower fees than traditional banks",
            "USSD works on basic phones without internet",
            "Local agents eliminate travel to banks",
            "Multi-language customer support",
            "Secure digital alternative to cash",
            "Simple account setup process"
        ],
        gain_creators=[
            "Instant money transfers across Ghana",
            "24/7 availability without bank hours",
            "Financial empowerment for unbanked",
            "Integration with traditional susu groups",
            "Loyalty rewards for frequent users",
            "Business growth through digital payments"
        ]
    )


def get_sample_completed_actions() -> List[CompletedAction]:
    """Return sample completed actions representing realistic business scenarios."""

    actions = []

    # Scenario 1: Successful pricing survey
    actions.append(CompletedAction(
        title="Makola Market Transaction Fee Survey",
        description="Conducted in-person survey with 50 traders at Makola Market to understand fee tolerance and payment preferences",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        Survey Results Summary:
        - Total respondents: 50 traders (vegetables, textiles, electronics)
        - Current fee acceptance: 60% accept 2% transaction fee
        - Price sensitivity: 30% want lower fees (prefer 1% or less)
        - Premium willingness: 10% willing to pay 3% for instant transfers
        - Key insights:
          * Traders prioritize speed over cost for urgent payments
          * Weekend/evening transactions most price-sensitive
          * Cross-border traders willing to pay higher fees
          * Integration with daily cash flow crucial
        - Average monthly transaction volume per trader: GHS 1,200
        - Preferred transaction times: Morning (8-10am), Evening (5-7pm)
        """,
        completion_date=datetime.now() - timedelta(days=2),
        success_metrics={
            "response_rate": 0.83,
            "data_quality_score": 0.92,
            "actionable_insights": 7,
            "follow_up_interviews": 12
        }
    ))

    # Scenario 2: Failed USSD interface test
    actions.append(CompletedAction(
        title="Rural USSD Interface Usability Testing",
        description="Tested current USSD menu flow with 100 rural users across 5 Northern region communities",
        outcome=ActionOutcome.FAILED,
        results_data="""
        Usability Test Results:
        - Total participants: 100 users (mix of literacy levels, ages 18-65)
        - Task completion rate: 55% (target was 85%)
        - Critical failure points:
          * 45% couldn't complete money transfer (menu too complex)
          * 38% got lost in sub-menu navigation
          * 52% needed agent assistance for first transaction
        - Language issues: 30% struggled with English prompts
        - Phone compatibility: 15% had issues with older phone models
        - User feedback themes:
          * "Too many steps to send money"
          * "Need more Twi language options"
          * "Menu timeout too short"
          * "Confirmation process confusing"
        - Average time to complete transfer: 4.2 minutes (target: 2 minutes)
        - Error rate: 28% (unacceptably high)
        """,
        completion_date=datetime.now() - timedelta(days=5),
        success_metrics={
            "task_completion_rate": 0.55,
            "user_satisfaction": 2.3,  # out of 5
            "error_rate": 0.28,
            "time_to_complete": 252  # seconds
        }
    ))

    # Scenario 3: Inconclusive agent network pilot
    actions.append(CompletedAction(
        title="Kumasi Region Agent Network Pilot",
        description="Deployed 20 trained agents across Kumasi markets and suburbs to test agent-based service model",
        outcome=ActionOutcome.INCONCLUSIVE,
        results_data="""
        Agent Network Pilot Results (30 days):
        - Agents deployed: 20 across Kejetia Market and 4 suburbs
        - Total transactions processed: 2,847 (above target of 2,000)
        - Transaction volume: GHS 45,600
        - Average daily transactions per agent: 4.7

        Positive indicators:
        - High customer satisfaction: 4.2/5 rating
        - Strong transaction growth: +15% week-over-week
        - Good geographic coverage achieved
        - Agent availability: 94% during business hours

        Concerning issues:
        - Agent complaints about low commissions (60% want increase)
        - 3 agents quit due to insufficient earnings
        - Float management problems (agents running out of cash)
        - Competition from established MTN Mobile Money agents

        Financial performance:
        - Revenue generated: GHS 1,824 (transaction fees)
        - Agent commissions paid: GHS 730 (40% of revenue)
        - Operational costs: GHS 1,200 (training, support, materials)
        - Net result: -GHS 106 (slight loss but close to breakeven)

        Mixed signals require deeper analysis before scaling.
        """,
        completion_date=datetime.now() - timedelta(days=7),
        success_metrics={
            "transaction_volume": 45600,
            "agent_retention_rate": 0.85,
            "customer_satisfaction": 4.2,
            "revenue_per_agent": 91.2,
            "breakeven_ratio": 0.94
        }
    ))

    # Scenario 4: Successful partnership negotiation
    actions.append(CompletedAction(
        title="ARB Apex Bank Partnership Agreement",
        description="Negotiated partnership terms with Association of Rural Banks for expanded rural coverage",
        outcome=ActionOutcome.SUCCESSFUL,
        results_data="""
        Partnership Agreement Summary:
        - Partner: ARB Apex Bank (represents 140 rural banks)
        - Coverage: Access to rural bank branches as cash-in/out points
        - Terms agreed:
          * Revenue sharing: 70% us, 30% ARB banks
          * Integration timeline: 6 months phased rollout
          * Minimum transaction guarantees: 500 per branch/month
          * Joint marketing and customer education programs

        Strategic benefits:
        - Immediate access to 140 rural locations
        - Trusted local presence in underbanked areas
        - Reduced agent recruitment costs
        - Regulatory credibility through bank partnerships
        - Customer trust through established financial institutions

        Implementation requirements:
        - Technical integration with ARB core banking system
        - Staff training for 420 bank employees
        - Co-branded marketing materials development
        - Compliance audit and approval process

        Projected impact:
        - Additional 25,000 potential customers in rural areas
        - Estimated monthly volume increase: GHS 180,000
        - Break-even expected within 8 months
        """,
        completion_date=datetime.now() - timedelta(days=1),
        success_metrics={
            "partner_branches": 140,
            "projected_new_customers": 25000,
            "expected_monthly_volume": 180000,
            "implementation_timeline": 6,  # months
            "revenue_share": 0.70
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


# Additional helper data for African market context
GHANAIAN_MARKET_CONTEXT = {
    "major_markets": [
        "Makola Market (Accra)",
        "Kejetia Market (Kumasi)",
        "Tamale Central Market",
        "Cape Coast Market",
        "Ho Central Market"
    ],
    "mobile_operators": [
        "MTN Ghana (Mobile Money leader)",
        "Vodafone Ghana (Cash platform)",
        "AirtelTigo (Money service)"
    ],
    "local_languages": [
        "Twi (Akan)",
        "Hausa",
        "Ewe",
        "Ga",
        "Dagbani"
    ],
    "regulatory_bodies": [
        "Bank of Ghana (Central Bank)",
        "National Communications Authority",
        "Ghana Association of Banks"
    ],
    "key_economic_indicators": {
        "mobile_penetration": "84%",
        "banking_penetration": "58%",
        "mobile_money_penetration": "39%",
        "rural_population": "43%",
        "informal_economy": "86%"
    }
}


def get_market_context() -> Dict:
    """Return Ghanaian market context data for AI agents."""
    return GHANAIAN_MARKET_CONTEXT