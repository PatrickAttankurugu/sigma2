"""
Multi-Agent System using LangGraph

This module implements a collaborative multi-agent system with 4 specialized agents:
1. Strategy Agent - Business strategy and BMC coherence analysis
2. Market Research Agent - Customer and market validation
3. Product Agent - Value proposition and competitive analysis
4. Execution Agent - Action analysis and next step recommendations

The agents work together using LangGraph to provide comprehensive business analysis.
"""

from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass
import json
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from modules.agent_tools import AgentTools
from modules.bmc_canvas import BusinessModelCanvas
from modules.utils import get_logger

logger = get_logger(__name__)


# ========== State Definition ==========

class AgentState(TypedDict):
    """State shared across all agents"""
    action_data: Dict[str, Any]  # The action/experiment being analyzed
    bmc: Dict[str, Any]  # Current BMC state
    business_stage: str  # validation/growth/scale

    # Agent outputs
    strategy_analysis: Optional[str]
    market_analysis: Optional[str]
    product_analysis: Optional[str]
    execution_analysis: Optional[str]

    # Final synthesis
    final_analysis: Optional[str]
    proposed_changes: Optional[List[Dict]]
    next_steps: Optional[List[Dict]]

    # Streaming callback
    stream_callback: Optional[Callable[[str, str], None]]


# ========== Agent Prompts ==========

STRATEGY_AGENT_PROMPT = """You are the Strategy Agent, responsible for high-level business strategy analysis.

Your role is to:
1. Analyze the overall Business Model Canvas for coherence and alignment
2. Identify strategic gaps or inconsistencies
3. Assess whether the action aligns with the business stage (validation/growth/scale)
4. Provide strategic context for other agents

Current Business Stage: {business_stage}

Action Being Analyzed:
{action_data}

Use your tools to analyze the BMC and provide strategic insights. Focus on:
- How this action fits into the overall business strategy
- Whether the BMC components support each other
- Strategic risks or opportunities revealed by this action

Provide a concise strategic analysis (2-3 sentences) that other agents can build upon.
"""

MARKET_RESEARCH_AGENT_PROMPT = """You are the Market Research Agent, responsible for customer and market validation.

Your role is to:
1. Analyze customer segments for specificity and viability
2. Research market trends and opportunities (when search is available)
3. Validate market assumptions from the action
4. Provide market context for recommendations

Action Being Analyzed:
{action_data}

Strategy Context (from Strategy Agent):
{strategy_analysis}

Use your tools to analyze customer segments and market opportunities. Focus on:
- How this action validates or invalidates customer assumptions
- Market trends that support or contradict the approach
- Customer segment refinements needed

Provide a concise market analysis (2-3 sentences) focusing on customer and market insights.
"""

PRODUCT_AGENT_PROMPT = """You are the Product Agent, responsible for value proposition and product analysis.

Your role is to:
1. Analyze value propositions for clarity and differentiation
2. Assess competitive positioning
3. Evaluate product-market fit based on action results
4. Recommend value proposition refinements

Action Being Analyzed:
{action_data}

Previous Context:
Strategy: {strategy_analysis}
Market: {market_analysis}

Use your tools to analyze value propositions and competitive positioning. Focus on:
- How this action validates or changes the value proposition
- Competitive differentiation insights
- Product-market fit improvements

Provide a concise product analysis (2-3 sentences) focusing on value proposition and positioning.
"""

EXECUTION_AGENT_PROMPT = """You are the Execution Agent, responsible for synthesizing insights and generating actionable recommendations.

Your role is to:
1. Analyze the specific action's impact on the business model
2. Synthesize insights from all other agents
3. Propose specific BMC changes with confidence scores
4. Generate prioritized next steps

Action Being Analyzed:
{action_data}

Context from Other Agents:
Strategy: {strategy_analysis}
Market: {market_analysis}
Product: {product_analysis}

Business Stage: {business_stage}

Use your tools and the context from other agents to:
1. Provide a comprehensive analysis of what this action means for the business
2. Propose specific changes to BMC sections (add/modify/remove items)
3. Generate 3-5 prioritized next steps with timelines and resources

Output your response as JSON with this structure:
{{
  "analysis": "Comprehensive 2-3 sentence synthesis",
  "changes": [
    {{
      "section": "customer_segments|value_propositions|business_models|market_opportunities",
      "type": "add|modify|remove",
      "current": "existing item (for modify/remove)",
      "new": "new/replacement text",
      "reason": "explanation based on action outcome and agent insights",
      "confidence": 0.85
    }}
  ],
  "next_steps": [
    {{
      "title": "Specific action title",
      "description": "Detailed description",
      "timeline": "Specific timeframe",
      "resources_needed": ["resource1", "resource2"],
      "success_metrics": ["metric1", "metric2"],
      "priority": "high|medium|low",
      "difficulty": "easy|medium|hard",
      "stage": "validation|growth|scale"
    }}
  ]
}}
"""


# ========== Multi-Agent System ==========

class MultiAgentSystem:
    """Orchestrates multiple AI agents working together on business analysis"""

    def __init__(self, bmc: BusinessModelCanvas, api_key: str, stream_callback: Optional[Callable[[str, str], None]] = None):
        self.bmc = bmc
        self.api_key = api_key
        self.stream_callback = stream_callback

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=2000
        )

        # Initialize tools
        self.agent_tools = AgentTools(bmc)

        # Build the agent graph
        self.graph = self._build_graph()

        logger.info("Multi-agent system initialized with 4 specialized agents")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("strategy_agent", self._strategy_agent_node)
        workflow.add_node("market_agent", self._market_agent_node)
        workflow.add_node("product_agent", self._product_agent_node)
        workflow.add_node("execution_agent", self._execution_agent_node)

        # Define the flow: sequential with insights building on each other
        workflow.set_entry_point("strategy_agent")
        workflow.add_edge("strategy_agent", "market_agent")
        workflow.add_edge("market_agent", "product_agent")
        workflow.add_edge("product_agent", "execution_agent")
        workflow.add_edge("execution_agent", END)

        return workflow.compile()

    async def _stream_update(self, agent_name: str, message: str):
        """Send streaming update if callback is provided"""
        if self.stream_callback:
            try:
                self.stream_callback(agent_name, message)
            except Exception as e:
                logger.error(f"Error in stream callback: {e}")

    async def _strategy_agent_node(self, state: AgentState) -> AgentState:
        """Strategy Agent node"""
        await self._stream_update("strategy", "🎯 Strategy Agent analyzing business model coherence...")

        try:
            # Get tool results
            tools = self.agent_tools.get_strategy_tools()
            bmc_coherence = self.agent_tools.analyze_bmc_coherence()
            business_stage = self.agent_tools.detect_business_stage()

            # Create analysis prompt
            prompt = STRATEGY_AGENT_PROMPT.format(
                business_stage=state['business_stage'],
                action_data=json.dumps(state['action_data'], indent=2)
            )

            # Add tool results to prompt
            prompt += f"\n\nBMC Coherence Analysis:\n{bmc_coherence}\n\nBusiness Stage Detection:\n{business_stage}"

            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            analysis = response.content

            await self._stream_update("strategy", f"✅ Strategy Analysis: {analysis[:200]}...")

            state['strategy_analysis'] = analysis
            return state

        except Exception as e:
            logger.error(f"Error in strategy agent: {e}")
            state['strategy_analysis'] = f"Strategy analysis encountered an error: {str(e)}"
            return state

    async def _market_agent_node(self, state: AgentState) -> AgentState:
        """Market Research Agent node"""
        await self._stream_update("market", "📊 Market Research Agent analyzing customer segments...")

        try:
            # Get tool results
            customer_analysis = self.agent_tools.analyze_customer_segments()

            # Create analysis prompt
            prompt = MARKET_RESEARCH_AGENT_PROMPT.format(
                action_data=json.dumps(state['action_data'], indent=2),
                strategy_analysis=state.get('strategy_analysis', '')
            )

            # Add tool results
            prompt += f"\n\nCustomer Segment Analysis:\n{customer_analysis}"

            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            analysis = response.content

            await self._stream_update("market", f"✅ Market Analysis: {analysis[:200]}...")

            state['market_analysis'] = analysis
            return state

        except Exception as e:
            logger.error(f"Error in market agent: {e}")
            state['market_analysis'] = f"Market analysis encountered an error: {str(e)}"
            return state

    async def _product_agent_node(self, state: AgentState) -> AgentState:
        """Product Agent node"""
        await self._stream_update("product", "🎨 Product Agent analyzing value propositions...")

        try:
            # Get tool results
            value_prop_analysis = self.agent_tools.analyze_value_propositions()
            competitive_assessment = self.agent_tools.assess_competitive_positioning()

            # Create analysis prompt
            prompt = PRODUCT_AGENT_PROMPT.format(
                action_data=json.dumps(state['action_data'], indent=2),
                strategy_analysis=state.get('strategy_analysis', ''),
                market_analysis=state.get('market_analysis', '')
            )

            # Add tool results
            prompt += f"\n\nValue Proposition Analysis:\n{value_prop_analysis}\n\nCompetitive Assessment:\n{competitive_assessment}"

            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            analysis = response.content

            await self._stream_update("product", f"✅ Product Analysis: {analysis[:200]}...")

            state['product_analysis'] = analysis
            return state

        except Exception as e:
            logger.error(f"Error in product agent: {e}")
            state['product_analysis'] = f"Product analysis encountered an error: {str(e)}"
            return state

    async def _execution_agent_node(self, state: AgentState) -> AgentState:
        """Execution Agent node - synthesizes all insights and generates recommendations"""
        await self._stream_update("execution", "⚡ Execution Agent synthesizing insights and generating recommendations...")

        try:
            # Get tool results
            action_impact = self.agent_tools.analyze_action_impact(json.dumps(state['action_data']))
            next_steps_suggestions = self.agent_tools.generate_next_steps("")

            # Create synthesis prompt
            prompt = EXECUTION_AGENT_PROMPT.format(
                action_data=json.dumps(state['action_data'], indent=2),
                strategy_analysis=state.get('strategy_analysis', ''),
                market_analysis=state.get('market_analysis', ''),
                product_analysis=state.get('product_analysis', ''),
                business_stage=state['business_stage']
            )

            # Add tool results
            prompt += f"\n\nAction Impact Analysis:\n{action_impact}\n\nNext Steps Framework:\n{next_steps_suggestions}"

            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            result = response.content

            # Parse JSON response
            try:
                # Try to extract JSON from response
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0].strip()
                else:
                    json_str = result

                parsed_result = json.loads(json_str)

                state['final_analysis'] = parsed_result.get('analysis', '')
                state['proposed_changes'] = parsed_result.get('changes', [])
                state['next_steps'] = parsed_result.get('next_steps', [])

                await self._stream_update("execution", f"✅ Final Analysis: {state['final_analysis']}")
                await self._stream_update("execution", f"📝 Proposed {len(state['proposed_changes'])} changes")
                await self._stream_update("execution", f"🎯 Generated {len(state['next_steps'])} next steps")

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing execution agent JSON response: {e}")
                state['final_analysis'] = result
                state['proposed_changes'] = []
                state['next_steps'] = []

            return state

        except Exception as e:
            logger.error(f"Error in execution agent: {e}")
            state['final_analysis'] = f"Execution analysis encountered an error: {str(e)}"
            state['proposed_changes'] = []
            state['next_steps'] = []
            return state

    async def analyze_action_async(
        self,
        action_data: Dict[str, Any],
        stream_callback: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an action using the multi-agent system.

        Args:
            action_data: Action data with keys: title, description, outcome, metrics
            stream_callback: Optional callback for streaming updates (agent_name, message)

        Returns:
            Dict with keys: analysis, changes, next_steps
        """
        try:
            # Initialize state
            initial_state: AgentState = {
                'action_data': action_data,
                'bmc': {
                    'customer_segments': self.bmc.get_section('customer_segments'),
                    'value_propositions': self.bmc.get_section('value_propositions'),
                    'business_models': self.bmc.get_section('business_models'),
                    'market_opportunities': self.bmc.get_section('market_opportunities')
                },
                'business_stage': self.bmc.get_business_stage(),
                'strategy_analysis': None,
                'market_analysis': None,
                'product_analysis': None,
                'execution_analysis': None,
                'final_analysis': None,
                'proposed_changes': None,
                'next_steps': None,
                'stream_callback': stream_callback or self.stream_callback
            }

            # Run the agent graph
            if stream_callback or self.stream_callback:
                await self._stream_update("system", "🚀 Starting multi-agent analysis...")

            final_state = await self.graph.ainvoke(initial_state)

            if stream_callback or self.stream_callback:
                await self._stream_update("system", "✨ Multi-agent analysis complete!")

            return {
                'analysis': final_state.get('final_analysis', ''),
                'changes': final_state.get('proposed_changes', []),
                'next_steps': final_state.get('next_steps', []),
                'agent_insights': {
                    'strategy': final_state.get('strategy_analysis', ''),
                    'market': final_state.get('market_analysis', ''),
                    'product': final_state.get('product_analysis', '')
                }
            }

        except Exception as e:
            logger.error(f"Error in multi-agent analysis: {e}")
            raise

    def analyze_action(
        self,
        action_data: Dict[str, Any],
        stream_callback: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for analyze_action_async.
        """
        return asyncio.run(self.analyze_action_async(action_data, stream_callback))
