"""
WebSocket Server for Real-Time Streaming

This module provides a WebSocket server for real-time communication between
the multi-agent system and the frontend. It enables streaming of agent outputs
and business design updates.

Usage:
    python websocket_server.py

The server will start on ws://localhost:8765
"""

import asyncio
import json
import os
from typing import Dict, Set, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from datetime import datetime
from dotenv import load_dotenv

from modules.bmc_canvas import BusinessModelCanvas
from modules.multi_agent_system import MultiAgentSystem
from modules.utils import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Store active connections and BMC instances per session
active_connections: Set[WebSocketServerProtocol] = set()
session_bmcs: Dict[str, BusinessModelCanvas] = {}
session_agents: Dict[str, MultiAgentSystem] = {}


class WebSocketServer:
    """WebSocket server for streaming multi-agent analysis"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        logger.info(f"WebSocket server initialized on {host}:{port}")

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection"""
        session_id = None

        try:
            # Register connection
            active_connections.add(websocket)
            logger.info(f"New connection from {websocket.remote_address}")

            # Send welcome message
            await self.send_message(websocket, "system", "info", {
                "message": "Connected to SIGMA Multi-Agent System",
                "timestamp": datetime.now().isoformat()
            })

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)

                except json.JSONDecodeError as e:
                    await self.send_message(websocket, "system", "error", {
                        "error": "Invalid JSON format",
                        "details": str(e)
                    })
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self.send_message(websocket, "system", "error", {
                        "error": "Error processing message",
                        "details": str(e)
                    })

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error in connection handler: {e}")
        finally:
            # Cleanup
            active_connections.discard(websocket)
            if session_id and session_id in session_bmcs:
                del session_bmcs[session_id]
            if session_id and session_id in session_agents:
                del session_agents[session_id]

    async def handle_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type")
        payload = data.get("payload", {})

        logger.info(f"Received message type: {message_type}")

        if message_type == "init_session":
            await self.handle_init_session(websocket, payload)

        elif message_type == "update_bmc":
            await self.handle_update_bmc(websocket, payload)

        elif message_type == "analyze_action":
            await self.handle_analyze_action(websocket, payload)

        elif message_type == "get_bmc":
            await self.handle_get_bmc(websocket, payload)

        elif message_type == "ping":
            await self.send_message(websocket, "system", "pong", {
                "timestamp": datetime.now().isoformat()
            })

        else:
            await self.send_message(websocket, "system", "error", {
                "error": "Unknown message type",
                "received_type": message_type
            })

    async def handle_init_session(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]):
        """Initialize a new session with BMC and multi-agent system"""
        try:
            session_id = payload.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Create BMC instance
            bmc = BusinessModelCanvas()

            # Initialize with any existing BMC data
            if "bmc_data" in payload:
                bmc_data = payload["bmc_data"]
                for section, items in bmc_data.items():
                    if section in ['customer_segments', 'value_propositions', 'business_models', 'market_opportunities']:
                        bmc.update_section(section, items)

            # Create multi-agent system
            def stream_callback(agent_name: str, message: str):
                """Callback for streaming agent updates"""
                asyncio.create_task(
                    self.send_message(websocket, "agent", "stream", {
                        "agent": agent_name,
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    })
                )

            agent_system = MultiAgentSystem(bmc, self.api_key, stream_callback)

            # Store in session
            session_bmcs[session_id] = bmc
            session_agents[session_id] = agent_system

            await self.send_message(websocket, "session", "initialized", {
                "session_id": session_id,
                "message": "Session initialized with multi-agent system",
                "business_stage": bmc.get_business_stage()
            })

            logger.info(f"Session initialized: {session_id}")

        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            await self.send_message(websocket, "system", "error", {
                "error": "Failed to initialize session",
                "details": str(e)
            })

    async def handle_update_bmc(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]):
        """Update BMC section"""
        try:
            session_id = payload.get("session_id")
            section = payload.get("section")
            items = payload.get("items", [])

            if not session_id or session_id not in session_bmcs:
                raise ValueError("Invalid session_id")

            if section not in ['customer_segments', 'value_propositions', 'business_models', 'market_opportunities']:
                raise ValueError(f"Invalid section: {section}")

            bmc = session_bmcs[session_id]
            bmc.update_section(section, items)

            await self.send_message(websocket, "bmc", "updated", {
                "session_id": session_id,
                "section": section,
                "items": items,
                "business_stage": bmc.get_business_stage()
            })

            logger.info(f"BMC updated for session {session_id}: {section}")

        except Exception as e:
            logger.error(f"Error updating BMC: {e}")
            await self.send_message(websocket, "system", "error", {
                "error": "Failed to update BMC",
                "details": str(e)
            })

    async def handle_analyze_action(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]):
        """Analyze an action using the multi-agent system"""
        try:
            session_id = payload.get("session_id")
            action_data = payload.get("action_data")

            if not session_id or session_id not in session_agents:
                raise ValueError("Invalid session_id or session not initialized")

            if not action_data:
                raise ValueError("action_data is required")

            # Validate action data
            required_fields = ['title', 'description', 'outcome']
            for field in required_fields:
                if field not in action_data:
                    raise ValueError(f"action_data missing required field: {field}")

            agent_system = session_agents[session_id]

            # Send analysis started message
            await self.send_message(websocket, "analysis", "started", {
                "session_id": session_id,
                "action_title": action_data.get("title")
            })

            # Run multi-agent analysis (streaming will happen via callback)
            result = await agent_system.analyze_action_async(action_data)

            # Send final results
            await self.send_message(websocket, "analysis", "completed", {
                "session_id": session_id,
                "result": result
            })

            logger.info(f"Action analyzed for session {session_id}: {action_data.get('title')}")

        except Exception as e:
            logger.error(f"Error analyzing action: {e}")
            await self.send_message(websocket, "system", "error", {
                "error": "Failed to analyze action",
                "details": str(e)
            })

    async def handle_get_bmc(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]):
        """Get current BMC state"""
        try:
            session_id = payload.get("session_id")

            if not session_id or session_id not in session_bmcs:
                raise ValueError("Invalid session_id")

            bmc = session_bmcs[session_id]

            bmc_data = {
                "customer_segments": bmc.get_section('customer_segments'),
                "value_propositions": bmc.get_section('value_propositions'),
                "business_models": bmc.get_section('business_models'),
                "market_opportunities": bmc.get_section('market_opportunities'),
                "business_stage": bmc.get_business_stage(),
                "risk_assessment": bmc.get_risk_assessment()
            }

            await self.send_message(websocket, "bmc", "data", {
                "session_id": session_id,
                "bmc": bmc_data
            })

        except Exception as e:
            logger.error(f"Error getting BMC: {e}")
            await self.send_message(websocket, "system", "error", {
                "error": "Failed to get BMC",
                "details": str(e)
            })

    async def send_message(
        self,
        websocket: WebSocketServerProtocol,
        category: str,
        message_type: str,
        data: Dict[str, Any]
    ):
        """Send a message to the client"""
        try:
            message = {
                "category": category,
                "type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    """Main entry point"""
    server = WebSocketServer(host="localhost", port=8765)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
