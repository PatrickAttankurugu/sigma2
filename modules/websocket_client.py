"""
WebSocket Client for Streamlit Integration

This module provides a WebSocket client that can be used with Streamlit
to enable real-time streaming of multi-agent analysis.
"""

import asyncio
import json
import threading
from typing import Dict, Any, Callable, Optional, List
import websockets
from websockets.client import WebSocketClientProtocol
from queue import Queue
import time
from modules.utils import get_logger

logger = get_logger(__name__)


class WebSocketClient:
    """WebSocket client for connecting to the multi-agent backend"""

    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.message_queue: Queue = Queue()
        self.stream_queue: Queue = Queue()
        self.connected = False
        self.running = False
        self.session_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the WebSocket client in a background thread"""
        if self.running:
            logger.warning("Client already running")
            return

        self.running = True

        # Start event loop in background thread
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

        # Wait for connection
        timeout = 5
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self.connected:
            logger.error("Failed to connect within timeout")
            raise ConnectionError("Could not connect to WebSocket server")

        logger.info(f"WebSocket client connected to {self.uri}")

    def _run_event_loop(self):
        """Run the asyncio event loop in a background thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
        finally:
            self._loop.close()

    async def _connect_and_listen(self):
        """Connect to WebSocket server and listen for messages"""
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                self.connected = True
                logger.info("WebSocket connected")

                # Listen for messages
                while self.running:
                    try:
                        # Check for outgoing messages
                        if not self.message_queue.empty():
                            message = self.message_queue.get()
                            await websocket.send(json.dumps(message))

                        # Receive messages with timeout
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=0.1
                            )
                            data = json.loads(message)
                            self._handle_message(data)
                        except asyncio.TimeoutError:
                            continue

                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        self.connected = False
                        break
                    except Exception as e:
                        logger.error(f"Error in message loop: {e}")
                        break

        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.connected = False

    def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages"""
        category = data.get("category")
        msg_type = data.get("type")
        payload = data.get("data", {})

        # Put stream messages in stream queue
        if category == "agent" and msg_type == "stream":
            self.stream_queue.put(payload)
        else:
            # All other messages go to message queue for processing
            self.message_queue.put(data)

    def send_message(self, message_type: str, payload: Dict[str, Any]):
        """Send a message to the server"""
        if not self.connected:
            raise ConnectionError("Not connected to WebSocket server")

        message = {
            "type": message_type,
            "payload": payload
        }
        self.message_queue.put(message)

    def init_session(self, session_id: str, bmc_data: Optional[Dict[str, List[str]]] = None):
        """Initialize a session"""
        self.session_id = session_id
        payload = {"session_id": session_id}
        if bmc_data:
            payload["bmc_data"] = bmc_data

        self.send_message("init_session", payload)

        # Wait for response
        response = self._wait_for_response("session", "initialized", timeout=5)
        return response

    def update_bmc(self, section: str, items: List[str]):
        """Update a BMC section"""
        if not self.session_id:
            raise ValueError("Session not initialized")

        self.send_message("update_bmc", {
            "session_id": self.session_id,
            "section": section,
            "items": items
        })

        # Wait for response
        response = self._wait_for_response("bmc", "updated", timeout=5)
        return response

    def analyze_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an action using the multi-agent system.
        Returns the final analysis result.
        """
        if not self.session_id:
            raise ValueError("Session not initialized")

        # Clear stream queue
        while not self.stream_queue.empty():
            self.stream_queue.get()

        self.send_message("analyze_action", {
            "session_id": self.session_id,
            "action_data": action_data
        })

        # Wait for completion (this may take a while)
        response = self._wait_for_response("analysis", "completed", timeout=120)
        return response.get("data", {}).get("result", {})

    def get_stream_updates(self) -> List[Dict[str, Any]]:
        """Get all pending stream updates"""
        updates = []
        while not self.stream_queue.empty():
            updates.append(self.stream_queue.get())
        return updates

    def get_bmc(self) -> Dict[str, Any]:
        """Get current BMC state"""
        if not self.session_id:
            raise ValueError("Session not initialized")

        self.send_message("get_bmc", {"session_id": self.session_id})

        response = self._wait_for_response("bmc", "data", timeout=5)
        return response.get("data", {}).get("bmc", {})

    def _wait_for_response(
        self,
        category: str,
        msg_type: str,
        timeout: float = 10
    ) -> Dict[str, Any]:
        """Wait for a specific response message"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.message_queue.empty():
                message = self.message_queue.get()
                if message.get("category") == category and message.get("type") == msg_type:
                    return message

            time.sleep(0.1)

        raise TimeoutError(f"Timeout waiting for {category}:{msg_type}")

    def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("WebSocket client stopped")


# Singleton instance for Streamlit
_client_instance: Optional[WebSocketClient] = None


def get_websocket_client(uri: str = "ws://localhost:8765") -> WebSocketClient:
    """Get or create the singleton WebSocket client instance"""
    global _client_instance

    if _client_instance is None or not _client_instance.connected:
        _client_instance = WebSocketClient(uri)
        try:
            _client_instance.start()
        except Exception as e:
            logger.error(f"Failed to start WebSocket client: {e}")
            logger.info("Falling back to non-streaming mode")
            return None

    return _client_instance


def close_websocket_client():
    """Close the WebSocket client"""
    global _client_instance

    if _client_instance:
        _client_instance.stop()
        _client_instance = None
