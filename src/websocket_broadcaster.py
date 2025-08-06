# src/websocket_broadcaster.py
import asyncio
import websockets
import json
from datetime import datetime
import logging


class WebSocketBroadcaster:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self._server = None
        logging.info(
            f"WebSocket Broadcaster initialized to listen on ws://{self.host}:{self.port}"
        )

    async def _register(self, websocket):
        """Registers a new client connection."""
        self.clients.add(websocket)
        logging.info(
            f"New client connection from {websocket.remote_address}. Total clients: {len(self.clients)}"
        )

    async def _unregister(self, websocket):
        """Unregisters a client connection."""
        self.clients.remove(websocket)
        logging.info(
            f"Client connection closed from {websocket.remote_address}. Total clients: {len(self.clients)}"
        )

    async def broadcast(self, message: dict):
        """Broadcasts a message to all connected clients."""
        if not self.clients:
            return

        message_to_send = json.dumps(
            {**message, "broadcast_ts": datetime.now().isoformat()}
        )

        # Use asyncio.gather to send messages concurrently
        await asyncio.gather(
            *[self._send_to_client(client, message_to_send) for client in self.clients]
        )

    async def _send_to_client(self, client, message: str):
        """Safely sends a message to a single client, handling potential errors."""
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            logging.warning(
                f"Attempted to send to a closed connection: {client.remote_address}."
            )
        except Exception as e:
            logging.error(
                f"Error sending message to {client.remote_address}: {e}", exc_info=True
            )

    async def _handler(self, websocket, path):
        """Handles incoming WebSocket connections."""
        await self._register(websocket)
        try:
            await websocket.wait_closed()
        finally:
            await self._unregister(websocket)

    async def start(self):
        """Starts the WebSocket server."""
        try:
            self._server = await websockets.serve(self._handler, self.host, self.port)
            logging.info(
                f"WebSocket server started successfully on ws://{self.host}:{self.port}."
            )
            await self._server.wait_closed()
        except OSError as e:
            logging.error(
                f"Failed to start WebSocket server on {self.host}:{self.port}. Error: {e}"
            )
            raise
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while starting the server: {e}"
            )
            raise

    def stop(self):
        """Stops the WebSocket server gracefully."""
        if self._server:
            self._server.close()
            logging.info("WebSocket server is shutting down.")
