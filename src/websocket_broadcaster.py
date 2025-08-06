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
        logging.info(
            f"📡 WebSocket Broadcaster prêt à écouter sur ws://{self.host}:{self.port}"
        )

    async def register(self, websocket):
        self.clients.add(websocket)
        logging.info(
            f"New connection: {websocket.remote_address}. Total clients: {len(self.clients)}"
        )

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        logging.info(
            f"Connection closed: {websocket.remote_address}. Total clients: {len(self.clients)}"
        )

    async def broadcast(self, message: dict):
        if not self.clients:
            return

        message_to_send = json.dumps(
            {**message, "broadcast_ts": datetime.now().isoformat()}
        )

        # Create tasks for sending messages
        tasks = [
            self._send_to_client(client, message_to_send)
            for client in self.clients.copy()
        ]
        await asyncio.gather(*tasks)

    async def _send_to_client(self, client, message: str):
        """Safely send a message to a single client, handling potential errors."""
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            logging.warning(
                f"Attempted to send to a closed connection: {client.remote_address}. Client will be removed."
            )
            # The client will be properly unregistered by the main handler
        except Exception as e:
            logging.error(f"Error sending message to {client.remote_address}: {e}")

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            await websocket.wait_closed()
        finally:
            await self.unregister(websocket)

    async def start(self):
        server = await websockets.serve(self.handler, self.host, self.port)
        logging.info("✅ Serveur WebSocket démarré.")
        await server.wait_closed()
