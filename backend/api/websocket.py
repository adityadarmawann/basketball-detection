"""
WebSocketManager — per-match connection registry with Redis pub/sub forwarding.

Usage
-----
manager = WebSocketManager()

# in WS endpoint
await manager.connect(websocket, match_id)
try:
    await manager.listen_redis(match_id, channel=f"match:{match_id}")
except WebSocketDisconnect:
    await manager.disconnect(websocket, match_id)
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Tracks active WebSocket connections grouped by match_id.
    Forwards Redis pub/sub messages to all subscribers of that match.
    """

    def __init__(self) -> None:
        # match_id → set[WebSocket]
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    # ── Connection lifecycle ───────────────────────────────────────────────

    async def connect(self, websocket: WebSocket, match_id: str) -> None:
        await websocket.accept()
        self._connections[match_id].add(websocket)
        logger.info("WS connected  match=%s  total=%d",
                    match_id, len(self._connections[match_id]))

    async def disconnect(self, websocket: WebSocket, match_id: str) -> None:
        self._connections[match_id].discard(websocket)
        if not self._connections[match_id]:
            del self._connections[match_id]
        logger.info("WS disconnected  match=%s  remaining=%d",
                    match_id, len(self._connections.get(match_id, set())))

    # ── Broadcast ─────────────────────────────────────────────────────────

    async def broadcast(self, match_id: str, message: str | dict) -> None:
        """
        Send message to every WebSocket subscribed to match_id.
        Dead connections are silently removed.
        """
        text = message if isinstance(message, str) else json.dumps(message, default=str)
        dead: list[WebSocket] = []

        for ws in list(self._connections.get(match_id, set())):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._connections[match_id].discard(ws)

    # ── Legacy broadcast (no match_id) for backwards compat ──────────────

    async def broadcast_all(self, message: str | dict) -> None:
        """Broadcast to every connected client regardless of match."""
        for match_id in list(self._connections):
            await self.broadcast(match_id, message)

    # ── Redis pub/sub listener ────────────────────────────────────────────

    async def listen_redis(
        self,
        match_id: str,
        channel: str,
        redis_client=None,
    ) -> None:
        """
        Subscribe to a Redis channel and forward messages to all WebSocket
        clients watching match_id.  Auto-reconnects on transient errors.

        Uses get_message(timeout=5) instead of listen() to avoid socket-level
        TimeoutError spam that occurs when no messages arrive for a while.
        get_message uses asyncio.wait_for internally and returns None on idle —
        no exception, no reconnect spam.

        If redis_client is None, the loop is a no-op (dev mode without Redis).
        """
        if redis_client is None:
            logger.debug("listen_redis: no Redis client, skipping for match=%s", match_id)
            await asyncio.Future()   # block forever until cancelled
            return

        while True:
            pubsub = redis_client.pubsub()
            try:
                await pubsub.subscribe(channel)
                logger.info("Redis sub: channel=%s  match=%s", channel, match_id)

                while True:
                    # Block up to 5 s waiting for a message; returns None on idle.
                    # Never raises TimeoutError — avoids reconnect spam on quiet channels.
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=5.0
                    )
                    if message is None:
                        await asyncio.sleep(0.02)   # yield to event loop briefly
                        continue
                    data = message.get("data", "")
                    if isinstance(data, bytes):
                        data = data.decode()
                    await self.broadcast(match_id, data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "Redis listener error match=%s: %s — reconnecting in 2 s",
                    match_id, e,
                )
                await asyncio.sleep(2.0)
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()
                except Exception:
                    pass
