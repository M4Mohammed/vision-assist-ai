"""
Headless end-to-end check for the backend frame-relay WebSocket.

Registers (or logs in) a user, opens the captioning WebSocket with the JWT in the query string,
sends one frame, and asserts a caption comes back. Also asserts a bad token is rejected at handshake.

Prereqs: backend on :8080 (Postgres up), fake AI on :9000 (AI_BASE_URL=http://localhost:9000).
Run:  vision-assist-ai/venv/Scripts/python.exe tools/relay_smoke_test.py
"""

import asyncio
import base64
import json
import os
import time
from io import BytesIO

import httpx
import websockets
from PIL import Image

# Defaults target a local run; for the deployed server set e.g.
#   RELAY_API=https://203-0-113-5.nip.io  RELAY_WS=wss://203-0-113-5.nip.io/ws/caption
API = os.getenv("RELAY_API", "http://localhost:8080")
WS = os.getenv("RELAY_WS", "ws://localhost:8080/ws/caption")


def tiny_jpeg_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (32, 32), "navy").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


async def get_token() -> str:
    email = f"relay-{int(time.time())}@example.com"
    password = "Password123!"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"{API}/api/auth/register",
            json={
                "firstName": "Relay",
                "lastName": "Test",
                "phone": "+12025550100",
                "email": email,
                "password": password,
            },
        )
        if r.status_code not in (200, 201):
            print(f"register -> {r.status_code}: {r.text}; trying login")
            r = await c.post(f"{API}/api/auth/login", json={"email": email, "password": password})
        r.raise_for_status()
        return r.json()["accessToken"]


async def main() -> None:
    token = await get_token()
    print("got access token")

    # 1) bad token must be rejected at the handshake
    try:
        async with websockets.connect(f"{WS}?token=not-a-jwt"):
            print("FAIL: bad token was accepted")
            return
    except Exception as e:  # InvalidStatus / handshake 401
        print(f"OK: bad token rejected ({type(e).__name__})")

    # 2) valid token: frame in -> caption out
    async with websockets.connect(f"{WS}?token={token}") as ws:
        await ws.send(json.dumps({"type": "frame", "data": tiny_jpeg_b64(), "ts": 1}))
        msg = await asyncio.wait_for(ws.recv(), timeout=15)
        print("RECV:", msg)
        data = json.loads(msg)
        assert data["type"] == "caption", data
        assert data["caption"], data
        assert data["classification"]["label"] in ("SAFE", "DANGEROUS"), data
        print("OK: caption roundtrip succeeded")


if __name__ == "__main__":
    asyncio.run(main())
