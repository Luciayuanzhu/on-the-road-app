from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from companion_backend.app import create_app
from companion_backend.config import AppConfig


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        app_env="test",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'test.db'}",
        public_base_url="http://localhost:8080",
        summary_delay_seconds=0.15,
        location_response_cooldown_seconds=0,
        gemini_api_key=None,
        gemini_model="gemini-2.5-flash",
        google_places_api_key=None,
        ticketmaster_api_key=None,
        gemini_base_url="https://generativelanguage.googleapis.com",
        google_places_base_url="https://places.googleapis.com",
        ticketmaster_base_url="https://app.ticketmaster.com/discovery/v2",
        google_places_language_code="en",
        gemini_live_model="gemini-2.5-flash-native-audio-preview-12-2025",
        gemini_live_ws_url="wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        gemini_live_audio_idle_seconds=0.05,
        ticketmaster_event_cache_seconds=180,
    )


async def make_client(tmp_path: Path) -> TestClient:
    app = create_app(make_config(tmp_path))
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


async def resolve_identity(client: TestClient, device_id: str = "device_abc") -> dict:
    response = await client.post("/v1/identity/resolve", json={"deviceId": device_id})
    assert response.status == 200
    return await response.json()


async def create_session(client: TestClient, identity: dict, visual_assist_enabled: bool = True) -> dict:
    response = await client.post(
        "/v1/sessions",
        json={
            "userId": identity["userId"],
            "deviceId": identity["deviceId"],
            "settings": {
                "movementMode": "walking",
                "contentPreferences": ["fun_facts", "history_culture"],
                "responseLength": "medium",
                "visualAssistEnabled": visual_assist_enabled,
                "companionStyle": "casual",
            },
        },
    )
    assert response.status == 201
    return await response.json()


async def ws_send(ws, event_type: str, payload: dict) -> None:
    await ws.send_json({"type": event_type, "payload": payload})


async def ws_recv_json(ws) -> dict:
    message = await ws.receive_json(timeout=1)
    return message


def test_identity_resolution_is_stable(tmp_path: Path) -> None:
    async def run() -> None:
        client = await make_client(tmp_path)
        try:
            first = await resolve_identity(client)
            second = await resolve_identity(client)
            assert first["deviceId"] == "device_abc"
            assert first["isNewUser"] is True
            assert second["isNewUser"] is False
            assert first["userId"] == second["userId"]
        finally:
            await client.close()

    asyncio.run(run())


def test_session_works_without_audio_and_visual_assist_is_explicit(tmp_path: Path) -> None:
    async def run() -> None:
        client = await make_client(tmp_path)
        try:
            identity = await resolve_identity(client)
            session = await create_session(client, identity)
            ws = await client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws_send(
                    ws,
                    "session.start",
                    {
                        "userId": identity["userId"],
                        "deviceId": identity["deviceId"],
                        "sessionId": session["sessionId"],
                    },
                )
                status_event = await ws_recv_json(ws)
                assert status_event == {
                    "type": "session.status",
                    "payload": {"status": "active"},
                }

                await ws_send(
                    ws,
                    "image.input.frame",
                    {
                        "mimeType": "image/jpeg",
                        "data": "dGVzdA==",
                        "timestamp": "2026-03-14T18:21:25Z",
                    },
                )
                error_event = await ws_recv_json(ws)
                assert error_event["type"] == "error"
                assert error_event["payload"]["code"] == "VISUAL_ASSIST_INACTIVE"

                await ws_send(
                    ws,
                    "location.update",
                    {
                        "latitude": 41.8819,
                        "longitude": -87.6278,
                        "timestamp": "2026-03-14T18:21:10Z",
                    },
                )
                hint_event = await ws_recv_json(ws)
                transcript_event = await ws_recv_json(ws)
                audio_event = await ws_recv_json(ws)

                assert hint_event["type"] == "context.hint"
                assert hint_event["payload"]["modeApplied"] == "walking"
                assert transcript_event["type"] == "assistant.transcript"
                assert "you’re near" in transcript_event["payload"]["text"].lower()
                assert audio_event["type"] == "assistant.audio.chunk"
                assert audio_event["payload"]["mimeType"] == "audio/pcm"
                assert audio_event["payload"]["data"]

                await ws_send(
                    ws,
                    "visual_assist.start",
                    {"timestamp": "2026-03-14T18:21:24Z"},
                )
                await ws_send(
                    ws,
                    "image.input.frame",
                    {
                        "mimeType": "image/jpeg",
                        "data": "dGVzdA==",
                        "timestamp": "2026-03-14T18:21:25Z",
                    },
                )
                await ws_send(
                    ws,
                    "visual_assist.stop",
                    {"timestamp": "2026-03-14T18:21:40Z"},
                )
                await ws_send(ws, "session.end", {})
                end_event = await ws_recv_json(ws)
                assert end_event == {
                    "type": "session.status",
                    "payload": {"status": "ended"},
                }
            finally:
                await ws.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_bookmark_lifecycle_and_async_summary(tmp_path: Path) -> None:
    async def run() -> None:
        client = await make_client(tmp_path)
        try:
            identity = await resolve_identity(client)
            session = await create_session(client, identity)
            ws = await client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws_send(
                    ws,
                    "session.start",
                    {
                        "userId": identity["userId"],
                        "deviceId": identity["deviceId"],
                        "sessionId": session["sessionId"],
                    },
                )
                await ws_recv_json(ws)

                await ws_send(
                    ws,
                    "text.input",
                    {
                        "text": "Bookmark this area for later.",
                        "timestamp": "2026-03-14T18:21:20Z",
                    },
                )
                await ws_recv_json(ws)
                await ws_recv_json(ws)
                await ws_recv_json(ws)

                create_response = await client.post(
                    "/v1/bookmarks",
                    json={
                        "userId": identity["userId"],
                        "deviceId": identity["deviceId"],
                        "sessionId": session["sessionId"],
                        "timestamp": "2026-03-14T18:25:30Z",
                        "coordinate": {
                            "latitude": 41.8819,
                            "longitude": -87.6278,
                            "timestamp": "2026-03-14T18:25:30Z",
                        },
                    },
                )
                assert create_response.status == 201
                created = await create_response.json()
                assert created["summaryStatus"] == "pending"
                bookmark_id = created["bookmarkId"]

                detail_response = await client.get(f"/v1/bookmarks/{bookmark_id}")
                detail = await detail_response.json()
                assert detail["summaryStatus"] == "pending"
                assert detail["summary"] is None
                assert detail["transcriptExcerpt"]

                patch_response = await client.patch(
                    f"/v1/bookmarks/{bookmark_id}/note",
                    json={
                        "userId": identity["userId"],
                        "deviceId": identity["deviceId"],
                        "noteText": "Check this at sunset.",
                    },
                )
                patched = await patch_response.json()
                assert patched["ok"] is True
                assert patched["noteText"] == "Check this at sunset."

                ready_event = await ws_recv_json(ws)
                assert ready_event == {
                    "type": "bookmark.summary_ready",
                    "payload": {"bookmarkId": bookmark_id},
                }

                ready_response = await client.get(f"/v1/bookmarks/{bookmark_id}")
                ready_detail = await ready_response.json()
                assert ready_detail["summaryStatus"] == "ready"
                assert ready_detail["summary"]["headline"] == ready_detail["title"]
                assert ready_detail["noteText"] == "Check this at sunset."

                list_response = await client.get(
                    "/v1/bookmarks",
                    params={"userId": identity["userId"]},
                )
                listed = await list_response.json()
                assert listed["items"][0]["bookmarkId"] == bookmark_id
                assert listed["items"][0]["summaryStatus"] == "ready"
            finally:
                await ws.close()
        finally:
            await client.close()

    asyncio.run(run())
