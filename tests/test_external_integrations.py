from __future__ import annotations

import asyncio
import json
from base64 import b64encode
from pathlib import Path

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from companion_backend.app import create_app
from companion_backend.config import AppConfig


def integration_config(
    tmp_path: Path,
    external_base_url: str,
    *,
    ticketmaster_api_key: str | None = None,
) -> AppConfig:
    return AppConfig(
        app_env="test",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'integration.db'}",
        public_base_url="http://localhost:8080",
        summary_delay_seconds=0.05,
        location_response_cooldown_seconds=0,
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-2.5-flash",
        google_places_api_key="test-places-key",
        ticketmaster_api_key=ticketmaster_api_key,
        gemini_base_url=external_base_url,
        google_places_base_url=external_base_url,
        ticketmaster_base_url=f"{external_base_url}/discovery/v2",
        google_places_language_code="en",
        gemini_live_model="gemini-2.5-flash-native-audio-preview-12-2025",
        gemini_live_ws_url=f"{external_base_url}/ws/live",
        gemini_live_audio_idle_seconds=0.05,
        ticketmaster_event_cache_seconds=120,
    )


async def make_backend_client(
    tmp_path: Path,
    external_base_url: str,
    *,
    ticketmaster_api_key: str | None = None,
) -> TestClient:
    app = create_app(
        integration_config(
            tmp_path,
            external_base_url,
            ticketmaster_api_key=ticketmaster_api_key,
        )
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


async def wait_until(predicate, timeout: float = 1.0, interval: float = 0.02) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition was not met before timeout")


def test_places_ticketmaster_and_gemini_are_used_for_live_flow(tmp_path: Path) -> None:
    async def run() -> None:
        captured_gemini_requests: list[dict] = []
        live_connections: list[dict] = []
        ticketmaster_requests: list[dict] = []

        async def search_nearby(_: web.Request) -> web.Response:
            return web.json_response(
                {
                    "places": [
                        {
                            "id": "place-riverwalk",
                            "displayName": {"text": "Chicago Riverwalk"},
                            "formattedAddress": "Chicago Riverwalk, Chicago, IL",
                            "primaryType": "tourist_attraction",
                            "types": ["tourist_attraction", "point_of_interest"],
                        },
                        {
                            "id": "place-cafe",
                            "displayName": {"text": "River Cafe"},
                            "formattedAddress": "123 River St, Chicago, IL",
                            "primaryType": "cafe",
                            "types": ["cafe", "establishment"],
                        },
                    ]
                }
            )

        async def search_text(_: web.Request) -> web.Response:
            return web.json_response(
                {
                    "places": [
                        {
                            "id": "place-park",
                            "displayName": {"text": "Millennium Park"},
                            "formattedAddress": "Millennium Park, Chicago, IL",
                            "primaryType": "park",
                            "types": ["park", "tourist_attraction"],
                        }
                    ]
                }
            )

        async def place_details(request: web.Request) -> web.Response:
            place_id = request.match_info["place_id"]
            name = "Chicago Riverwalk" if place_id == "place-riverwalk" else "Millennium Park"
            primary_type = "tourist_attraction" if place_id == "place-riverwalk" else "park"
            return web.json_response(
                {
                    "id": place_id,
                    "displayName": {"text": name},
                    "formattedAddress": f"{name}, Chicago, IL",
                    "primaryType": primary_type,
                    "types": [primary_type, "point_of_interest"],
                }
            )

        async def gemini_generate(request: web.Request) -> web.Response:
            payload = await request.json()
            captured_gemini_requests.append(payload)
            generation_config = payload.get("generationConfig", {})
            if generation_config.get("responseMimeType") == "application/json":
                summary = {
                    "headline": "Riverwalk snapshot",
                    "shortSummary": "A saved moment near the Chicago Riverwalk.",
                    "whatWasNearby": ["Chicago Riverwalk", "River Cafe"],
                    "activitiesMentioned": ["landmarks and highlights", "cafes and quick breaks"],
                    "tags": ["riverwalk", "city", "water"],
                }
                text = json.dumps(summary)
            else:
                parts = payload["contents"][0]["parts"]
                has_image = any("inline_data" in part for part in parts)
                prompt_text = " ".join(part.get("text", "") for part in parts)
                if has_image:
                    text = "From the camera view, that looks like the Chicago Riverwalk area."
                elif "location changed" in prompt_text:
                    text = "You are near Chicago Riverwalk with cafes and landmarks close by."
                elif "destination was updated" in prompt_text:
                    text = "Destination updated to Millennium Park; I will bias toward nearby landmarks."
                else:
                    text = "That is the Chicago Riverwalk corridor with local highlights nearby."
            return web.json_response(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": text}],
                            }
                        }
                    ]
                }
            )

        async def ticketmaster_events(request: web.Request) -> web.Response:
            ticketmaster_requests.append(dict(request.query))
            return web.json_response(
                {
                    "_embedded": {
                        "events": [
                            {
                                "name": "Jazz on the River",
                                "dates": {
                                    "start": {
                                        "localDate": "2026-03-14",
                                        "localTime": "19:00:00",
                                    }
                                },
                                "_embedded": {"venues": [{"name": "River Theater"}]},
                                "classifications": [
                                    {
                                        "segment": {"name": "Music"},
                                        "genre": {"name": "Jazz"},
                                    }
                                ],
                            },
                            {
                                "name": "Architecture Night Walk",
                                "dates": {
                                    "start": {
                                        "localDate": "2026-03-15",
                                        "localTime": "18:30:00",
                                    }
                                },
                                "_embedded": {"venues": [{"name": "Downtown Meetup Point"}]},
                                "classifications": [
                                    {
                                        "segment": {"name": "Arts & Theatre"},
                                        "genre": {"name": "Miscellaneous Theatre"},
                                    }
                                ],
                            },
                        ]
                    }
                }
            )

        async def live_ws(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            connection_state: dict[str, list] = {"messages": []}
            live_connections.append(connection_state)
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                payload = json.loads(message.data)
                connection_state["messages"].append(payload)
                if "setup" in payload:
                    await ws.send_json({"setupComplete": {}})
                elif payload.get("clientContent"):
                    await ws.send_json(
                        {
                            "serverContent": {
                                "outputTranscription": {
                                    "text": "From the camera view, that looks like the Chicago Riverwalk area."
                                },
                                "modelTurn": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "mimeType": "audio/pcm",
                                                "data": b64encode(b"live-text-audio").decode("ascii"),
                                            }
                                        }
                                    ]
                                },
                                "turnComplete": True,
                                "generationComplete": True,
                            }
                        }
                    )
                elif payload.get("realtimeInput", {}).get("audioStreamEnd"):
                    await ws.send_json(
                        {
                            "serverContent": {
                                "inputTranscription": {"text": "What is around me?"},
                                "outputTranscription": {"text": "You are near Chicago Riverwalk."},
                                "modelTurn": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "mimeType": "audio/pcm",
                                                "data": b64encode(b"audio").decode("ascii"),
                                            }
                                        }
                                    ]
                                },
                                "turnComplete": True,
                                "generationComplete": True,
                            }
                        }
                    )
            return ws

        external_app = web.Application()
        external_app.router.add_post("/v1/places:searchNearby", search_nearby)
        external_app.router.add_post("/v1/places:searchText", search_text)
        external_app.router.add_get("/v1/places/{place_id}", place_details)
        external_app.router.add_get("/discovery/v2/events.json", ticketmaster_events)
        external_app.router.add_post("/v1beta/models/{model}:generateContent", gemini_generate)
        external_app.router.add_get("/ws/live", live_ws)

        external_server = TestServer(external_app)
        await external_server.start_server()
        backend_client = await make_backend_client(
            tmp_path,
            str(external_server.make_url("")).rstrip("/"),
            ticketmaster_api_key="test-ticketmaster-key",
        )
        try:
            identity_response = await backend_client.post(
                "/v1/identity/resolve",
                json={"deviceId": "device_external"},
            )
            identity = await identity_response.json()

            session_response = await backend_client.post(
                "/v1/sessions",
                json={
                    "userId": identity["userId"],
                    "deviceId": identity["deviceId"],
                    "settings": {
                        "movementMode": "walking",
                        "contentPreferences": ["fun_facts", "nature_scenery"],
                        "responseLength": "medium",
                        "visualAssistEnabled": True,
                        "companionStyle": "casual",
                    },
                },
            )
            session = await session_response.json()

            ws = await backend_client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws.send_json(
                    {
                        "type": "session.start",
                        "payload": {
                            "userId": identity["userId"],
                            "deviceId": identity["deviceId"],
                            "sessionId": session["sessionId"],
                        },
                    }
                )
                await ws.receive_json(timeout=1)

                await ws.send_json(
                    {
                        "type": "location.update",
                        "payload": {
                            "latitude": 41.8819,
                            "longitude": -87.6278,
                            "timestamp": "2026-03-14T18:21:10Z",
                        },
                    }
                )
                hint_event = await ws.receive_json(timeout=1)
                transcript_event = await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)
                assert hint_event["payload"]["nearbyPlaceName"] == "Chicago Riverwalk"
                assert "Chicago Riverwalk" in transcript_event["payload"]["text"]
                await wait_until(
                    lambda: any(
                        "setup" in message
                        and "nearby_events=2026-03-14 19:00: Jazz on the River at River Theater"
                        in message["setup"]["systemInstruction"]["parts"][0]["text"]
                        for connection in live_connections
                        for message in connection["messages"]
                    )
                )
                assert len(ticketmaster_requests) == 1

                await ws.send_json(
                    {
                        "type": "location.update",
                        "payload": {
                            "latitude": 41.8821,
                            "longitude": -87.6279,
                            "timestamp": "2026-03-14T18:21:14Z",
                        },
                    }
                )
                await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)
                assert len(ticketmaster_requests) == 1

                await ws.send_json(
                    {
                        "type": "visual_assist.start",
                        "payload": {"timestamp": "2026-03-14T18:21:24Z"},
                    }
                )
                await ws.send_json(
                    {
                        "type": "image.input.frame",
                        "payload": {
                            "mimeType": "image/jpeg",
                            "data": "dGVzdA==",
                            "timestamp": "2026-03-14T18:21:25Z",
                        },
                    }
                )
                await ws.send_json(
                    {
                        "type": "text.input",
                        "payload": {
                            "text": "What am I looking at?",
                            "timestamp": "2026-03-14T18:21:30Z",
                        },
                    }
                )
                user_echo = await ws.receive_json(timeout=1)
                visual_transcript = await ws.receive_json(timeout=1)
                visual_audio = await ws.receive_json(timeout=1)
                assert user_echo["type"] == "user.transcript"
                assert "camera view" in visual_transcript["payload"]["text"]
                assert visual_audio["type"] == "assistant.audio.chunk"

                assert any(
                    "clientContent" in message
                    for connection in live_connections
                    for message in connection["messages"]
                )
                assert not any(
                    "What am I looking at?" in json.dumps(request)
                    for request in captured_gemini_requests
                )

                connections_before_settings = len(live_connections)
                await ws.send_json(
                    {
                        "type": "session.update_settings",
                        "payload": {
                            "settings": {
                                "movementMode": "ride",
                                "contentPreferences": ["fun_facts", "nature_scenery"],
                                "responseLength": "medium",
                                "visualAssistEnabled": True,
                                "companionStyle": "casual",
                            }
                        },
                    }
                )
                settings_hint = await ws.receive_json(timeout=1)
                assert settings_hint["type"] == "context.hint"
                await wait_until(lambda: len(live_connections) > connections_before_settings)
                ride_connections = [
                    connection
                    for connection in live_connections[connections_before_settings:]
                    if any(
                        "setup" in message
                        and "movement_mode=ride"
                        in message["setup"]["systemInstruction"]["parts"][0]["text"]
                        for message in connection["messages"]
                    )
                ]
                assert ride_connections
                assert any(
                    message.get("realtimeInput", {}).get("video", {}).get("data") == "dGVzdA=="
                    for connection in ride_connections
                    for message in connection["messages"]
                )

                connections_before_destination = len(live_connections)
                await ws.send_json(
                    {
                        "type": "destination.update",
                        "payload": {
                            "label": "Millennium Park",
                            "latitude": 41.8826,
                            "longitude": -87.6226,
                            "placeId": "place-park",
                        },
                    }
                )
                destination_transcript = await ws.receive_json(timeout=1)
                assert destination_transcript["type"] == "assistant.transcript"
                await wait_until(lambda: len(live_connections) > connections_before_destination)
                destination_connections = [
                    connection
                    for connection in live_connections[connections_before_destination:]
                    if any(
                        "setup" in message
                        and "destination=Millennium Park"
                        in message["setup"]["systemInstruction"]["parts"][0]["text"]
                        for message in connection["messages"]
                    )
                ]
                assert destination_connections
                assert any(
                    message.get("realtimeInput", {}).get("video", {}).get("data") == "dGVzdA=="
                    for connection in destination_connections
                    for message in connection["messages"]
                )

                connections_before_clear = len(live_connections)
                await ws.send_json(
                    {
                        "type": "destination.update",
                        "payload": None,
                    }
                )
                clear_transcript = await ws.receive_json(timeout=1)
                assert clear_transcript["type"] == "assistant.transcript"
                await wait_until(lambda: len(live_connections) > connections_before_clear)
                cleared_connections = [
                    connection
                    for connection in live_connections[connections_before_clear:]
                    if any(
                        "setup" in message
                        and "destination=no destination set"
                        in message["setup"]["systemInstruction"]["parts"][0]["text"]
                        for message in connection["messages"]
                    )
                ]
                assert cleared_connections
                assert any(
                    message.get("realtimeInput", {}).get("video", {}).get("data") == "dGVzdA=="
                    for connection in cleared_connections
                    for message in connection["messages"]
                )

                connections_before_stop_refresh = len(live_connections)
                await ws.send_json(
                    {
                        "type": "visual_assist.stop",
                        "payload": {"timestamp": "2026-03-14T18:21:40Z"},
                    }
                )
                await ws.send_json(
                    {
                        "type": "session.update_settings",
                        "payload": {
                            "settings": {
                                "movementMode": "walking",
                                "contentPreferences": ["fun_facts"],
                                "responseLength": "medium",
                                "visualAssistEnabled": False,
                                "companionStyle": "casual",
                            }
                        },
                    }
                )
                await ws.receive_json(timeout=1)
                await wait_until(
                    lambda: any(
                        any(
                            "setup" in message
                            and "movement_mode=walking"
                            in message["setup"]["systemInstruction"]["parts"][0]["text"]
                            for message in connection["messages"]
                        )
                        and not any(
                            "video" in message.get("realtimeInput", {})
                            for message in connection["messages"]
                        )
                        for connection in live_connections[connections_before_stop_refresh:]
                    )
                )

                bookmark_response = await backend_client.post(
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
                bookmark = await bookmark_response.json()
                assert bookmark["title"] == "Near Chicago Riverwalk"

                ready_event = await ws.receive_json(timeout=1)
                assert ready_event["type"] == "bookmark.summary_ready"

                detail_response = await backend_client.get(
                    f"/v1/bookmarks/{bookmark['bookmarkId']}"
                )
                detail = await detail_response.json()
                assert detail["summary"]["headline"] == "Riverwalk snapshot"
                assert detail["summary"]["whatWasNearby"][0] == "Chicago Riverwalk"

                assert any(
                    any("inline_data" in part for part in request["contents"][0]["parts"])
                    for request in captured_gemini_requests
                    if request.get("generationConfig", {}).get("responseMimeType") != "application/json"
                )
                assert live_connections
            finally:
                await ws.close()
        finally:
            await backend_client.close()
            await external_server.close()

    asyncio.run(run())


def test_audio_chunks_roundtrip_through_gemini_live(tmp_path: Path) -> None:
    async def run() -> None:
        async def live_ws(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                payload = json.loads(message.data)
                if "setup" in payload:
                    await ws.send_json({"setupComplete": {}})
                elif payload.get("realtimeInput", {}).get("audioStreamEnd"):
                    await ws.send_json(
                        {
                            "serverContent": {
                                "inputTranscription": {"text": "Tell me about this area"},
                                "outputTranscription": {"text": "You are passing the Chicago Riverwalk."},
                                "modelTurn": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "mimeType": "audio/pcm",
                                                "data": b64encode(b"live-audio").decode("ascii"),
                                            }
                                        }
                                    ]
                                },
                                "turnComplete": True,
                                "generationComplete": True,
                            }
                        }
                    )
            return ws

        external_app = web.Application()
        external_app.router.add_get("/ws/live", live_ws)
        external_server = TestServer(external_app)
        await external_server.start_server()
        client = await make_backend_client(
            tmp_path,
            str(external_server.make_url("")).rstrip("/"),
        )
        try:
            identity_response = await client.post(
                "/v1/identity/resolve",
                json={"deviceId": "device_live_audio"},
            )
            identity = await identity_response.json()
            session_response = await client.post(
                "/v1/sessions",
                json={
                    "userId": identity["userId"],
                    "deviceId": identity["deviceId"],
                    "settings": {
                        "movementMode": "walking",
                        "contentPreferences": ["fun_facts"],
                        "responseLength": "medium",
                        "visualAssistEnabled": False,
                        "companionStyle": "casual",
                    },
                },
            )
            session = await session_response.json()
            ws = await client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws.send_json(
                    {
                        "type": "session.start",
                        "payload": {
                            "userId": identity["userId"],
                            "deviceId": identity["deviceId"],
                            "sessionId": session["sessionId"],
                        },
                    }
                )
                await ws.receive_json(timeout=1)

                await ws.send_json(
                    {
                        "type": "location.update",
                        "payload": {
                            "latitude": 41.8819,
                            "longitude": -87.6278,
                            "timestamp": "2026-03-14T18:21:10Z",
                        },
                    }
                )
                await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)

                await ws.send_json(
                    {
                        "type": "audio.input.chunk",
                        "payload": {
                            "mimeType": "audio/pcm",
                            "data": b64encode(b"user-audio").decode("ascii"),
                            "timestamp": "2026-03-14T18:21:21Z",
                        },
                    }
                )

                user_transcript = await ws.receive_json(timeout=2)
                assistant_transcript = await ws.receive_json(timeout=2)
                assistant_audio = await ws.receive_json(timeout=2)

                assert user_transcript["type"] == "user.transcript"
                assert user_transcript["payload"]["text"] == "Tell me about this area"
                assert assistant_transcript["type"] == "assistant.transcript"
                assert "Chicago Riverwalk" in assistant_transcript["payload"]["text"]
                assert assistant_audio["type"] == "assistant.audio.chunk"
                assert assistant_audio["payload"]["data"] == b64encode(b"live-audio").decode("ascii")
            finally:
                await ws.close()
        finally:
            await client.close()
            await external_server.close()

    asyncio.run(run())


def test_ticketmaster_missing_key_degrades_without_breaking_session(tmp_path: Path) -> None:
    async def run() -> None:
        ticketmaster_requests = 0

        async def search_nearby(_: web.Request) -> web.Response:
            return web.json_response(
                {
                    "places": [
                        {
                            "id": "place-riverwalk",
                            "displayName": {"text": "Chicago Riverwalk"},
                            "formattedAddress": "Chicago Riverwalk, Chicago, IL",
                            "primaryType": "tourist_attraction",
                            "types": ["tourist_attraction", "point_of_interest"],
                        }
                    ]
                }
            )

        async def gemini_generate(_: web.Request) -> web.Response:
            return web.json_response(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "You are near Chicago Riverwalk with local highlights nearby."}],
                            }
                        }
                    ]
                }
            )

        async def ticketmaster_events(_: web.Request) -> web.Response:
            nonlocal ticketmaster_requests
            ticketmaster_requests += 1
            return web.json_response({})

        async def live_ws(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                payload = json.loads(message.data)
                if "setup" in payload:
                    await ws.send_json({"setupComplete": {}})
                elif payload.get("clientContent"):
                    await ws.send_json(
                        {
                            "serverContent": {
                                "outputTranscription": {"text": "Live text still works without events context."},
                                "turnComplete": True,
                                "generationComplete": True,
                            }
                        }
                    )
            return ws

        external_app = web.Application()
        external_app.router.add_post("/v1/places:searchNearby", search_nearby)
        external_app.router.add_get("/discovery/v2/events.json", ticketmaster_events)
        external_app.router.add_post("/v1beta/models/{model}:generateContent", gemini_generate)
        external_app.router.add_get("/ws/live", live_ws)

        external_server = TestServer(external_app)
        await external_server.start_server()
        client = await make_backend_client(
            tmp_path,
            str(external_server.make_url("")).rstrip("/"),
            ticketmaster_api_key=None,
        )
        try:
            identity_response = await client.post(
                "/v1/identity/resolve",
                json={"deviceId": "device_ticketmaster_missing"},
            )
            identity = await identity_response.json()
            session_response = await client.post(
                "/v1/sessions",
                json={
                    "userId": identity["userId"],
                    "deviceId": identity["deviceId"],
                    "settings": {
                        "movementMode": "walking",
                        "contentPreferences": ["events_activities"],
                        "responseLength": "medium",
                        "visualAssistEnabled": False,
                        "companionStyle": "casual",
                    },
                },
            )
            session = await session_response.json()
            ws = await client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws.send_json(
                    {
                        "type": "session.start",
                        "payload": {
                            "userId": identity["userId"],
                            "deviceId": identity["deviceId"],
                            "sessionId": session["sessionId"],
                        },
                    }
                )
                await ws.receive_json(timeout=1)

                await ws.send_json(
                    {
                        "type": "location.update",
                        "payload": {
                            "latitude": 41.8819,
                            "longitude": -87.6278,
                            "timestamp": "2026-03-14T18:21:10Z",
                        },
                    }
                )
                hint_event = await ws.receive_json(timeout=1)
                transcript_event = await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)

                assert hint_event["type"] == "context.hint"
                assert transcript_event["type"] == "assistant.transcript"
                assert ticketmaster_requests == 0
            finally:
                await ws.close()
        finally:
            await client.close()
            await external_server.close()

    asyncio.run(run())


def test_ticketmaster_failure_degrades_without_breaking_session(tmp_path: Path) -> None:
    async def run() -> None:
        ticketmaster_requests = 0

        async def search_nearby(_: web.Request) -> web.Response:
            return web.json_response(
                {
                    "places": [
                        {
                            "id": "place-riverwalk",
                            "displayName": {"text": "Chicago Riverwalk"},
                            "formattedAddress": "Chicago Riverwalk, Chicago, IL",
                            "primaryType": "tourist_attraction",
                            "types": ["tourist_attraction", "point_of_interest"],
                        }
                    ]
                }
            )

        async def gemini_generate(_: web.Request) -> web.Response:
            return web.json_response(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "You are near Chicago Riverwalk even without event context."}],
                            }
                        }
                    ]
                }
            )

        async def ticketmaster_events(_: web.Request) -> web.Response:
            nonlocal ticketmaster_requests
            ticketmaster_requests += 1
            raise web.HTTPInternalServerError(text="upstream failure")

        async def live_ws(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                payload = json.loads(message.data)
                if "setup" in payload:
                    await ws.send_json({"setupComplete": {}})
                elif payload.get("clientContent"):
                    await ws.send_json(
                        {
                            "serverContent": {
                                "outputTranscription": {"text": "Text still routes through Gemini Live."},
                                "turnComplete": True,
                                "generationComplete": True,
                            }
                        }
                    )
            return ws

        external_app = web.Application()
        external_app.router.add_post("/v1/places:searchNearby", search_nearby)
        external_app.router.add_get("/discovery/v2/events.json", ticketmaster_events)
        external_app.router.add_post("/v1beta/models/{model}:generateContent", gemini_generate)
        external_app.router.add_get("/ws/live", live_ws)

        external_server = TestServer(external_app)
        await external_server.start_server()
        client = await make_backend_client(
            tmp_path,
            str(external_server.make_url("")).rstrip("/"),
            ticketmaster_api_key="test-ticketmaster-key",
        )
        try:
            identity_response = await client.post(
                "/v1/identity/resolve",
                json={"deviceId": "device_ticketmaster_failure"},
            )
            identity = await identity_response.json()
            session_response = await client.post(
                "/v1/sessions",
                json={
                    "userId": identity["userId"],
                    "deviceId": identity["deviceId"],
                    "settings": {
                        "movementMode": "walking",
                        "contentPreferences": ["events_activities"],
                        "responseLength": "medium",
                        "visualAssistEnabled": False,
                        "companionStyle": "casual",
                    },
                },
            )
            session = await session_response.json()
            ws = await client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws.send_json(
                    {
                        "type": "session.start",
                        "payload": {
                            "userId": identity["userId"],
                            "deviceId": identity["deviceId"],
                            "sessionId": session["sessionId"],
                        },
                    }
                )
                await ws.receive_json(timeout=1)

                await ws.send_json(
                    {
                        "type": "location.update",
                        "payload": {
                            "latitude": 41.8819,
                            "longitude": -87.6278,
                            "timestamp": "2026-03-14T18:21:10Z",
                        },
                    }
                )
                hint_event = await ws.receive_json(timeout=1)
                transcript_event = await ws.receive_json(timeout=1)
                await ws.receive_json(timeout=1)
                assert hint_event["type"] == "context.hint"
                assert transcript_event["type"] == "assistant.transcript"
                assert ticketmaster_requests == 1

                await ws.send_json(
                    {
                        "type": "text.input",
                        "payload": {
                            "text": "Any events nearby?",
                            "timestamp": "2026-03-14T18:21:30Z",
                        },
                    }
                )
                user_echo = await ws.receive_json(timeout=1)
                assistant_transcript = await ws.receive_json(timeout=1)
                assert user_echo["type"] == "user.transcript"
                assert assistant_transcript["type"] == "assistant.transcript"
                assert "Text still routes through Gemini Live." in assistant_transcript["payload"]["text"]
            finally:
                await ws.close()
        finally:
            await client.close()
            await external_server.close()

    asyncio.run(run())


def test_text_input_falls_back_when_live_is_unavailable(tmp_path: Path) -> None:
    async def run() -> None:
        external_app = web.Application()
        external_server = TestServer(external_app)
        await external_server.start_server()
        client = await make_backend_client(
            tmp_path,
            str(external_server.make_url("")).rstrip("/"),
        )
        try:
            identity_response = await client.post(
                "/v1/identity/resolve",
                json={"deviceId": "device_text_fallback"},
            )
            identity = await identity_response.json()
            session_response = await client.post(
                "/v1/sessions",
                json={
                    "userId": identity["userId"],
                    "deviceId": identity["deviceId"],
                    "settings": {
                        "movementMode": "walking",
                        "contentPreferences": ["fun_facts"],
                        "responseLength": "medium",
                        "visualAssistEnabled": False,
                        "companionStyle": "casual",
                    },
                },
            )
            session = await session_response.json()
            ws = await client.ws_connect(f"/v1/live/{session['sessionId']}")
            try:
                await ws.send_json(
                    {
                        "type": "session.start",
                        "payload": {
                            "userId": identity["userId"],
                            "deviceId": identity["deviceId"],
                            "sessionId": session["sessionId"],
                        },
                    }
                )
                await ws.receive_json(timeout=1)

                await ws.send_json(
                    {
                        "type": "text.input",
                        "payload": {
                            "text": "Fallback text turn",
                            "timestamp": "2026-03-14T18:21:30Z",
                        },
                    }
                )
                user_echo = await ws.receive_json(timeout=1)
                assistant_transcript = await ws.receive_json(timeout=1)
                assistant_audio = await ws.receive_json(timeout=1)
                assert user_echo["type"] == "user.transcript"
                assert user_echo["payload"]["text"] == "Fallback text turn"
                assert assistant_transcript["type"] == "assistant.transcript"
                assert assistant_audio["type"] == "assistant.audio.chunk"
            finally:
                await ws.close()
        finally:
            await client.close()
            await external_server.close()

    asyncio.run(run())
