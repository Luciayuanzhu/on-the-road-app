from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from aiohttp import WSMsgType, web
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .config import AppConfig
from .database import create_engine_and_session_factory
from .models import Base
from .schemas import (
    AudioInputChunkPayload,
    BookmarkCreateRequest,
    BookmarkNotePatchRequest,
    Destination,
    IdentityResolutionRequest,
    ImageInputFramePayload,
    LocationUpdatePayload,
    SessionCreateRequest,
    SessionStartPayload,
    SessionStatusPayload,
    SessionStatus,
    SessionUpdateSettingsPayload,
    SettingsUpsertRequest,
    TextInputPayload,
    VisualAssistPayload,
    WebsocketEnvelope,
)
from .services import (
    ApiError,
    BookmarkService,
    IdentityService,
    LiveSessionManager,
    SessionService,
    SettingsService,
    SummaryService,
    generate_silence_chunk,
    parse_payload,
)
from .providers import ContextProvider

CONFIG_KEY = web.AppKey("config", AppConfig)
ENGINE_KEY = web.AppKey("db_engine", Engine)
SESSION_FACTORY_KEY = web.AppKey("session_factory", sessionmaker)
LIVE_MANAGER_KEY = web.AppKey("live_manager", LiveSessionManager)
IDENTITY_SERVICE_KEY = web.AppKey("identity_service", IdentityService)
SETTINGS_SERVICE_KEY = web.AppKey("settings_service", SettingsService)
SESSION_SERVICE_KEY = web.AppKey("session_service", SessionService)
SUMMARY_SERVICE_KEY = web.AppKey("summary_service", SummaryService)
BOOKMARK_SERVICE_KEY = web.AppKey("bookmark_service", BookmarkService)


def model_json(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)


@web.middleware
async def request_id_middleware(request: web.Request, handler):
    request["request_id"] = request.headers.get("X-Request-ID", str(uuid4()))
    response = await handler(request)
    response.headers["X-Request-ID"] = request["request_id"]
    return response


@web.middleware
async def error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except ApiError as exc:
        return web.json_response({"code": exc.code, "message": exc.message}, status=exc.status)
    except ValidationError as exc:
        return web.json_response(
            {"code": "BAD_REQUEST", "message": exc.errors()[0]["msg"]},
            status=400,
        )


def build_websocket_url(request: web.Request, session_id: str, public_base_url: str | None) -> str:
    base_url = public_base_url or str(request.url.origin())
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, f"/v1/live/{session_id}", "", ""))


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def resolve_identity(request: web.Request) -> web.Response:
    body = await request.json()
    payload = IdentityResolutionRequest.model_validate(body)
    response = request.app[IDENTITY_SERVICE_KEY].resolve(payload.device_id)
    return web.json_response(model_json(response))


async def create_session(request: web.Request) -> web.Response:
    body = await request.json()
    payload = SessionCreateRequest.model_validate(body)
    response = request.app[SESSION_SERVICE_KEY].create_session(
        payload,
        "",
    )
    response.websocket_url = build_websocket_url(
        request,
        response.session_id,
        request.app[CONFIG_KEY].public_base_url,
    )
    return web.json_response(model_json(response), status=201)


async def upsert_settings(request: web.Request) -> web.Response:
    body = await request.json()
    payload = SettingsUpsertRequest.model_validate(body)
    request.app[SETTINGS_SERVICE_KEY].upsert(payload.user_id, payload.device_id, payload.settings)
    return web.json_response({"ok": True})


async def get_settings(request: web.Request) -> web.Response:
    user_id = request.match_info["user_id"]
    response = request.app[SETTINGS_SERVICE_KEY].get(user_id)
    return web.json_response(model_json(response))


async def create_bookmark(request: web.Request) -> web.Response:
    body = await request.json()
    payload = BookmarkCreateRequest.model_validate(body)
    response = request.app[BOOKMARK_SERVICE_KEY].create(payload)
    return web.json_response(model_json(response), status=201)


async def list_bookmarks(request: web.Request) -> web.Response:
    user_id = request.query.get("userId")
    if not user_id:
        raise ApiError(400, "BAD_REQUEST", "userId is required")
    response = request.app[BOOKMARK_SERVICE_KEY].list(user_id)
    return web.json_response(model_json(response))


async def get_bookmark(request: web.Request) -> web.Response:
    response = request.app[BOOKMARK_SERVICE_KEY].get(request.match_info["bookmark_id"])
    return web.json_response(model_json(response))


async def patch_bookmark_note(request: web.Request) -> web.Response:
    body = await request.json()
    payload = BookmarkNotePatchRequest.model_validate(body)
    response = request.app[BOOKMARK_SERVICE_KEY].update_note(
        request.match_info["bookmark_id"],
        payload.user_id,
        payload.device_id,
        payload.note_text,
    )
    return web.json_response(model_json(response))


async def live_session(request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    session_id = request.match_info["session_id"]
    live_manager: LiveSessionManager = request.app[LIVE_MANAGER_KEY]
    await live_manager.register(session_id, ws)
    session_service: SessionService = request.app[SESSION_SERVICE_KEY]

    try:
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue

            try:
                envelope = WebsocketEnvelope.model_validate_json(message.data)
            except ValidationError as exc:
                await ws.send_json(
                    {"type": "error", "payload": {"code": "BAD_REQUEST", "message": exc.errors()[0]["msg"]}}
                )
                continue

            event_type = envelope.type
            payload = envelope.payload
            now = datetime.now(UTC)

            try:
                if event_type == "session.start":
                    parsed = parse_payload(SessionStartPayload, payload)
                    session_service.start(parsed.session_id, parsed.user_id, parsed.device_id)
                    await ws.send_json(
                        {
                            "type": "session.status",
                            "payload": model_json(SessionStatusPayload(status=SessionStatus.active)),
                        }
                    )
                elif event_type == "session.update_settings":
                    parsed = parse_payload(SessionUpdateSettingsPayload, payload)
                    hint = await session_service.update_settings(session_id, parsed.settings)
                    await ws.send_json({"type": "context.hint", "payload": model_json(hint)})
                elif event_type == "location.update":
                    parsed = parse_payload(LocationUpdatePayload, payload)
                    assistant_text, hint = await session_service.update_location(session_id, parsed)
                    await ws.send_json({"type": "context.hint", "payload": model_json(hint)})
                    if assistant_text:
                        session_service.record_assistant_text(session_id, assistant_text, parsed.timestamp)
                        await ws.send_json(
                            {
                                "type": "assistant.transcript",
                                "payload": {
                                    "text": assistant_text,
                                    "isFinal": True,
                                    "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                                },
                            }
                        )
                        await ws.send_json(
                            {
                                "type": "assistant.audio.chunk",
                                "payload": {
                                    "mimeType": "audio/pcm",
                                    "data": generate_silence_chunk(),
                                    "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                                },
                            }
                        )
                elif event_type == "destination.update":
                    parsed = parse_payload(Destination, payload)
                    assistant_text = await session_service.update_destination(session_id, parsed)
                    session_service.record_assistant_text(session_id, assistant_text, now)
                    await ws.send_json(
                        {
                            "type": "assistant.transcript",
                            "payload": {
                                "text": assistant_text,
                                "isFinal": True,
                                "timestamp": now.isoformat().replace("+00:00", "Z"),
                            },
                        }
                    )
                elif event_type == "text.input":
                    parsed = parse_payload(TextInputPayload, payload)
                    session_service.record_user_text(session_id, parsed.text, parsed.timestamp)
                    await ws.send_json(
                        {
                            "type": "user.transcript",
                            "payload": {
                                "text": parsed.text,
                                "isFinal": True,
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        }
                    )
                    assistant_text = await session_service.generate_text_reply(session_id, parsed.text)
                    session_service.record_assistant_text(session_id, assistant_text, parsed.timestamp)
                    await ws.send_json(
                        {
                            "type": "assistant.transcript",
                            "payload": {
                                "text": assistant_text,
                                "isFinal": True,
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        }
                    )
                    await ws.send_json(
                        {
                            "type": "assistant.audio.chunk",
                            "payload": {
                                "mimeType": "audio/pcm",
                                "data": generate_silence_chunk(),
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        }
                    )
                elif event_type == "audio.input.chunk":
                    parsed = parse_payload(AudioInputChunkPayload, payload)
                    session_service.record_audio_event(session_id, parsed.timestamp)
                elif event_type == "visual_assist.start":
                    parse_payload(VisualAssistPayload, payload)
                    session_service.set_visual_assist(session_id, True)
                elif event_type == "image.input.frame":
                    parsed = parse_payload(ImageInputFramePayload, payload)
                    if not session_service.is_visual_assist_active(session_id):
                        await ws.send_json(
                            {
                                "type": "error",
                                "payload": {
                                    "code": "VISUAL_ASSIST_INACTIVE",
                                    "message": "visual_assist.start is required before image.input.frame",
                                },
                            }
                        )
                        continue
                    session_service.record_image_event(session_id, parsed.timestamp)
                elif event_type == "visual_assist.stop":
                    parse_payload(VisualAssistPayload, payload)
                    session_service.set_visual_assist(session_id, False)
                elif event_type == "session.pause":
                    session_service.update_status(session_id, SessionStatus.paused)
                    await ws.send_json(
                        {
                            "type": "session.status",
                            "payload": model_json(SessionStatusPayload(status=SessionStatus.paused)),
                        }
                    )
                elif event_type == "session.resume":
                    session_service.update_status(session_id, SessionStatus.active)
                    await ws.send_json(
                        {
                            "type": "session.status",
                            "payload": model_json(SessionStatusPayload(status=SessionStatus.active)),
                        }
                    )
                elif event_type == "session.end":
                    session_service.update_status(session_id, SessionStatus.ended)
                    await ws.send_json(
                        {
                            "type": "session.status",
                            "payload": model_json(SessionStatusPayload(status=SessionStatus.ended)),
                        }
                    )
                    break
                else:
                    await ws.send_json(
                        {
                            "type": "error",
                            "payload": {"code": "UNKNOWN_EVENT", "message": f"Unsupported event {event_type}"},
                        }
                    )
            except ApiError as exc:
                await ws.send_json({"type": "error", "payload": {"code": exc.code, "message": exc.message}})
    finally:
        await live_manager.unregister(session_id, ws)

    return ws


def create_app(config: AppConfig | None = None) -> web.Application:
    config = config or AppConfig.from_env()
    engine, session_factory = create_engine_and_session_factory(config.database_url)
    Base.metadata.create_all(engine)

    app = web.Application(middlewares=[error_middleware, request_id_middleware])
    live_manager = LiveSessionManager()
    identity_service = IdentityService(session_factory)
    settings_service = SettingsService(session_factory, identity_service)
    session_service = SessionService(
        config=config,
        session_factory=session_factory,
        identity_service=identity_service,
        settings_service=settings_service,
        context_provider=ContextProvider(),
    )
    summary_service = SummaryService(config, session_factory, live_manager)
    bookmark_service = BookmarkService(
        session_factory,
        identity_service,
        session_service,
        summary_service,
    )

    app[CONFIG_KEY] = config
    app[ENGINE_KEY] = engine
    app[SESSION_FACTORY_KEY] = session_factory
    app[LIVE_MANAGER_KEY] = live_manager
    app[IDENTITY_SERVICE_KEY] = identity_service
    app[SETTINGS_SERVICE_KEY] = settings_service
    app[SESSION_SERVICE_KEY] = session_service
    app[SUMMARY_SERVICE_KEY] = summary_service
    app[BOOKMARK_SERVICE_KEY] = bookmark_service

    async def dispose_engine(app: web.Application) -> None:
        app[ENGINE_KEY].dispose()

    app.on_cleanup.append(dispose_engine)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/identity/resolve", resolve_identity)
    app.router.add_post("/v1/sessions", create_session)
    app.router.add_post("/v1/settings", upsert_settings)
    app.router.add_get("/v1/settings/{user_id}", get_settings)
    app.router.add_post("/v1/bookmarks", create_bookmark)
    app.router.add_get("/v1/bookmarks", list_bookmarks)
    app.router.add_get("/v1/bookmarks/{bookmark_id}", get_bookmark)
    app.router.add_patch("/v1/bookmarks/{bookmark_id}/note", patch_bookmark_note)
    app.router.add_get("/v1/live/{session_id}", live_session)
    return app
