from __future__ import annotations

from datetime import UTC, datetime
import logging
from time import perf_counter
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
    GeminiLiveBridge,
    IdentityService,
    LiveSessionManager,
    SessionService,
    SettingsService,
    SummaryService,
    generate_silence_chunk,
    parse_payload,
)
from .providers import ContextProvider, GeminiClient

CONFIG_KEY = web.AppKey("config", AppConfig)
ENGINE_KEY = web.AppKey("db_engine", Engine)
SESSION_FACTORY_KEY = web.AppKey("session_factory", sessionmaker)
LIVE_MANAGER_KEY = web.AppKey("live_manager", LiveSessionManager)
IDENTITY_SERVICE_KEY = web.AppKey("identity_service", IdentityService)
SETTINGS_SERVICE_KEY = web.AppKey("settings_service", SettingsService)
SESSION_SERVICE_KEY = web.AppKey("session_service", SessionService)
SUMMARY_SERVICE_KEY = web.AppKey("summary_service", SummaryService)
BOOKMARK_SERVICE_KEY = web.AppKey("bookmark_service", BookmarkService)
LIVE_BRIDGE_KEY = web.AppKey("live_bridge", GeminiLiveBridge)

logger = logging.getLogger(__name__)


def model_json(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)


def format_log_fields(**fields) -> str:
    return " ".join(f"{key}={value!r}" for key, value in fields.items())


def log_live(event: str, **fields) -> None:
    details = format_log_fields(**fields)
    logger.warning("%s%s", event, f" {details}" if details else "")


def scalar_value(value):
    return getattr(value, "value", value)


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
    log_live(
        "LIVE_SESSION_CREATE",
        request_id=request["request_id"],
        session_id=response.session_id,
        user_id=payload.user_id,
        device_id=payload.device_id,
        status=scalar_value(response.status),
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
    response = await request.app[BOOKMARK_SERVICE_KEY].create(payload)
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

    request_id = request["request_id"]
    session_id = request.match_info["session_id"]
    ws_id = uuid4().hex[:8]
    live_manager: LiveSessionManager = request.app[LIVE_MANAGER_KEY]
    await live_manager.register(session_id, ws)
    session_service: SessionService = request.app[SESSION_SERVICE_KEY]
    live_bridge: GeminiLiveBridge = request.app[LIVE_BRIDGE_KEY]

    log_live(
        "LIVE_WS_OPEN",
        request_id=request_id,
        session_id=session_id,
        ws_id=ws_id,
        remote=request.remote,
        ua=request.headers.get("User-Agent", ""),
        open_connections=len(live_manager.connections.get(session_id, set())),
    )

    async def send_ws_event(event_type: str, payload: dict, **extra_fields) -> None:
        send_started = perf_counter()
        log_live(
            "LIVE_WS_SEND_BEGIN",
            request_id=request_id,
            session_id=session_id,
            ws_id=ws_id,
            ws_event=event_type,
            **extra_fields,
        )
        try:
            await ws.send_json({"type": event_type, "payload": payload})
        except Exception as exc:
            logger.exception(
                "LIVE_WS_SEND_FAIL %s",
                format_log_fields(
                    request_id=request_id,
                    session_id=session_id,
                    ws_id=ws_id,
                    ws_event=event_type,
                    error_type=type(exc).__name__,
                    **extra_fields,
                ),
            )
            raise
        log_live(
            "LIVE_WS_SEND_OK",
            request_id=request_id,
            session_id=session_id,
            ws_id=ws_id,
            ws_event=event_type,
            duration_ms=int((perf_counter() - send_started) * 1000),
            **extra_fields,
        )

    try:
        async for message in ws:
            message_type = getattr(message.type, "name", str(message.type))
            if isinstance(message.data, str):
                message_bytes = len(message.data.encode("utf-8"))
            elif isinstance(message.data, (bytes, bytearray)):
                message_bytes = len(message.data)
            else:
                message_bytes = 0
            log_live(
                "LIVE_WS_FRAME_IN",
                request_id=request_id,
                session_id=session_id,
                ws_id=ws_id,
                aiohttp_type=message_type,
                bytes=message_bytes,
            )
            if message.type != WSMsgType.TEXT:
                log_live(
                    "LIVE_WS_FRAME_IGNORED",
                    request_id=request_id,
                    session_id=session_id,
                    ws_id=ws_id,
                    aiohttp_type=message_type,
                )
                continue

            try:
                envelope = WebsocketEnvelope.model_validate_json(message.data)
            except ValidationError as exc:
                log_live(
                    "LIVE_WS_ENVELOPE_ERROR",
                    request_id=request_id,
                    session_id=session_id,
                    ws_id=ws_id,
                    error_type=type(exc).__name__,
                )
                await send_ws_event(
                    "error",
                    {"code": "BAD_REQUEST", "message": exc.errors()[0]["msg"]},
                    error_code="BAD_REQUEST",
                )
                continue

            event_type = envelope.type
            payload = envelope.payload
            now = datetime.now(UTC)
            log_live(
                "LIVE_WS_ENVELOPE_OK",
                request_id=request_id,
                session_id=session_id,
                ws_id=ws_id,
                ws_event=event_type,
            )

            try:
                if event_type == "session.start":
                    parsed = parse_payload(SessionStartPayload, payload)
                    start_started = perf_counter()
                    log_live(
                        "LIVE_SESSION_START_BEGIN",
                        request_id=request_id,
                        session_id=session_id,
                        ws_id=ws_id,
                        user_id=parsed.user_id,
                        device_id=parsed.device_id,
                        payload_session_id=parsed.session_id,
                    )
                    session_service.start(parsed.session_id, parsed.user_id, parsed.device_id)
                    log_live(
                        "LIVE_SESSION_START_OK",
                        request_id=request_id,
                        session_id=session_id,
                        ws_id=ws_id,
                        db_status=SessionStatus.active.value,
                        duration_ms=int((perf_counter() - start_started) * 1000),
                    )
                    await send_ws_event(
                        "session.status",
                        model_json(SessionStatusPayload(status=SessionStatus.active)),
                        status=SessionStatus.active.value,
                    )
                    await live_bridge.refresh_if_active(session_id)
                elif event_type == "session.update_settings":
                    parsed = parse_payload(SessionUpdateSettingsPayload, payload)
                    hint = await session_service.update_settings(session_id, parsed.settings)
                    await send_ws_event("context.hint", model_json(hint))
                    await live_bridge.restart_session(session_id)
                elif event_type == "location.update":
                    parsed = parse_payload(LocationUpdatePayload, payload)
                    assistant_text, hint = await session_service.update_location(session_id, parsed)
                    await send_ws_event("context.hint", model_json(hint))
                    await live_bridge.refresh_if_active(session_id)
                    if assistant_text:
                        session_service.record_assistant_text(session_id, assistant_text, parsed.timestamp)
                        await send_ws_event(
                            "assistant.transcript",
                            {
                                "text": assistant_text,
                                "isFinal": True,
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        )
                        await send_ws_event(
                            "assistant.audio.chunk",
                            {
                                "mimeType": "audio/pcm",
                                "data": generate_silence_chunk(),
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        )
                elif event_type == "destination.update":
                    parsed = parse_payload(Destination, payload) if payload is not None else None
                    assistant_text = await session_service.update_destination(session_id, parsed)
                    await live_bridge.restart_session(session_id)
                    session_service.record_assistant_text(session_id, assistant_text, now)
                    await send_ws_event(
                        "assistant.transcript",
                        {
                            "text": assistant_text,
                            "isFinal": True,
                            "timestamp": now.isoformat().replace("+00:00", "Z"),
                        },
                    )
                elif event_type == "text.input":
                    parsed = parse_payload(TextInputPayload, payload)
                    fallback_text = await session_service.prepare_text_turn(session_id, parsed.text)
                    await live_bridge.refresh_if_active(session_id)
                    sent_to_live = await live_bridge.send_text_turn(
                        session_id,
                        parsed.text,
                        parsed.timestamp,
                    )
                    if not sent_to_live:
                        # Fallback is only allowed when the active Gemini Live session cannot be
                        # established or write-path fails. This keeps text and audio on the same
                        # real-time source whenever Live is healthy.
                        session_service.record_user_text(session_id, parsed.text, parsed.timestamp)
                        await send_ws_event(
                            "user.transcript",
                            {
                                "text": parsed.text,
                                "isFinal": True,
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        )
                        assistant_text = await session_service.generate_text_reply_fallback(
                            session_id,
                            parsed.text,
                            fallback_text,
                        )
                        session_service.record_assistant_text(session_id, assistant_text, parsed.timestamp)
                        await send_ws_event(
                            "assistant.transcript",
                            {
                                "text": assistant_text,
                                "isFinal": True,
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        )
                        await send_ws_event(
                            "assistant.audio.chunk",
                            {
                                "mimeType": "audio/pcm",
                                "data": generate_silence_chunk(),
                                "timestamp": parsed.timestamp.isoformat().replace("+00:00", "Z"),
                            },
                        )
                elif event_type == "audio.input.chunk":
                    parsed = parse_payload(AudioInputChunkPayload, payload)
                    session_service.record_audio_event(session_id, parsed.timestamp)
                    await live_bridge.send_audio_chunk(
                        session_id,
                        parsed.mime_type,
                        parsed.data,
                    )
                elif event_type == "visual_assist.start":
                    parse_payload(VisualAssistPayload, payload)
                    session_service.set_visual_assist(session_id, True)
                elif event_type == "image.input.frame":
                    parsed = parse_payload(ImageInputFramePayload, payload)
                    if not session_service.is_visual_assist_active(session_id):
                        await send_ws_event(
                            "error",
                            {
                                "code": "VISUAL_ASSIST_INACTIVE",
                                "message": "visual_assist.start is required before image.input.frame",
                            },
                            error_code="VISUAL_ASSIST_INACTIVE",
                        )
                        continue
                    session_service.remember_image_frame(
                        session_id,
                        parsed.mime_type,
                        parsed.data,
                        parsed.timestamp,
                    )
                    session_service.record_image_event(session_id, parsed.timestamp)
                    await live_bridge.send_image_frame(
                        session_id,
                        parsed.mime_type,
                        parsed.data,
                    )
                elif event_type == "visual_assist.stop":
                    parse_payload(VisualAssistPayload, payload)
                    session_service.set_visual_assist(session_id, False)
                    session_service.clear_image_frame(session_id)
                    await live_bridge.restart_session(session_id)
                elif event_type == "session.pause":
                    session_service.update_status(session_id, SessionStatus.paused)
                    await live_bridge.close_session(session_id)
                    await send_ws_event(
                        "session.status",
                        model_json(SessionStatusPayload(status=SessionStatus.paused)),
                        status=SessionStatus.paused.value,
                    )
                elif event_type == "session.resume":
                    session_service.update_status(session_id, SessionStatus.active)
                    await send_ws_event(
                        "session.status",
                        model_json(SessionStatusPayload(status=SessionStatus.active)),
                        status=SessionStatus.active.value,
                    )
                elif event_type == "session.end":
                    session_service.update_status(session_id, SessionStatus.ended)
                    await live_bridge.close_session(session_id)
                    await send_ws_event(
                        "session.status",
                        model_json(SessionStatusPayload(status=SessionStatus.ended)),
                        status=SessionStatus.ended.value,
                    )
                    break
                else:
                    await send_ws_event(
                        "error",
                        {"code": "UNKNOWN_EVENT", "message": f"Unsupported event {event_type}"},
                        error_code="UNKNOWN_EVENT",
                    )
            except ApiError as exc:
                logger.warning(
                    "LIVE_SESSION_EVENT_ERROR %s",
                    format_log_fields(
                        request_id=request_id,
                        session_id=session_id,
                        ws_id=ws_id,
                        ws_event=event_type,
                        error_code=exc.code,
                        error_message=exc.message,
                    ),
                )
                await send_ws_event("error", {"code": exc.code, "message": exc.message}, error_code=exc.code)
    finally:
        await live_bridge.close_session(session_id)
        await live_manager.unregister(session_id, ws)
        log_live(
            "LIVE_WS_CLOSE",
            request_id=request_id,
            session_id=session_id,
            ws_id=ws_id,
            close_code=ws.close_code,
            remaining_connections=len(live_manager.connections.get(session_id, set())),
        )

    return ws


def create_app(config: AppConfig | None = None) -> web.Application:
    config = config or AppConfig.from_env()
    engine, session_factory = create_engine_and_session_factory(config.database_url)
    Base.metadata.create_all(engine)

    app = web.Application(middlewares=[error_middleware, request_id_middleware])
    live_manager = LiveSessionManager()
    identity_service = IdentityService(session_factory)
    settings_service = SettingsService(session_factory, identity_service)
    context_provider = ContextProvider(config)
    gemini_client = GeminiClient(config)
    session_service = SessionService(
        config=config,
        session_factory=session_factory,
        identity_service=identity_service,
        settings_service=settings_service,
        context_provider=context_provider,
        gemini_client=gemini_client,
    )
    summary_service = SummaryService(
        config,
        session_factory,
        live_manager,
        context_provider,
        gemini_client,
    )
    live_bridge = GeminiLiveBridge(config, live_manager, session_service)
    bookmark_service = BookmarkService(
        session_factory,
        identity_service,
        session_service,
        summary_service,
        context_provider,
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
    app[LIVE_BRIDGE_KEY] = live_bridge

    async def dispose_engine(app: web.Application) -> None:
        await app[LIVE_BRIDGE_KEY].close()
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
