from __future__ import annotations

import base64
import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .config import AppConfig
from .database import utcnow
from .models import Bookmark, BookmarkSummary, SessionRecord, TranscriptChunk, User, UserSetting
from .prompt_policy import (
    build_destination_cleared_response,
    build_destination_response,
    build_location_response,
    build_system_prompt,
    build_text_response,
)
from .providers import ContextProvider, GeminiClient, MediaFrame
from .schemas import (
    DEFAULT_SETTINGS,
    BookmarkCreateRequest,
    BookmarkCreateResponse,
    BookmarkDetailResponse,
    BookmarkListItem,
    BookmarkListResponse,
    BookmarkNotePatchResponse,
    BookmarkSummaryPayload,
    ContextHintPayload,
    Destination,
    IdentityResolutionResponse,
    LocationUpdatePayload,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatus,
    SettingsGetResponse,
    SummaryStatus,
    TranscriptExcerptItem,
    UserSettings,
)

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def datetime_to_zulu(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def generate_silence_chunk(duration_ms: int = 250, sample_rate: int = 16000) -> str:
    frame_count = int(sample_rate * duration_ms / 1000)
    pcm = b"\x00\x00" * frame_count
    return base64.b64encode(pcm).decode("ascii")


@dataclass(slots=True)
class LiveSessionManager:
    connections: dict[str, set[web.WebSocketResponse]]

    def __init__(self) -> None:
        self.connections = {}

    async def register(self, session_id: str, ws: web.WebSocketResponse) -> None:
        self.connections.setdefault(session_id, set()).add(ws)

    async def unregister(self, session_id: str, ws: web.WebSocketResponse) -> None:
        session_connections = self.connections.get(session_id)
        if not session_connections:
            return
        session_connections.discard(ws)
        if not session_connections:
            self.connections.pop(session_id, None)

    async def emit(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        for ws in list(self.connections.get(session_id, set())):
            if not ws.closed:
                await ws.send_json({"type": event_type, "payload": payload})


@dataclass(slots=True)
class GeminiLiveConnection:
    ws: Any
    prompt_state: str
    setup_complete: asyncio.Event
    reader_task: asyncio.Task[None]
    audio_end_task: asyncio.Task[None] | None = None
    last_user_transcript: str = ""
    last_user_transcript_recorded: str = ""
    last_assistant_transcript: str = ""
    last_assistant_transcript_recorded: str = ""
    audio_chunk_count: int = 0


class IdentityService:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def resolve(self, device_id: str) -> IdentityResolutionResponse:
        with self.session_factory() as db:
            user = db.scalar(select(User).where(User.device_id == device_id))
            is_new = False
            if user is None:
                is_new = True
                user = User(id=new_id("user"), device_id=device_id)
                db.add(user)
                db.commit()
            return IdentityResolutionResponse(
                deviceId=device_id,
                userId=user.id,
                isNewUser=is_new,
            )

    def require_user(self, db: Session, user_id: str, device_id: str) -> User:
        user = db.scalar(select(User).where(User.id == user_id))
        if user is None or user.device_id != device_id:
            raise ApiError(404, "IDENTITY_NOT_FOUND", "User and device mapping was not found")
        return user


class SettingsService:
    def __init__(self, session_factory: sessionmaker, identity_service: IdentityService):
        self.session_factory = session_factory
        self.identity_service = identity_service

    def get_or_create(self, db: Session, user_id: str, device_id: str) -> UserSetting:
        existing = db.scalar(select(UserSetting).where(UserSetting.user_id == user_id))
        if existing is not None:
            return existing
        settings = UserSetting(
            user_id=user_id,
            device_id=device_id,
            movement_mode=DEFAULT_SETTINGS.movement_mode,
            content_preferences=list(DEFAULT_SETTINGS.content_preferences),
            response_length=DEFAULT_SETTINGS.response_length,
            visual_assist_enabled=DEFAULT_SETTINGS.visual_assist_enabled,
            companion_style=DEFAULT_SETTINGS.companion_style,
        )
        db.add(settings)
        db.flush()
        return settings

    def upsert(self, user_id: str, device_id: str, settings: UserSettings) -> None:
        with self.session_factory() as db:
            self.identity_service.require_user(db, user_id, device_id)
            row = self.get_or_create(db, user_id, device_id)
            row.device_id = device_id
            row.movement_mode = settings.movement_mode
            row.content_preferences = list(settings.content_preferences)
            row.response_length = settings.response_length
            row.visual_assist_enabled = settings.visual_assist_enabled
            row.companion_style = settings.companion_style
            db.commit()

    def get(self, user_id: str) -> SettingsGetResponse:
        with self.session_factory() as db:
            user = db.scalar(select(User).where(User.id == user_id))
            if user is None:
                raise ApiError(404, "IDENTITY_NOT_FOUND", "User was not found")
            row = self.get_or_create(db, user.id, user.device_id)
            db.commit()
            return SettingsGetResponse(
                userId=user.id,
                deviceId=user.device_id,
                settings=self.to_model(row),
            )

    @staticmethod
    def to_model(row: UserSetting) -> UserSettings:
        return UserSettings(
            movementMode=row.movement_mode,
            contentPreferences=row.content_preferences,
            responseLength=row.response_length,
            visualAssistEnabled=row.visual_assist_enabled,
            companionStyle=row.companion_style,
        )


class SessionService:
    def __init__(
        self,
        config: AppConfig,
        session_factory: sessionmaker,
        identity_service: IdentityService,
        settings_service: SettingsService,
        context_provider: ContextProvider,
        gemini_client: GeminiClient,
    ):
        self.config = config
        self.session_factory = session_factory
        self.identity_service = identity_service
        self.settings_service = settings_service
        self.context_provider = context_provider
        self.gemini_client = gemini_client
        self.latest_image_frames: dict[str, MediaFrame] = {}

    def create_session(self, request: SessionCreateRequest, websocket_url: str) -> SessionCreateResponse:
        with self.session_factory() as db:
            self.identity_service.require_user(db, request.user_id, request.device_id)
            settings = self.settings_service.get_or_create(db, request.user_id, request.device_id)
            settings.movement_mode = request.settings.movement_mode
            settings.content_preferences = list(request.settings.content_preferences)
            settings.response_length = request.settings.response_length
            settings.visual_assist_enabled = request.settings.visual_assist_enabled
            settings.companion_style = request.settings.companion_style

            session_id = new_id("sess")
            prompt = build_system_prompt(
                request.settings,
                self.context_provider.fallback_enrich(
                    request.destination.latitude if request.destination else None,
                    request.destination.longitude if request.destination else None,
                    request.destination.label if request.destination else None,
                    request.settings.movement_mode,
                ),
                request.destination.label if request.destination else None,
                False,
                [],
            )
            session = SessionRecord(
                id=session_id,
                user_id=request.user_id,
                device_id=request.device_id,
                status=SessionStatus.connecting,
                destination_label=request.destination.label if request.destination else None,
                destination_latitude=request.destination.latitude if request.destination else None,
                destination_longitude=request.destination.longitude if request.destination else None,
                destination_place_id=request.destination.place_id if request.destination else None,
                settings_snapshot=request.settings.model_dump(mode="json", by_alias=True),
                prompt_state=prompt,
                visual_assist_active=False,
            )
            db.add(session)
            db.commit()
            return SessionCreateResponse(
                sessionId=session_id,
                status=SessionStatus.connecting,
                websocketUrl=websocket_url,
                settingsEcho=request.settings,
            )

    def get_session(self, db: Session, session_id: str) -> SessionRecord:
        session = db.scalar(select(SessionRecord).where(SessionRecord.id == session_id))
        if session is None:
            raise ApiError(404, "SESSION_NOT_FOUND", "Session was not found")
        return session

    def start(self, session_id: str, user_id: str, device_id: str) -> None:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            if session.user_id != user_id or session.device_id != device_id:
                raise ApiError(403, "SESSION_FORBIDDEN", "Session does not belong to this identity")
            session.status = SessionStatus.active
            db.commit()

    def update_status(self, session_id: str, status: SessionStatus) -> None:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            session.status = status
            db.commit()

    def prompt_state(self, session_id: str) -> str:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            return session.prompt_state

    def get_settings(self, session: SessionRecord) -> UserSettings:
        return UserSettings.model_validate(session.settings_snapshot)

    async def update_settings(self, session_id: str, settings: UserSettings) -> ContextHintPayload:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            session.settings_snapshot = settings.model_dump(mode="json", by_alias=True)
            context = await self.context_provider.enrich(
                session.current_latitude,
                session.current_longitude,
                session.destination_label,
                settings.movement_mode,
                session.destination_place_id,
            )
            session.prompt_state = build_system_prompt(
                settings,
                context,
                session.destination_label,
                session.visual_assist_active,
                self.fetch_recent_transcript_lines(db, session_id),
            )
            db.commit()
            return ContextHintPayload(
                nearbyPlaceName=context.nearby_places[0],
                modeApplied=settings.movement_mode,
            )

    async def update_location(self, session_id: str, payload: LocationUpdatePayload) -> tuple[str | None, ContextHintPayload]:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            settings = self.get_settings(session)
            session.current_latitude = payload.latitude
            session.current_longitude = payload.longitude
            session.current_location_timestamp = payload.timestamp
            context = await self.context_provider.enrich(
                payload.latitude,
                payload.longitude,
                session.destination_label,
                settings.movement_mode,
                session.destination_place_id,
            )
            session.prompt_state = build_system_prompt(
                settings,
                context,
                session.destination_label,
                session.visual_assist_active,
                self.fetch_recent_transcript_lines(db, session_id),
            )
            should_speak = self.should_emit_location_response(session.last_context_hint_at)
            session.last_context_hint_at = utcnow()
            db.commit()

            response = None
            if should_speak:
                fallback = build_location_response(settings, context, session.destination_label)
                response = await self.generate_model_text(
                    session.prompt_state,
                    (
                        "The user's location changed. Give a short live companion update grounded in the "
                        f"nearby places and activities. Nearby places: {', '.join(context.nearby_places)}. "
                        f"Nearby activities: {', '.join(context.nearby_activities)}."
                    ),
                    self.latest_image_frames.get(session_id),
                    fallback,
                )
            return response, ContextHintPayload(
                nearbyPlaceName=context.nearby_places[0],
                modeApplied=settings.movement_mode,
            )

    async def update_destination(self, session_id: str, destination: Destination | None) -> str:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            settings = self.get_settings(session)
            if destination is None:
                session.destination_label = None
                session.destination_latitude = None
                session.destination_longitude = None
                session.destination_place_id = None
            else:
                session.destination_label = destination.label
                session.destination_latitude = destination.latitude
                session.destination_longitude = destination.longitude
                session.destination_place_id = destination.place_id
            context = await self.context_provider.enrich(
                session.current_latitude,
                session.current_longitude,
                session.destination_label,
                settings.movement_mode,
                session.destination_place_id,
            )
            session.prompt_state = build_system_prompt(
                settings,
                context,
                session.destination_label,
                session.visual_assist_active,
                self.fetch_recent_transcript_lines(db, session_id),
            )
            db.commit()
            if destination is None:
                fallback = build_destination_cleared_response(settings, context)
                user_prompt = (
                    "The destination was cleared. Respond as the live companion in one concise turn, "
                    "staying grounded in the user's current surroundings and nearby activity context."
                )
            else:
                fallback = build_destination_response(settings, destination.label, context)
                user_prompt = (
                    f"The destination was updated to {destination.label}. "
                    "Respond as the live companion in one concise turn, factoring in nearby places and route context."
                )
            return await self.generate_model_text(
                session.prompt_state,
                user_prompt,
                self.latest_image_frames.get(session_id),
                fallback,
            )

    def should_emit_location_response(self, last_context_hint_at: datetime | None) -> bool:
        if last_context_hint_at is None:
            return True
        return utcnow() - datetime_to_zulu(last_context_hint_at) >= timedelta(
            seconds=self.config.location_response_cooldown_seconds
        )

    def set_visual_assist(self, session_id: str, active: bool) -> None:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            session.visual_assist_active = active
            db.commit()

    def is_visual_assist_active(self, session_id: str) -> bool:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            return session.visual_assist_active

    def remember_image_frame(
        self,
        session_id: str,
        mime_type: str,
        data: str,
        timestamp: datetime,
    ) -> None:
        self.latest_image_frames[session_id] = MediaFrame(
            mime_type=mime_type,
            data=data,
            timestamp_iso=timestamp.isoformat().replace("+00:00", "Z"),
        )

    def clear_image_frame(self, session_id: str) -> None:
        self.latest_image_frames.pop(session_id, None)

    def latest_image_frame(self, session_id: str) -> MediaFrame | None:
        return self.latest_image_frames.get(session_id)

    def record_user_text(self, session_id: str, text: str, timestamp: datetime) -> None:
        self.record_transcript(session_id, "user", text, "text", timestamp)

    def record_assistant_text(self, session_id: str, text: str, timestamp: datetime) -> None:
        self.record_transcript(session_id, "assistant", text, "text", timestamp)

    def record_audio_event(self, session_id: str, timestamp: datetime) -> None:
        self.record_transcript(session_id, "user", "", "audio", timestamp)

    def record_image_event(self, session_id: str, timestamp: datetime) -> None:
        self.record_transcript(session_id, "assistant", "", "image", timestamp)

    def record_transcript(
        self,
        session_id: str,
        role: str,
        text: str,
        source_type: str,
        timestamp: datetime,
    ) -> None:
        with self.session_factory() as db:
            chunk = TranscriptChunk(
                session_id=session_id,
                role=role,
                text=text,
                source_type=source_type,
                is_final=True,
                event_timestamp=timestamp,
            )
            db.add(chunk)
            db.commit()

    def fetch_recent_transcript_lines(self, db: Session, session_id: str) -> list[str]:
        rows = db.scalars(
            select(TranscriptChunk)
            .where(TranscriptChunk.session_id == session_id, TranscriptChunk.text != "")
            .order_by(TranscriptChunk.created_at.desc())
            .limit(8)
        ).all()
        return [f"{row.role}:{row.text}" for row in reversed(rows)]

    async def prepare_text_turn(self, session_id: str, user_text: str) -> str:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            settings = self.get_settings(session)
            context = await self.context_provider.enrich(
                session.current_latitude,
                session.current_longitude,
                session.destination_label,
                settings.movement_mode,
                session.destination_place_id,
            )
            session.prompt_state = build_system_prompt(
                settings,
                context,
                session.destination_label,
                session.visual_assist_active,
                self.fetch_recent_transcript_lines(db, session_id),
            )
            db.commit()
            return build_text_response(
                user_text,
                settings,
                context,
                session.destination_label,
                session.visual_assist_active,
            )

    async def generate_text_reply_fallback(
        self,
        session_id: str,
        user_text: str,
        fallback_text: str,
    ) -> str:
        with self.session_factory() as db:
            session = self.get_session(db, session_id)
            return await self.generate_model_text(
                session.prompt_state,
                (
                    "Respond to the user's latest message as the live companion. "
                    f"Latest user text: {user_text}"
                ),
                self.latest_image_frames.get(session_id) if session.visual_assist_active else None,
                fallback_text,
            )

    async def generate_model_text(
        self,
        system_prompt: str,
        user_prompt: str,
        image_frame: MediaFrame | None,
        fallback_text: str,
    ) -> str:
        try:
            text = await self.gemini_client.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_frame=image_frame,
            )
        except Exception:
            text = None
        return text or fallback_text

    def transcript_window(self, session_id: str, center_time: datetime) -> list[TranscriptExcerptItem]:
        with self.session_factory() as db:
            rows = db.scalars(
                select(TranscriptChunk)
                .where(
                    TranscriptChunk.session_id == session_id,
                    TranscriptChunk.text != "",
                )
                .order_by(TranscriptChunk.created_at.desc())
                .limit(12)
            ).all()
            rows = list(reversed(rows))
            if not rows:
                return []
            eligible: list[TranscriptChunk] = []
            for row in rows:
                row_time = row.event_timestamp or row.created_at
                if abs((datetime_to_zulu(row_time) - center_time).total_seconds()) <= 300:
                    eligible.append(row)
            excerpt_rows = eligible or rows[-6:]
            return [TranscriptExcerptItem(role=row.role, text=row.text) for row in excerpt_rows]


class GeminiLiveBridge:
    def __init__(
        self,
        config: AppConfig,
        live_manager: LiveSessionManager,
        session_service: SessionService,
    ):
        self.config = config
        self.live_manager = live_manager
        self.session_service = session_service
        self.connections: dict[str, GeminiLiveConnection] = {}
        self.http_session: ClientSession | None = None
        self.timeout = ClientTimeout(total=None, sock_connect=20)

    @property
    def enabled(self) -> bool:
        return bool(self.config.gemini_api_key and self.config.gemini_live_ws_url)

    async def startup(self) -> None:
        if self.enabled and self.http_session is None:
            self.http_session = ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        for session_id in list(self.connections):
            await self.close_session(session_id)
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None

    async def refresh_if_active(self, session_id: str) -> None:
        try:
            await self.ensure_session(session_id)
        except Exception as exc:
            logger.warning("Gemini Live refresh degraded for session %s: %s", session_id, type(exc).__name__)
            return

    async def restart_session(self, session_id: str) -> None:
        if not self.enabled:
            return
        await self.close_session(session_id)
        try:
            await self.ensure_session(session_id)
        except Exception as exc:
            logger.warning("Gemini Live restart degraded for session %s: %s", session_id, type(exc).__name__)
            return

    async def ensure_session(self, session_id: str) -> GeminiLiveConnection | None:
        if not self.enabled:
            return None
        await self.startup()
        prompt_state = self.session_service.prompt_state(session_id)
        existing = self.connections.get(session_id)
        if existing is not None and not existing.ws.closed and existing.prompt_state == prompt_state:
            return existing
        if existing is not None:
            await self.close_session(session_id)
        return await self._connect(session_id, prompt_state)

    async def _connect(self, session_id: str, prompt_state: str) -> GeminiLiveConnection:
        if self.http_session is None:
            raise RuntimeError("Gemini live client session is not initialized")
        ws = await self.http_session.ws_connect(
            self.config.gemini_live_ws_url,
            headers={"x-goog-api-key": self.config.gemini_api_key or ""},
            heartbeat=30,
            receive_timeout=None,
        )
        setup_complete = asyncio.Event()
        connection = GeminiLiveConnection(
            ws=ws,
            prompt_state=prompt_state,
            setup_complete=setup_complete,
            reader_task=asyncio.create_task(self._reader(session_id)),
        )
        self.connections[session_id] = connection
        await ws.send_json(
            {
                "setup": {
                    "model": f"models/{self.config.gemini_live_model}",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                    },
                    "systemInstruction": {
                        "parts": [{"text": prompt_state}],
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                }
            }
        )
        await asyncio.wait_for(setup_complete.wait(), timeout=10)
        latest_frame = self.session_service.latest_image_frame(session_id)
        if latest_frame is not None and self.session_service.is_visual_assist_active(session_id):
            await ws.send_json(
                {
                    "realtimeInput": {
                        "video": {
                            "mimeType": latest_frame.mime_type,
                            "data": latest_frame.data,
                        }
                    }
                }
            )
        return connection

    async def _reader(self, session_id: str) -> None:
        connection = self.connections[session_id]
        ws = connection.ws
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                data = message.json()
                if data.get("setupComplete") == {} or "setupComplete" in data:
                    connection.setup_complete.set()
                    continue
                server_content = data.get("serverContent")
                if server_content:
                    await self._handle_server_content(session_id, connection, server_content)
        except Exception as exc:
            logger.warning("Gemini Live reader closed for session %s: %s", session_id, type(exc).__name__)
            pass
        finally:
            if connection.audio_end_task is not None:
                connection.audio_end_task.cancel()
            if self.connections.get(session_id) is connection:
                self.connections.pop(session_id, None)

    async def _handle_server_content(
        self,
        session_id: str,
        connection: GeminiLiveConnection,
        server_content: dict[str, Any],
    ) -> None:
        is_final = bool(server_content.get("turnComplete") or server_content.get("generationComplete"))
        timestamp = utcnow().isoformat().replace("+00:00", "Z")

        input_transcription = server_content.get("inputTranscription") or {}
        input_text = input_transcription.get("text", "").strip()
        if input_text and input_text != connection.last_user_transcript:
            connection.last_user_transcript = input_text
            await self.live_manager.emit(
                session_id,
                "user.transcript",
                {
                    "text": input_text,
                    "isFinal": is_final,
                    "timestamp": timestamp,
                },
            )
            if is_final and input_text != connection.last_user_transcript_recorded:
                self.session_service.record_transcript(
                    session_id,
                    "user",
                    input_text,
                    "audio",
                    utcnow(),
                )
                connection.last_user_transcript_recorded = input_text

        output_transcription = server_content.get("outputTranscription") or {}
        output_text = output_transcription.get("text", "").strip()
        if output_text and output_text != connection.last_assistant_transcript:
            connection.last_assistant_transcript = output_text
            await self.live_manager.emit(
                session_id,
                "assistant.transcript",
                {
                    "text": output_text,
                    "isFinal": is_final,
                    "timestamp": timestamp,
                },
            )
            if is_final and output_text != connection.last_assistant_transcript_recorded:
                self.session_service.record_assistant_text(session_id, output_text, utcnow())
                connection.last_assistant_transcript_recorded = output_text

        model_turn = server_content.get("modelTurn") or {}
        for part in model_turn.get("parts", []):
            inline_data = part.get("inlineData") or {}
            if inline_data.get("data"):
                await self.live_manager.emit(
                    session_id,
                    "assistant.audio.chunk",
                    {
                        "mimeType": "audio/pcm",
                        "data": inline_data["data"],
                        "timestamp": timestamp,
                    },
                )
            text = part.get("text", "").strip()
            if text and text != connection.last_assistant_transcript:
                connection.last_assistant_transcript = text
                await self.live_manager.emit(
                    session_id,
                    "assistant.transcript",
                    {
                        "text": text,
                        "isFinal": is_final,
                        "timestamp": timestamp,
                    },
                )
                if is_final and text != connection.last_assistant_transcript_recorded:
                    self.session_service.record_assistant_text(session_id, text, utcnow())
                    connection.last_assistant_transcript_recorded = text

    async def send_audio_chunk(
        self,
        session_id: str,
        mime_type: str,
        data: str,
    ) -> bool:
        try:
            connection = await self.ensure_session(session_id)
        except Exception as exc:
            logger.warning(
                "Gemini Live audio input unavailable for session %s: %s",
                session_id,
                type(exc).__name__,
            )
            return False
        if connection is None:
            return False
        await connection.ws.send_json(
            {
                "realtimeInput": {
                    "audio": {
                        "mimeType": mime_type,
                        "data": data,
                    }
                }
            }
        )
        connection.audio_chunk_count += 1
        if connection.audio_end_task is not None:
            connection.audio_end_task.cancel()
        connection.audio_end_task = asyncio.create_task(self._delayed_audio_end(session_id))
        return True

    async def send_text_turn(
        self,
        session_id: str,
        text: str,
        timestamp: datetime,
    ) -> bool:
        try:
            connection = await self.ensure_session(session_id)
        except Exception as exc:
            logger.warning(
                "Gemini Live text turn fell back for session %s: %s",
                session_id,
                type(exc).__name__,
            )
            return False
        if connection is None:
            return False
        try:
            await connection.ws.send_json(
                {
                    "clientContent": {
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": text}],
                            }
                        ],
                        "turnComplete": True,
                    }
                }
            )
        except Exception as exc:
            logger.warning(
                "Gemini Live text turn write failed for session %s: %s",
                session_id,
                type(exc).__name__,
            )
            return False
        iso_timestamp = timestamp.isoformat().replace("+00:00", "Z")
        await self.live_manager.emit(
            session_id,
            "user.transcript",
            {
                "text": text,
                "isFinal": True,
                "timestamp": iso_timestamp,
            },
        )
        self.session_service.record_transcript(
            session_id,
            "user",
            text,
            "text",
            timestamp,
        )
        return True

    async def send_image_frame(
        self,
        session_id: str,
        mime_type: str,
        data: str,
    ) -> bool:
        try:
            connection = await self.ensure_session(session_id)
        except Exception as exc:
            logger.warning(
                "Gemini Live image input unavailable for session %s: %s",
                session_id,
                type(exc).__name__,
            )
            return False
        if connection is None:
            return False
        await connection.ws.send_json(
            {
                "realtimeInput": {
                    "video": {
                        "mimeType": mime_type,
                        "data": data,
                    }
                }
            }
        )
        return True

    async def _delayed_audio_end(self, session_id: str) -> None:
        try:
            connection = self.connections.get(session_id)
            if connection is None:
                return
            await asyncio.sleep(self._audio_end_delay_seconds(connection))
            connection = self.connections.get(session_id)
            if connection is None or connection.ws.closed:
                return
            if connection.audio_chunk_count <= 0:
                return
            await connection.ws.send_json({"realtimeInput": {"audioStreamEnd": True}})
            connection.audio_chunk_count = 0
        except asyncio.CancelledError:
            return

    def _audio_end_delay_seconds(self, connection: GeminiLiveConnection) -> float:
        # The client contract has no explicit "mic off" event, so the backend must infer
        # end-of-utterance. In non-test environments we keep a safer minimum idle window
        # and add a small grace period for very short bursts to avoid closing on brief pauses.
        base = self.config.gemini_live_audio_idle_seconds
        if self.config.app_env != "test":
            base = max(base, 2.0)
            if connection.audio_chunk_count < 6:
                base += 0.35
        return base

    async def close_session(self, session_id: str) -> None:
        connection = self.connections.pop(session_id, None)
        if connection is None:
            return
        if connection.audio_end_task is not None:
            connection.audio_end_task.cancel()
        connection.reader_task.cancel()
        try:
            await connection.ws.close()
        except Exception:
            pass
        try:
            await connection.reader_task
        except (asyncio.CancelledError, Exception):
            pass


class SummaryService:
    def __init__(
        self,
        config: AppConfig,
        session_factory: sessionmaker,
        live_manager: LiveSessionManager,
        context_provider: ContextProvider,
        gemini_client: GeminiClient,
    ):
        self.config = config
        self.session_factory = session_factory
        self.live_manager = live_manager
        self.context_provider = context_provider
        self.gemini_client = gemini_client

    async def generate(self, bookmark_id: str) -> None:
        await asyncio.sleep(self.config.summary_delay_seconds)
        session_id: str | None = None
        try:
            with self.session_factory() as db:
                bookmark = db.scalar(select(Bookmark).where(Bookmark.id == bookmark_id))
                summary = db.scalar(
                    select(BookmarkSummary).where(BookmarkSummary.bookmark_id == bookmark_id)
                )
                if bookmark is None or summary is None:
                    return
                session_id = bookmark.session_id
                session = db.scalar(select(SessionRecord).where(SessionRecord.id == bookmark.session_id))
                context = await self.context_provider.enrich(
                    bookmark.latitude,
                    bookmark.longitude,
                    session.destination_label if session is not None else None,
                    UserSettings.model_validate(session.settings_snapshot).movement_mode if session is not None else DEFAULT_SETTINGS.movement_mode,
                    bookmark.place_id,
                )
                summary_json = await self.build_summary(
                    bookmark.title,
                    bookmark.transcript_excerpt,
                    bookmark.note_text,
                    context.nearby_places,
                    context.nearby_activities,
                )
                summary.summary_status = SummaryStatus.ready
                summary.summary_json = summary_json.model_dump(mode="json", by_alias=True)
                summary.transcript_excerpt_json = bookmark.transcript_excerpt
                bookmark.summary_status = SummaryStatus.ready
                db.commit()
        except Exception as exc:
            with self.session_factory() as db:
                summary = db.scalar(
                    select(BookmarkSummary).where(BookmarkSummary.bookmark_id == bookmark_id)
                )
                bookmark = db.scalar(select(Bookmark).where(Bookmark.id == bookmark_id))
                if summary is not None:
                    summary.summary_status = SummaryStatus.failed
                    summary.error_message = str(exc)
                if bookmark is not None:
                    bookmark.summary_status = SummaryStatus.failed
                    session_id = bookmark.session_id
                db.commit()
            return

        if session_id:
            await self.live_manager.emit(
                session_id,
                "bookmark.summary_ready",
                {"bookmarkId": bookmark_id},
            )

    async def build_summary(
        self,
        title: str,
        transcript_excerpt: list[dict[str, Any]],
        note_text: str,
        nearby_places: list[str],
        nearby_activities: list[str],
    ) -> BookmarkSummaryPayload:
        try:
            summary = await self.gemini_client.generate_summary(
                title=title,
                transcript_excerpt=transcript_excerpt,
                note_text=note_text,
                nearby_places=nearby_places,
                nearby_activities=nearby_activities,
            )
        except Exception:
            summary = None
        if summary is not None:
            return summary

        text_lines = [item["text"] for item in transcript_excerpt if item.get("text")]
        short = text_lines[-1] if text_lines else "A saved companion moment with lightweight route context."
        nearby = [title.replace("Near ", "")] if title.startswith("Near ") else [title]
        tags = [token.lower() for token in title.replace(",", "").split()[:3]]
        if note_text:
            tags.append("note")
        return BookmarkSummaryPayload(
            headline=title,
            shortSummary=short[:180],
            whatWasNearby=nearby,
            activitiesMentioned=["live companion context"],
            tags=list(dict.fromkeys(tags)),
        )


class BookmarkService:
    def __init__(
        self,
        session_factory: sessionmaker,
        identity_service: IdentityService,
        session_service: SessionService,
        summary_service: SummaryService,
        context_provider: ContextProvider,
    ):
        self.session_factory = session_factory
        self.identity_service = identity_service
        self.session_service = session_service
        self.summary_service = summary_service
        self.context_provider = context_provider

    async def create(self, request: BookmarkCreateRequest) -> BookmarkCreateResponse:
        with self.session_factory() as db:
            self.identity_service.require_user(db, request.user_id, request.device_id)
            session = self.session_service.get_session(db, request.session_id)
            if session.user_id != request.user_id or session.device_id != request.device_id:
                raise ApiError(403, "BOOKMARK_FORBIDDEN", "Bookmark does not belong to this identity")
            excerpt_items = self.session_service.transcript_window(request.session_id, request.timestamp)
            excerpt = [item.model_dump(mode="json", by_alias=True) for item in excerpt_items]
            settings = self.session_service.get_settings(session)
            context = await self.context_provider.enrich(
                request.coordinate.latitude,
                request.coordinate.longitude,
                session.destination_label,
                settings.movement_mode,
                session.destination_place_id,
            )
            title = context.title_hint
            bookmark = Bookmark(
                id=new_id("bm"),
                user_id=request.user_id,
                device_id=request.device_id,
                session_id=request.session_id,
                title=title if title.startswith("Near ") else f"Near {title}",
                latitude=request.coordinate.latitude,
                longitude=request.coordinate.longitude,
                place_id=context.place_id or session.destination_place_id,
                note_text="",
                summary_status=SummaryStatus.pending,
                transcript_excerpt=excerpt,
            )
            db.add(bookmark)
            summary = BookmarkSummary(
                bookmark_id=bookmark.id,
                summary_status=SummaryStatus.pending,
                transcript_excerpt_json=excerpt,
            )
            db.add(summary)
            db.commit()
            asyncio.create_task(self.summary_service.generate(bookmark.id))
            return BookmarkCreateResponse(
                bookmarkId=bookmark.id,
                title=bookmark.title,
                latitude=bookmark.latitude,
                longitude=bookmark.longitude,
                placeId=bookmark.place_id,
                noteText=bookmark.note_text,
                summaryStatus=bookmark.summary_status,
                createdAt=bookmark.created_at,
            )

    def list(self, user_id: str) -> BookmarkListResponse:
        with self.session_factory() as db:
            rows = db.scalars(
                select(Bookmark).where(Bookmark.user_id == user_id).order_by(Bookmark.created_at.desc())
            ).all()
            items = [
                BookmarkListItem(
                    bookmarkId=row.id,
                    title=row.title,
                    latitude=row.latitude,
                    longitude=row.longitude,
                    createdAt=row.created_at,
                    summaryStatus=row.summary_status,
                )
                for row in rows
            ]
            return BookmarkListResponse(items=items)

    def get(self, bookmark_id: str) -> BookmarkDetailResponse:
        with self.session_factory() as db:
            bookmark = db.scalar(select(Bookmark).where(Bookmark.id == bookmark_id))
            summary = db.scalar(select(BookmarkSummary).where(BookmarkSummary.bookmark_id == bookmark_id))
            if bookmark is None or summary is None:
                raise ApiError(404, "BOOKMARK_NOT_FOUND", "Bookmark was not found")
            transcript = [
                TranscriptExcerptItem.model_validate(item) for item in summary.transcript_excerpt_json
            ]
            summary_payload = (
                BookmarkSummaryPayload.model_validate(summary.summary_json)
                if summary.summary_json
                else None
            )
            return BookmarkDetailResponse(
                bookmarkId=bookmark.id,
                title=bookmark.title,
                latitude=bookmark.latitude,
                longitude=bookmark.longitude,
                createdAt=bookmark.created_at,
                noteText=bookmark.note_text,
                summaryStatus=bookmark.summary_status,
                transcriptExcerpt=transcript,
                summary=summary_payload,
            )

    def update_note(self, bookmark_id: str, user_id: str, device_id: str, note_text: str) -> BookmarkNotePatchResponse:
        with self.session_factory() as db:
            self.identity_service.require_user(db, user_id, device_id)
            bookmark = db.scalar(select(Bookmark).where(Bookmark.id == bookmark_id))
            if bookmark is None:
                raise ApiError(404, "BOOKMARK_NOT_FOUND", "Bookmark was not found")
            if bookmark.user_id != user_id or bookmark.device_id != device_id:
                raise ApiError(403, "BOOKMARK_FORBIDDEN", "Bookmark does not belong to this identity")
            bookmark.note_text = note_text
            db.commit()
            return BookmarkNotePatchResponse(
                ok=True,
                bookmarkId=bookmark_id,
                noteText=note_text,
            )


def parse_payload(model_class: type, payload: dict[str, Any]) -> Any:
    try:
        return model_class.model_validate(payload)
    except ValidationError as exc:
        raise ApiError(400, "BAD_REQUEST", exc.errors()[0]["msg"]) from exc
