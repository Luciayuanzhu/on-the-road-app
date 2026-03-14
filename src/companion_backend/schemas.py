from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)


class MovementMode(str, Enum):
    walking = "walking"
    ride = "ride"


class ResponseLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class CompanionStyle(str, Enum):
    guide_like = "guide_like"
    casual = "casual"


class ContentPreference(str, Enum):
    fun_facts = "fun_facts"
    history_culture = "history_culture"
    nature_scenery = "nature_scenery"
    food_cafes = "food_cafes"
    events_activities = "events_activities"
    hidden_gems = "hidden_gems"


class SessionStatus(str, Enum):
    idle = "idle"
    connecting = "connecting"
    active = "active"
    paused = "paused"
    ended = "ended"
    error = "error"


class SummaryStatus(str, Enum):
    pending = "pending"
    ready = "ready"
    failed = "failed"


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class UserSettings(ApiModel):
    movement_mode: MovementMode = Field(alias="movementMode")
    content_preferences: list[ContentPreference] = Field(alias="contentPreferences")
    response_length: ResponseLength = Field(alias="responseLength")
    visual_assist_enabled: bool = Field(alias="visualAssistEnabled")
    companion_style: CompanionStyle = Field(alias="companionStyle")


DEFAULT_SETTINGS = UserSettings(
    movementMode=MovementMode.walking,
    contentPreferences=[ContentPreference.fun_facts, ContentPreference.nature_scenery],
    responseLength=ResponseLength.medium,
    visualAssistEnabled=False,
    companionStyle=CompanionStyle.casual,
)


class Destination(ApiModel):
    label: str
    latitude: float
    longitude: float
    place_id: str | None = Field(default=None, alias="placeId")


class Coordinate(ApiModel):
    latitude: float
    longitude: float
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class IdentityResolutionRequest(ApiModel):
    device_id: str = Field(alias="deviceId", min_length=1)


class IdentityResolutionResponse(ApiModel):
    device_id: str = Field(alias="deviceId")
    user_id: str = Field(alias="userId")
    is_new_user: bool = Field(alias="isNewUser")


class SessionCreateRequest(ApiModel):
    user_id: str = Field(alias="userId")
    device_id: str = Field(alias="deviceId")
    settings: UserSettings
    destination: Destination | None = None


class SessionCreateResponse(ApiModel):
    session_id: str = Field(alias="sessionId")
    status: SessionStatus
    websocket_url: str = Field(alias="websocketUrl")
    settings_echo: UserSettings = Field(alias="settingsEcho")


class SettingsUpsertRequest(ApiModel):
    user_id: str = Field(alias="userId")
    device_id: str = Field(alias="deviceId")
    settings: UserSettings


class OkResponse(ApiModel):
    ok: bool


class SettingsGetResponse(ApiModel):
    user_id: str = Field(alias="userId")
    device_id: str = Field(alias="deviceId")
    settings: UserSettings


class BookmarkCreateRequest(ApiModel):
    user_id: str = Field(alias="userId")
    device_id: str = Field(alias="deviceId")
    session_id: str = Field(alias="sessionId")
    timestamp: datetime
    coordinate: Coordinate

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class BookmarkCreateResponse(ApiModel):
    bookmark_id: str = Field(alias="bookmarkId")
    title: str
    latitude: float
    longitude: float
    place_id: str | None = Field(default=None, alias="placeId")
    note_text: str = Field(alias="noteText")
    summary_status: SummaryStatus = Field(alias="summaryStatus")
    created_at: datetime = Field(alias="createdAt")


class BookmarkListItem(ApiModel):
    bookmark_id: str = Field(alias="bookmarkId")
    title: str
    latitude: float
    longitude: float
    created_at: datetime = Field(alias="createdAt")
    summary_status: SummaryStatus = Field(alias="summaryStatus")


class BookmarkListResponse(ApiModel):
    items: list[BookmarkListItem]


class TranscriptExcerptItem(ApiModel):
    role: str
    text: str


class BookmarkSummaryPayload(ApiModel):
    headline: str
    short_summary: str = Field(alias="shortSummary")
    what_was_nearby: list[str] = Field(alias="whatWasNearby")
    activities_mentioned: list[str] = Field(alias="activitiesMentioned")
    tags: list[str]


class BookmarkDetailResponse(ApiModel):
    bookmark_id: str = Field(alias="bookmarkId")
    title: str
    latitude: float
    longitude: float
    created_at: datetime = Field(alias="createdAt")
    note_text: str = Field(alias="noteText")
    summary_status: SummaryStatus = Field(alias="summaryStatus")
    transcript_excerpt: list[TranscriptExcerptItem] = Field(alias="transcriptExcerpt")
    summary: BookmarkSummaryPayload | None


class BookmarkNotePatchRequest(ApiModel):
    user_id: str = Field(alias="userId")
    device_id: str = Field(alias="deviceId")
    note_text: str = Field(alias="noteText")


class BookmarkNotePatchResponse(ApiModel):
    ok: bool
    bookmark_id: str = Field(alias="bookmarkId")
    note_text: str = Field(alias="noteText")


class SessionStartPayload(ApiModel):
    user_id: str = Field(alias="userId")
    device_id: str = Field(alias="deviceId")
    session_id: str = Field(alias="sessionId")


class SessionUpdateSettingsPayload(ApiModel):
    settings: UserSettings


class LocationUpdatePayload(ApiModel):
    latitude: float
    longitude: float
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class TextInputPayload(ApiModel):
    text: str
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class AudioInputChunkPayload(ApiModel):
    mime_type: str = Field(alias="mimeType")
    data: str
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class VisualAssistPayload(ApiModel):
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class ImageInputFramePayload(ApiModel):
    mime_type: str = Field(alias="mimeType")
    data: str
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class SessionStatusPayload(ApiModel):
    status: SessionStatus


class AssistantTranscriptPayload(ApiModel):
    text: str
    is_final: bool = Field(alias="isFinal")
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class AssistantAudioChunkPayload(ApiModel):
    mime_type: str = Field(alias="mimeType")
    data: str
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class UserTranscriptPayload(ApiModel):
    text: str
    is_final: bool = Field(alias="isFinal")
    timestamp: datetime

    _normalize_timestamp = field_validator("timestamp", mode="after")(ensure_utc)


class ContextHintPayload(ApiModel):
    nearby_place_name: str = Field(alias="nearbyPlaceName")
    mode_applied: MovementMode = Field(alias="modeApplied")


class BookmarkSummaryReadyPayload(ApiModel):
    bookmark_id: str = Field(alias="bookmarkId")


class ErrorPayload(ApiModel):
    code: str
    message: str


class WebsocketEnvelope(ApiModel):
    type: str
    payload: dict[str, Any] | None
