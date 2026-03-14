from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from .config import AppConfig
from .schemas import BookmarkSummaryPayload

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextSnapshot:
    nearby_places: list[str]
    nearby_activities: list[str]
    title_hint: str
    place_id: str | None = None
    nearby_events: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlaceCandidate:
    place_id: str | None
    display_name: str
    formatted_address: str | None
    primary_type: str | None
    types: list[str]


@dataclass(slots=True)
class MediaFrame:
    mime_type: str
    data: str
    timestamp_iso: str


@dataclass(slots=True)
class EventCandidate:
    name: str
    venue_name: str | None
    start_local_date: str | None
    start_local_time: str | None
    segment_name: str | None
    genre_name: str | None


@dataclass(slots=True)
class CachedEventSearch:
    cached_at_monotonic: float
    events: list[EventCandidate]


TYPE_ACTIVITY_MAP = {
    "tourist_attraction": "landmarks and highlights",
    "museum": "history and culture",
    "art_gallery": "art stops",
    "park": "nature and scenery",
    "cafe": "cafes and quick breaks",
    "restaurant": "food stops",
    "point_of_interest": "local points of interest",
    "establishment": "local spots",
}


class GeminiClient:
    def __init__(self, config: AppConfig):
        self.api_key = config.gemini_api_key
        self.model = config.gemini_model
        self.base_url = config.gemini_base_url.rstrip("/")
        self.timeout = ClientTimeout(total=30)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_frame: MediaFrame | None = None,
    ) -> str | None:
        if not self.enabled:
            return None

        parts: list[dict[str, Any]] = []
        if image_frame is not None:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": image_frame.mime_type,
                        "data": image_frame.data,
                    }
                }
            )
        parts.append({"text": user_prompt})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.6,
                "maxOutputTokens": 320,
            },
        }
        data = await self._post_generate_content(payload)
        return self._extract_text(data)

    async def generate_summary(
        self,
        *,
        title: str,
        transcript_excerpt: list[dict[str, Any]],
        note_text: str,
        nearby_places: list[str],
        nearby_activities: list[str],
    ) -> BookmarkSummaryPayload | None:
        if not self.enabled:
            return None

        prompt = (
            "Summarize this saved live companion bookmark. "
            "Return compact factual output only.\n"
            f"title: {title}\n"
            f"nearby_places: {', '.join(nearby_places) or 'unknown'}\n"
            f"nearby_activities: {', '.join(nearby_activities) or 'unknown'}\n"
            f"user_note: {note_text or 'none'}\n"
            f"transcript_excerpt: {json.dumps(transcript_excerpt, ensure_ascii=True)}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string"},
                        "shortSummary": {"type": "string"},
                        "whatWasNearby": {"type": "array", "items": {"type": "string"}},
                        "activitiesMentioned": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "headline",
                        "shortSummary",
                        "whatWasNearby",
                        "activitiesMentioned",
                        "tags",
                    ],
                },
            },
        }
        data = await self._post_generate_content(payload)
        text = self._extract_text(data)
        if not text:
            return None
        return BookmarkSummaryPayload.model_validate_json(text)

    async def _post_generate_content(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Gemini API key is not configured")
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        async with ClientSession(timeout=self.timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str | None:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        if not text_parts:
            return None
        return "".join(text_parts).strip()


class ContextProvider:
    def __init__(self, config: AppConfig):
        self.api_key = config.google_places_api_key
        self.base_url = config.google_places_base_url.rstrip("/")
        self.language_code = config.google_places_language_code
        self.ticketmaster_api_key = config.ticketmaster_api_key
        self.ticketmaster_base_url = config.ticketmaster_base_url.rstrip("/")
        self.ticketmaster_event_cache_seconds = config.ticketmaster_event_cache_seconds
        self.ticketmaster_event_cache: dict[str, CachedEventSearch] = {}
        self.timeout = ClientTimeout(total=20)

    def fallback_enrich(
        self,
        latitude: float | None,
        longitude: float | None,
        destination_label: str | None,
        movement_mode: str,
    ) -> ContextSnapshot:
        if destination_label:
            places = [destination_label, f"Approach around {destination_label}"]
            title_hint = f"Near {destination_label}"
        elif latitude is not None and longitude is not None:
            coordinate_label = f"{latitude:.4f}, {longitude:.4f}"
            places = [f"Area near {coordinate_label}", "Local streetscape"]
            title_hint = f"Near {coordinate_label}"
        else:
            places = ["Current area", "Nearby route"]
            title_hint = "Saved companion moment"

        activities = (
            ["good views along the route", "light stops worth checking"]
            if movement_mode == "walking"
            else ["route-side highlights", "quick stop opportunities"]
        )
        return ContextSnapshot(
            nearby_places=places,
            nearby_activities=activities,
            title_hint=title_hint,
            nearby_events=[],
        )

    async def enrich(
        self,
        latitude: float | None,
        longitude: float | None,
        destination_label: str | None,
        movement_mode: str,
        destination_place_id: str | None = None,
    ) -> ContextSnapshot:
        fallback = self.fallback_enrich(latitude, longitude, destination_label, movement_mode)
        try:
            nearby: list[PlaceCandidate] = []
            if self.api_key and latitude is not None and longitude is not None:
                try:
                    nearby = await self.search_nearby(latitude, longitude, movement_mode)
                except Exception as exc:
                    logger.warning("Places nearby search degraded: %s", type(exc).__name__)

            destination_place: PlaceCandidate | None = None
            if self.api_key and destination_place_id:
                try:
                    destination_place = await self.get_place_details(destination_place_id)
                except Exception as exc:
                    logger.warning("Places details degraded: %s", type(exc).__name__)
            elif self.api_key and destination_label:
                try:
                    destination_place = await self.search_text(
                        destination_label,
                        latitude,
                        longitude,
                    )
                except Exception as exc:
                    logger.warning("Places text search degraded: %s", type(exc).__name__)

            nearby_events: list[EventCandidate] = []
            if latitude is not None and longitude is not None:
                try:
                    nearby_events = await self.search_events(latitude, longitude, movement_mode)
                except Exception as exc:
                    logger.warning("Ticketmaster events degraded: %s", type(exc).__name__)

            place_names: list[str] = []
            place_id = destination_place.place_id if destination_place else None
            title_hint = fallback.title_hint
            if destination_place is not None:
                title_hint = f"Near {destination_place.display_name}"
                place_names.append(destination_place.display_name)
            for item in nearby:
                if item.display_name not in place_names:
                    place_names.append(item.display_name)
                if place_id is None and item.place_id:
                    place_id = item.place_id
                if title_hint == fallback.title_hint:
                    title_hint = f"Near {item.display_name}"

            nearby_places = place_names[:4] or fallback.nearby_places
            nearby_activity_labels = self._merge_unique(
                self._derive_activities(nearby, movement_mode) + self._derive_event_activities(nearby_events)
            )
            return ContextSnapshot(
                nearby_places=nearby_places,
                nearby_activities=nearby_activity_labels[:4] or fallback.nearby_activities,
                title_hint=title_hint,
                place_id=place_id,
                nearby_events=self._summarize_events(nearby_events),
            )
        except Exception as exc:
            logger.warning("Context enrichment degraded to fallback: %s", type(exc).__name__)
            return fallback

    async def search_nearby(
        self,
        latitude: float,
        longitude: float,
        movement_mode: str,
    ) -> list[PlaceCandidate]:
        radius = 900.0 if movement_mode == "walking" else 2200.0
        body = {
            "maxResultCount": 5,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius": radius,
                }
            },
            "rankPreference": "POPULARITY",
            "languageCode": self.language_code,
        }
        headers = {
            "X-Goog-Api-Key": self.api_key or "",
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.primaryType,places.types",
            "Content-Type": "application/json",
        }
        data = await self._post_json("/v1/places:searchNearby", headers=headers, body=body)
        return [self._parse_place(item) for item in data.get("places", [])]

    async def search_text(
        self,
        query: str,
        latitude: float | None,
        longitude: float | None,
    ) -> PlaceCandidate | None:
        body: dict[str, Any] = {
            "textQuery": query,
            "maxResultCount": 1,
            "languageCode": self.language_code,
        }
        if latitude is not None and longitude is not None:
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius": 5000.0,
                }
            }
        headers = {
            "X-Goog-Api-Key": self.api_key or "",
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.primaryType,places.types",
            "Content-Type": "application/json",
        }
        data = await self._post_json("/v1/places:searchText", headers=headers, body=body)
        places = data.get("places", [])
        return self._parse_place(places[0]) if places else None

    async def get_place_details(self, place_id: str) -> PlaceCandidate:
        headers = {
            "X-Goog-Api-Key": self.api_key or "",
            "X-Goog-FieldMask": "id,displayName,formattedAddress,primaryType,types",
            "Content-Type": "application/json",
        }
        data = await self._get_json(f"/v1/places/{place_id}", headers=headers)
        return self._parse_place(data)

    async def search_events(
        self,
        latitude: float,
        longitude: float,
        movement_mode: str,
    ) -> list[EventCandidate]:
        if not self.ticketmaster_api_key:
            return []

        cache_key = self._ticketmaster_cache_key(latitude, longitude, movement_mode)
        cached = self.ticketmaster_event_cache.get(cache_key)
        now = monotonic()
        if cached is not None and now - cached.cached_at_monotonic < self.ticketmaster_event_cache_seconds:
            return list(cached.events)

        params = {
            "apikey": self.ticketmaster_api_key,
            "latlong": f"{latitude:.6f},{longitude:.6f}",
            "radius": f"{self._ticketmaster_radius_km(movement_mode):.1f}",
            "unit": "km",
            "size": "4",
            "sort": "date,asc",
            "locale": "*",
            "startDateTime": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        data = await self._get_json(
            "/events.json",
            headers={},
            params=params,
            base_url=self.ticketmaster_base_url,
        )
        events = [self._parse_event(item) for item in data.get("_embedded", {}).get("events", [])]
        self.ticketmaster_event_cache[cache_key] = CachedEventSearch(
            cached_at_monotonic=now,
            events=events,
        )
        return list(events)

    async def _post_json(
        self,
        path: str,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        async with ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}{path}", headers=headers, json=body) as response:
                response.raise_for_status()
                return await response.json()

    async def _get_json(
        self,
        path: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        async with ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{base_url or self.base_url}{path}", headers=headers, params=params) as response:
                response.raise_for_status()
                return await response.json()

    def _derive_activities(
        self,
        places: list[PlaceCandidate],
        movement_mode: str,
    ) -> list[str]:
        labels: list[str] = []
        for place in places:
            for place_type in [place.primary_type, *place.types]:
                if place_type and place_type in TYPE_ACTIVITY_MAP:
                    label = TYPE_ACTIVITY_MAP[place_type]
                    if label not in labels:
                        labels.append(label)
        if movement_mode == "ride":
            labels.append("route-side stops")
        else:
            labels.append("walkable nearby stops")
        return labels[:4]

    @staticmethod
    def _derive_event_activities(events: list[EventCandidate]) -> list[str]:
        if not events:
            return []
        labels = ["upcoming live events"]
        for event in events:
            for raw_label in [event.genre_name, event.segment_name]:
                if not raw_label:
                    continue
                label = f"{raw_label.lower()} events"
                if label not in labels:
                    labels.append(label)
        return labels

    @staticmethod
    def _summarize_events(events: list[EventCandidate]) -> list[str]:
        summaries: list[str] = []
        for event in events[:3]:
            time_bits = [bit for bit in [event.start_local_date, event.start_local_time] if bit]
            prefix = f"{' '.join(time_bits)}: " if time_bits else ""
            venue = f" at {event.venue_name}" if event.venue_name else ""
            summaries.append(f"{prefix}{event.name}{venue}")
        return summaries

    @staticmethod
    def _merge_unique(labels: list[str]) -> list[str]:
        merged: list[str] = []
        for label in labels:
            if label and label not in merged:
                merged.append(label)
        return merged

    @staticmethod
    def _ticketmaster_radius_km(movement_mode: str) -> float:
        return 10.0 if movement_mode == "ride" else 3.0

    @staticmethod
    def _ticketmaster_cache_key(latitude: float, longitude: float, movement_mode: str) -> str:
        precision = 2 if movement_mode == "ride" else 3
        return f"{movement_mode}:{round(latitude, precision)}:{round(longitude, precision)}"

    @staticmethod
    def _parse_place(data: dict[str, Any]) -> PlaceCandidate:
        display_name = data.get("displayName", {})
        if isinstance(display_name, dict):
            display_text = display_name.get("text") or "Unknown place"
        else:
            display_text = display_name or "Unknown place"
        return PlaceCandidate(
            place_id=data.get("id"),
            display_name=display_text,
            formatted_address=data.get("formattedAddress"),
            primary_type=data.get("primaryType"),
            types=data.get("types", []),
        )

    @staticmethod
    def _parse_event(data: dict[str, Any]) -> EventCandidate:
        dates = data.get("dates", {}).get("start", {})
        local_time = dates.get("localTime")
        if isinstance(local_time, str) and len(local_time) >= 5:
            local_time = local_time[:5]
        classifications = data.get("classifications") or []
        first_classification = classifications[0] if classifications else {}
        venues = data.get("_embedded", {}).get("venues") or []
        first_venue = venues[0] if venues else {}
        return EventCandidate(
            name=data.get("name") or "Nearby event",
            venue_name=first_venue.get("name"),
            start_local_date=dates.get("localDate"),
            start_local_time=local_time,
            segment_name=(first_classification.get("segment") or {}).get("name"),
            genre_name=(first_classification.get("genre") or {}).get("name"),
        )
