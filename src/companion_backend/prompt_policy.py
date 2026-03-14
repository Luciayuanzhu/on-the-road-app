from __future__ import annotations

from .providers import ContextSnapshot
from .schemas import ResponseLength, UserSettings


def build_system_prompt(
    settings: UserSettings,
    context: ContextSnapshot,
    destination_label: str | None,
    visual_assist_active: bool,
    transcript_lines: list[str],
) -> str:
    preferences = ", ".join(settings.content_preferences) or "general local context"
    transcript_summary = " | ".join(transcript_lines[-4:]) if transcript_lines else "none yet"
    destination_line = destination_label or "no destination set"
    return (
        "You are the live companion. "
        f"movement_mode={settings.movement_mode}; "
        f"response_length={settings.response_length}; "
        f"companion_style={settings.companion_style}; "
        f"content_preferences={preferences}; "
        f"visual_assist_active={str(visual_assist_active).lower()}; "
        f"destination={destination_line}; "
        f"nearby_places={', '.join(context.nearby_places)}; "
        f"nearby_activities={', '.join(context.nearby_activities)}; "
        f"recent_transcript={transcript_summary}"
    )


def _join_focus(settings: UserSettings) -> str:
    if not settings.content_preferences:
        return "local context"
    return ", ".join(pref.replace("_", " ") for pref in settings.content_preferences[:3])


def _style_prefix(settings: UserSettings) -> str:
    return "Guide note" if settings.companion_style == "guide_like" else "Quick take"


def _length_suffix(response_length: ResponseLength, extra: str) -> str:
    if response_length == ResponseLength.short:
        return ""
    if response_length == ResponseLength.medium:
        return f" {extra}"
    return f" {extra} Keep moving and I’ll layer in more context as things change."


def build_location_response(
    settings: UserSettings,
    context: ContextSnapshot,
    destination_label: str | None,
) -> str:
    base = (
        f"{_style_prefix(settings)}: you’re near {context.nearby_places[0]}. "
        f"I’m watching for { _join_focus(settings) } as your {settings.movement_mode} session continues."
    )
    if destination_label:
        base += f" I’m also keeping {destination_label} in frame."
    return base + _length_suffix(
        settings.response_length,
        f"Nearby activity context includes {', '.join(context.nearby_activities[:2])}.",
    )


def build_destination_response(
    settings: UserSettings,
    destination_label: str,
    context: ContextSnapshot,
) -> str:
    return (
        f"{_style_prefix(settings)}: destination updated to {destination_label}. "
        f"I’ll bias the companion toward {', '.join(context.nearby_places[:2])} and { _join_focus(settings) }."
        + _length_suffix(
            settings.response_length,
            f"I’m also tracking {', '.join(context.nearby_activities[:2])}.",
        )
    )


def build_text_response(
    user_text: str,
    settings: UserSettings,
    context: ContextSnapshot,
    destination_label: str | None,
    visual_assist_active: bool,
) -> str:
    visual_line = (
        " Camera assist is active, so I can fold in what you explicitly show me."
        if visual_assist_active
        else ""
    )
    destination_line = f" Destination context: {destination_label}." if destination_label else ""
    return (
        f"{_style_prefix(settings)}: on “{user_text}”, the clearest read is around {context.nearby_places[0]}. "
        f"I’ll keep the answer tuned for { _join_focus(settings) }.{destination_line}{visual_line}"
        + _length_suffix(
            settings.response_length,
            f"Relevant nearby activity includes {', '.join(context.nearby_activities[:2])}.",
        )
    )
