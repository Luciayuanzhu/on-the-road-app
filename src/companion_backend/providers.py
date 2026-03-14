from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ContextSnapshot:
    nearby_places: list[str]
    nearby_activities: list[str]
    title_hint: str


class ContextProvider:
    def enrich(
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
        )
