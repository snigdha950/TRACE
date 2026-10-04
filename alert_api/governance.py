from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GOVERNANCE_VERSION = "trace-operational-safety/1.0"

REQUIRED_RELEASE_FLAGS = {
    "prototype_alert": True,
    "official_warning": False,
    "calibrated_probability": False,
}


@dataclass(frozen=True)
class GovernanceDecision:
    machine_route: str
    human_review_required: bool
    public_warning_release_allowed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GOVERNANCE_VERSION,
            "machine_route": self.machine_route,
            "human_review_required": self.human_review_required,
            "public_warning_release_allowed": self.public_warning_release_allowed,
            "reasons": list(self.reasons),
            **REQUIRED_RELEASE_FLAGS,
        }


def evaluate_operational_safety(*, provenance_ok: bool, research_route_ready: bool, official_authority_approval: bool = False) -> dict[str, Any]:
    """Fail-closed safety policy for the TRACE research prototype.

    This does not authorize a meteorological warning. It only determines whether
    research-prototype output may proceed to an internal human-review queue.
    """
    reasons: list[str] = []
    if not provenance_ok:
        reasons.append("frozen evidence/provenance integrity is not verified")
        return GovernanceDecision(
            machine_route="BLOCKED_FAIL_CLOSED",
            human_review_required=True,
            public_warning_release_allowed=False,
            reasons=tuple(reasons),
        ).as_dict()

    if not research_route_ready:
        reasons.append("research route prerequisites are not satisfied")
        return GovernanceDecision(
            machine_route="BLOCKED_RESEARCH_ROUTE_NOT_READY",
            human_review_required=True,
            public_warning_release_allowed=False,
            reasons=tuple(reasons),
        ).as_dict()

    reasons.append("prototype output may enter internal human review")
    reasons.append("TRACE is not an official warning authority")
    if official_authority_approval:
        reasons.append("authority approval input is recorded, but this prototype still cannot self-release an official warning")
    return GovernanceDecision(
        machine_route="INTERNAL_HUMAN_REVIEW_QUEUE",
        human_review_required=True,
        public_warning_release_allowed=False,
        reasons=tuple(reasons),
    ).as_dict()


def replay_stream_records(records: list[dict[str, Any]], *, provenance_ok: bool, research_route_ready: bool) -> dict[str, Any]:
    """Deterministic controlled replay of already-produced alert records.

    This is a software-operability test, not live NWP ingestion or an operational
    streaming claim.
    """
    decision = evaluate_operational_safety(
        provenance_ok=provenance_ok,
        research_route_ready=research_route_ready,
    )
    replayed = []
    for idx, record in enumerate(records):
        replayed.append({
            "sequence": idx,
            "event_id": record.get("event_id"),
            "forecast_lead_hours": record.get("forecast_lead_hours"),
            "prototype_severity": record.get("severity"),
            "route": decision["machine_route"],
            "human_review_required": True,
            "official_warning": False,
        })
    return {
        "schema": "trace-controlled-stream-replay/1.0",
        "mode": "OFFLINE_CONTROLLED_REPLAY_NOT_LIVE_STREAMING",
        "record_count": len(replayed),
        "records": replayed,
        "governance": decision,
        "operational_streaming_validated": False,
        "boundary": "This replay verifies ordering and fail-closed governance for saved prototype outputs. It does not demonstrate continuous operational NWP ingestion, official-warning issuance, authentication, rate limiting, or 24/7 monitoring.",
    }
