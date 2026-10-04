from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from provider_ingest import validate_provider_metadata
from governance import evaluate_operational_safety, replay_stream_records

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data/alerts.json").read_text())
STAGE2 = json.loads((HERE / "data/stage2_benchmark.json").read_text())
ALERTS = {int(a["forecast_lead_hours"]): a for a in DATA["alerts"]}
ROOT = HERE.parent
FANI_TEMPORAL_PATH = ROOT / "evidence_registry" / "fani_temporal" / "FANI_TEMPORAL_EVIDENCE.json"
FANI_TEMPORAL = json.loads(FANI_TEMPORAL_PATH.read_text())
HAZARD_EVENT_TARGETS_PATH = ROOT / "evidence_registry" / "OFFICIAL_HAZARD_EVENT_TARGETS_2019.json"
HAZARD_EVENT_TARGETS = json.loads(HAZARD_EVENT_TARGETS_PATH.read_text())
FANI_TEMPORAL_RECORD_HASHES = {
    "FANI_TEMPORAL_EVIDENCE.json": "1bc6b717473a6292c0c45c7f832ee63a5807374101e80361da03d26aaf90b1a1",
    "TEMPORAL_FANI_RESULTS.md": "8896f0da3ec5c476219e016d9b3d54e3a65f11a0c69934b65e1f4cb7f6dede07",
    "TRACE_CURRENT_STATE_FANI.json": "7e89eaedd70f47726ef1df4730f3eb7475de67f8c2794701788f1f264d81f814",
}

API_VERSION = "1.8.0-prototype"
MODEL_VERSION = "trace-wind-spherical-gnn-bulbul-frozen-5seed"
EVIDENCE_VERSION = "bulbul2019-alert-evidence-v1-fani-temporal-v1-stage2-controlled-hrrr-v3-governance-v1"
SEVERITY_RULE = "TRACE_SEVERITY_RULE_V1_NOT_CALIBRATED"
DEFAULT_FLAGS = {
    "prototype_alert": True,
    "official_warning": False,
    "calibrated_probability": False,
}
CLAIM_BOUNDARIES = [
    "Research prototype; not an operational warning service.",
    "Severity tiers are transparent prototype categories, not official IMD/NCMRWF warnings.",
    "The learned classifier score is not a calibrated probability.",
    "GEFSv12 is a prototype substitute; direct NEPS-G validation is not included.",
    "The anomaly core is a model evidence coordinate, not an exact cyclone-center estimate.",
    "The 5 km map ring is a UI locator marker only, not a predicted hazard/impact radius and not Stage-2 output.",
    "Stage-2 HRRR evidence is controlled reconstruction, not same-event BULBUL downscaling skill.",
    "Sequential FANI wind-footprint tracking was executed, but its centroids are not validated cyclone centers and errors remain large.",
    "Physics-aware constraints and diagnostics were tested; full thermodynamic/PDE conservation is not demonstrated.",
    "All prototype alerts require human review; the API never authorizes public or official warning release.",
]
THRESHOLDS = {
    "fixed_wind_sensitivity_ms": 12.5,
    "fixed_wind_primary_ms": 17.5,
    "efi_sensitivity": 0.5,
    "efi_primary": 0.8,
    "member_support_threshold": 0.4,
    "seed_consensus_threshold": 0.6,
    "prototype_severe_seed_agreement_threshold": 0.8,
}
RULE_PROVENANCE = (
    "Transparent prototype rule over frozen TRACE evidence. The 0.6 seed-consensus gate "
    "comes from the frozen alert footprint consensus. The 0.8 severe-tier seed agreement is "
    "a documented alert-layer category threshold, not a calibrated probability or official warning policy."
)

WEAKNESS_MITIGATION = [
    {
        "weakness": "Direct NEPS-G operational validation missing",
        "v1_5_improvement": "Added a tested provider-ingest schema validator, CLI preflight, and frozen NEPS-G validation protocol in addition to the provider-adapter map.",
        "v1_6_improvement": "Rating-based communication polish: clearer slide claim boundaries, evidence notes, and safe-claim wording; scientific metrics unchanged.",
        "status_after_v1_5": "INGEST_VALIDATOR_TESTED; DIRECT_SKILL_DATA_ACCESS_BLOCKED",
        "remaining_scientific_gap": "No direct NEPS-G skill metric is bundled; GEFSv12 remains a transparent substitute.",
        "what_closes_it": "Run the frozen Stage-1 pipeline on historical NEPS-G/NCUM events with the same no-retuning protocol.",
    },
    {
        "weakness": "Multi-hazard validation missing",
        "v1_5_improvement": "Added a tested hazard-generic upper/lower-tail footprint evaluator plus a frozen heat-wave validation protocol; real second-hazard skill remains unexecuted.",
        "status_after_v1_5": "CYCLONE_WIND_VALIDATED; GENERIC_HEAT_COLD_EVALUATOR_TESTED; REAL_SECOND_HAZARD_NOT_VALIDATED",
        "remaining_scientific_gap": "No held-out heat-dome, cold-wave, hail or frost skill metric is bundled.",
        "what_closes_it": "Add at least one held-out heat/cold event with ERA5/IMDAA climatology and official impact labels.",
    },
    {
        "weakness": "Same-event Stage-1 to Stage-2 India validation missing",
        "v1_5_improvement": "Added /api/v1/integration/audit to expose the exact handoff contract, route decision and same-event validation boundary.",
        "status_after_v1_5": "INTEGRATED_SOFTWARE_CONTRACT_VERIFIED; SAME_EVENT_MET_SKILL_NOT_CLAIMED",
        "remaining_scientific_gap": "Stage-2 HRRR reconstruction is controlled evidence, not BULBUL/India same-event downscaling skill.",
        "what_closes_it": "Pair an Indian high-resolution reference field with a same-event coarse forecast slice and rerun 12->5 km validation.",
    },
    {
        "weakness": "GNN message-passing advantage is inconclusive",
        "v1_5_improvement": "Added a cross-event architecture evidence gate using frozen FANI and BULBUL aggregates; it explicitly retains message passing as experimental rather than pretending it is a proven default.",
        "status_after_v1_5": "FROZEN_CROSS_EVENT_ARCHITECTURE_GATE_AVAILABLE",
        "remaining_scientific_gap": "Full graph message passing still does not have a stable independent advantage over no-message nonlinear ablation.",
        "what_closes_it": "More diverse held-out events and ablations showing when neighbor exchange improves contiguous footprints or reduces false objects.",
    },
    {
        "weakness": "Screening judges may miss the value quickly",
        "v1_5_improvement": "Added a 30-second screening brief and surfaced recovered sequential FANI tracking evidence in the API/dashboard so judges see measured wins and boundaries immediately.",
        "status_after_v1_5": "SCREENING_BRIEF_PLUS_EXECUTED_TRACKING_EVIDENCE",
        "remaining_scientific_gap": "Better communication cannot replace missing NEPS-G/multi-hazard evidence.",
        "what_closes_it": "Add the missing scientific evaluations, then update the one-screen evidence map.",
    },
]

MODEL_SELECTION_DEFENCE = {
    "schema": "trace-model-selection-defence/1.0",
    "core_position": "TRACE keeps the spherical learned detector because it improves held-out BULBUL IoU over analytical/linear baselines and preserves Earth-geometry handling; it does not claim graph message passing is independently proven necessary.",
    "bulbul_iou_evidence": {
        "analytical_sot_climate99_iou": 0.281,
        "wind_linear_iou": 0.287,
        "wind_no_message_nonlinear_iou": 0.423,
        "wind_spherical_gnn_iou": 0.430,
        "multivariable_gnn_iou": 0.397,
    },
    "defensible_claims": [
        "Learned nonlinear detection materially improves over analytical SOT and linear baseline on held-out BULBUL.",
        "Spherical/icosahedral representation directly matches the official PS geometry requirement.",
        "Full message-passing GNN is retained as an architecture-compatible path, not as a proven universal winner.",
    ],
    "non_claims": [
        "No claim that message passing is always necessary.",
        "No claim that multivariable context always improves nonlinear IoU/AP.",
        "No claim that the model is operationally validated on NEPS-G.",
    ],
}

HAZARD_READINESS = {
    "schema": "trace-hazard-readiness/1.0",
    "validated_now": [
        {"hazard": "severe_cyclone_or_extreme_wind", "status": "HELD_OUT_EVIDENCE_AVAILABLE", "evidence": "FANI/BULBUL wind-focused Stage-1 evaluation + January negative control"}
    ],
    "tested_generic_evaluator": {
        "module": "hazard_footprint.py",
        "capability": "upper/lower-tail ensemble-support footprint scoring with IoU/precision/recall",
        "unit_tested": True,
        "scientific_skill_validation": False,
    },
    "software_ready_not_validated": [
        {"hazard": "heat_dome", "required_variables": ["2m_temperature", "geopotential_or_height", "climatological_percentile"], "validation_status": "NOT_EXECUTED_IN_BUNDLE", "frozen_protocol": "validation_protocols/HEATWAVE_VALIDATION_PROTOCOL_V1.json"},
        {"hazard": "cold_wave_or_frost", "required_variables": ["2m_temperature", "minimum_temperature", "surface_or_crop-zone mask", "climatological_percentile"], "validation_status": "NOT_EXECUTED_IN_BUNDLE"},
        {"hazard": "hail_or_severe_convective_risk", "required_variables": ["instability proxy", "wind shear proxy", "precipitation/reflectivity proxy"], "validation_status": "NOT_EXECUTED_IN_BUNDLE"},
    ],
    "boundary": "This is a typed expansion plan and judge-facing readiness map; it is not a claim of multi-hazard skill.",
}

DATA_ADAPTERS = {
    "schema": "trace-data-adapter-readiness/1.0",
    "implemented_prototype_provider": {
        "provider": "NOAA GEFSv12 retrospective ensemble",
        "role": "public substitute for medium-range ensemble prototyping",
        "status": "IMPLEMENTED_AND_MEASURED",
    },
    "operational_target_provider": {
        "provider": "NCMRWF NEPS-G / NCUM-style Indian operational ecosystem",
        "role": "official-PS target input family",
        "status": "TESTED_INGEST_CONTRACT; DIRECT_VALIDATION_NOT_INCLUDED",
        "expected_contract": ["lat", "lon", "forecast_initialization_time", "valid_time", "ensemble_member", "pressure/surface variable", "units", "grid spacing metadata"],
        "official_public_description": {
            "nominal_horizontal_resolution_km": 12,
            "forecast_horizon_days": 10,
            "ensemble_description": "22 perturbed members plus control in the documented lagged ensemble configuration",
            "sources": [
                "https://nwp.ncmrwf.gov.in/HomePage/index.php",
                "https://www.ncmrwf.gov.in/ncmrwf/NEPSG-Writeup-for-WEB-June2020.pdf",
            ],
        },
        "research_data_access": "https://rds.ncmrwf.gov.in/login",
        "frozen_validation_protocol": "validation_protocols/NEPSG_VALIDATION_PROTOCOL_V1.json",
    },
    "adapter_principle": "Provider I/O is separated from TRACE scoring/evidence logic, so target-provider replacement should not change frozen thresholds or model-selection rules without a new frozen validation protocol.",
}

INTEGRATION_AUDIT = {
    "schema": "trace-stage1-stage2-integration-audit/1.0",
    "stage1_to_stage2_contract": "READY_SOFTWARE_HANDOFF",
    "contract_fields": ["event_id", "valid_time", "lead_hours", "anomaly_core_coordinate", "crop_bbox", "requested_input_grid_km", "requested_output_grid_km", "preferred_route", "diffusion_route"],
    "current_route_decision": "deterministic_mean_plus_exact_coarse_projection",
    "stage2_evidence_type": "controlled HRRR 12->5 km reconstruction benchmark",
    "same_event_india_validation": False,
    "safe_interpretation": "Software composability is verified; same-event meteorological downscaling skill is future validation.",
}

MODEL_SELECTION_GATE = {
    "schema": "trace-model-selection-gate/1.0",
    "frozen_event_evidence": {
        "FANI_2019_all_lead_climate99_iou": {"no_message": 0.18286, "spherical_gnn": 0.18035, "gnn_minus_no_message": -0.00251},
        "BULBUL_2019_wind_climate99_iou": {"no_message": 0.42280, "spherical_gnn": 0.42980, "gnn_minus_no_message": 0.00700},
    },
    "decision": "RETAIN_MESSAGE_PASSING_AS_EXPERIMENTAL_BRANCH_NOT_PROVEN_DEFAULT",
    "reason": "Frozen held-out events disagree on the incremental value of message passing. Spherical geometry and nonlinear learning remain useful, but the graph-message increment is context-dependent and small in current evidence.",
    "scientific_significance_test": False,
    "no_retraining_or_retuning_for_this_gate": True,
}

SCREENING_BRIEF = {
    "schema": "trace-screening-brief/1.0",
    "thirty_second_position": "TRACE automatically detects and tracks medium-range extreme-weather footprints on spherical ensemble fields, then routes detected regions to an evidence-gated 12-to-5 km refinement layer and a transparent prototype alert API.",
    "three_measured_proofs": [
        "Held-out BULBUL wind spherical GNN: Climate99 IoU about 0.430, AP about 0.607; analytical SOT about 0.281.",
        "Executed FANI sequential wind-footprint tracking: nine +72 to +120 h frames; best member covered 9/9 frames with mean centroid error about 329 km, while large errors are retained.",
        "Controlled four-event HRRR 12-to-5 km deterministic route: speed MAE about 0.3178 m/s, q99 absolute error about 0.1673 m/s, threshold IoU about 0.5846, exact coarse U/V consistency near machine zero.",
    ],
    "three_boundaries": [
        "GEFSv12 substitutes for NEPS-G; direct target-provider skill is not yet measured.",
        "Heat/cold/hail/frost real-event skill is not yet measured.",
        "Stage-2 is controlled HRRR reconstruction, not same-event India downscaling validation.",
    ],
    "fast_proof_endpoints": ["/api/v1/tracking/evidence", "/api/v1/model-selection/gate", "/api/v1/provenance"],
}


app = FastAPI(
    title="TRACE Extreme-Weather Alert API",
    version=API_VERSION,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url=None,
    description=(
        "Research-prototype API built on frozen TRACE evidence. Severity tiers are prototype "
        "heuristics, not official meteorological warnings or calibrated probabilities. All outputs require human review and cannot self-authorize public warning release."
    ),
)




def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_integrity() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for lead, alert_obj in sorted(ALERTS.items()):
        sample = HERE / "source_evidence" / f"BULBUL_2019_{lead}.npz"
        expected = alert_obj["provenance"]["sample_sha256"]
        actual = sha256_file(sample) if sample.exists() else None
        checks.append({"kind": "sample", "lead_hours": lead, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    expected_ckpts = next(iter(ALERTS.values()))["provenance"]["checkpoint_sha256_by_seed"]
    for seed, expected in sorted(expected_ckpts.items(), key=lambda kv: int(kv[0])):
        ckpt = HERE / "source_evidence" / f"gnn_seed{seed}.pt"
        actual = sha256_file(ckpt) if ckpt.exists() else None
        checks.append({"kind": "checkpoint", "seed": int(seed), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    expected_alerts_hash = "89a92f51c82ba823fb0449314582947e0510abcee8a01a797897642fc0552d73"
    actual_alerts_hash = sha256_file(HERE / "data/alerts.json")
    checks.append({"kind": "frozen_alert_collection", "expected_sha256": expected_alerts_hash, "actual_sha256": actual_alerts_hash, "match": actual_alerts_hash == expected_alerts_hash})
    fani_dir = ROOT / "evidence_registry" / "fani_temporal"
    for name, expected in FANI_TEMPORAL_RECORD_HASHES.items():
        artifact = fani_dir / name
        actual = sha256_file(artifact) if artifact.exists() else None
        checks.append({"kind": "supplemental_executed_fani_temporal_record", "artifact": name, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    passed = all(c["match"] for c in checks)
    return {
        "schema": "trace-provenance-integrity/1.0",
        "status": "PASS" if passed else "FAIL",
        "all_frozen_artifacts_verified": passed,
        "checks": checks,
        "note": "Integrity verifies bundled frozen inputs/checkpoints, the frozen alert collection, and recovered executed FANI temporal evidence records; record integrity does not erase the stated scientific limitations.",
    }


def polygon_bbox(poly: dict[str, Any]) -> dict[str, float]:
    ring = validate_geojson_polygon(poly)["coordinates"][0]
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return {"west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats)}




def metadata() -> dict[str, Any]:
    return {
        "schema": "trace-api-metadata/1.5",
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "evidence_version": EVIDENCE_VERSION,
        **DEFAULT_FLAGS,
        "limitations": CLAIM_BOUNDARIES,
    }


def enrich(obj: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(obj)
    out.update({k: v for k, v in metadata().items() if k not in out})
    return out


def enrich_alert(alert: dict[str, Any]) -> dict[str, Any]:
    out = enrich(alert)
    out["rule_provenance"] = RULE_PROVENANCE
    out["severity_thresholds"] = THRESHOLDS
    return out


def validate_geojson_polygon(poly: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(poly, dict) or poly.get("type") != "Polygon":
        raise ValueError("macro_footprint_geojson must be a GeoJSON Polygon")
    coords = poly.get("coordinates")
    if not isinstance(coords, list) or not coords or not isinstance(coords[0], list):
        raise ValueError("GeoJSON Polygon coordinates must contain at least one ring")
    ring = coords[0]
    if len(ring) < 4:
        raise ValueError("GeoJSON Polygon outer ring must contain at least four positions")
    if ring[0] != ring[-1]:
        raise ValueError("GeoJSON Polygon outer ring must be closed")
    for point in ring:
        if not isinstance(point, list | tuple) or len(point) < 2:
            raise ValueError("Each GeoJSON position must be [lon, lat]")
        lon, lat = point[0], point[1]
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            raise ValueError("GeoJSON lon/lat must be numeric")
        if not (-180 <= float(lon) <= 180 and -90 <= float(lat) <= 90):
            raise ValueError("GeoJSON lon/lat outside valid range")
    return poly


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True, populate_by_name=True)


class EvidenceInput(StrictBase):
    max_ensemble_mean_wind_ms: FiniteFloat = Field(ge=0, description="Maximum ensemble-mean wind speed inside the candidate footprint.")
    max_efi: FiniteFloat = Field(ge=-1, le=1, description="Extreme Forecast Index. Positive extremes approach +1.")
    max_member_support_fraction: FiniteFloat = Field(ge=0, le=1)
    mean_seed_agreement: FiniteFloat = Field(ge=0, le=1)
    max_sot: FiniteFloat = Field(description="Shift of Tails evidence. Positive values corroborate tail shift.")


class SeverityResponse(StrictBase):
    schema_id: Literal["trace-severity-response/1.5"] = Field("trace-severity-response/1.5", alias="schema", serialization_alias="schema")
    api_version: str = API_VERSION
    model_version: str = MODEL_VERSION
    evidence_version: str = EVIDENCE_VERSION
    prototype_alert: bool = True
    official_warning: bool = False
    calibrated_probability: bool = False
    severity: Literal["LOW", "MODERATE", "SEVERE"]
    rule: str
    rule_provenance: str
    thresholds: dict[str, float]
    explanation: list[str]


class AlertRequest(StrictBase):
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    valid_time: AwareDatetime
    forecast_lead_hours: int = Field(ge=72, le=240, description="Medium-range forecast lead, 3 to 10 days.")
    core_lat: FiniteFloat = Field(ge=-90, le=90)
    core_lon: FiniteFloat = Field(ge=-180, le=180)
    macro_footprint_area_km2: FiniteFloat = Field(gt=0, le=510_100_000)
    macro_footprint_geojson: dict[str, Any]
    evidence: EvidenceInput

    @field_validator("macro_footprint_geojson")
    @classmethod
    def _polygon(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_geojson_polygon(v)


class ProviderMetadataRequest(StrictBase):
    provider_name: str = Field(min_length=1, max_length=128)
    dimensions: dict[str, int]
    variables: dict[str, Any] | list[str]
    coordinates: dict[str, Any] | list[str] = Field(default_factory=dict)
    initialization_time: AwareDatetime | None = None
    valid_times: list[AwareDatetime] = Field(default_factory=list)
    lead_hours: list[FiniteFloat] = Field(default_factory=list)


class ReplayRecord(StrictBase):
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    forecast_lead_hours: int = Field(ge=72, le=240)
    severity: Literal["LOW", "MODERATE", "SEVERE"]


class ControlledReplayRequest(StrictBase):
    records: list[ReplayRecord] = Field(min_length=1, max_length=100)


class TrackFrame(StrictBase):
    valid_time: AwareDatetime
    forecast_lead_hours: int = Field(ge=0, le=384)
    core_lat: FiniteFloat = Field(ge=-90, le=90)
    core_lon: FiniteFloat = Field(ge=-180, le=180)
    macro_footprint_geojson: dict[str, Any]

    @field_validator("macro_footprint_geojson")
    @classmethod
    def _polygon(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_geojson_polygon(v)


class TrackRequest(StrictBase):
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    frames: list[TrackFrame] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def _strictly_increasing_times(self):
        times = [f.valid_time for f in self.frames]
        if len(set(times)) != len(times):
            raise ValueError("track frame valid_time values must be unique")
        if times != sorted(times):
            raise ValueError("track frames must be ordered by increasing valid_time")
        return self


def classify(e: EvidenceInput) -> SeverityResponse:
    severe = (
        e.max_ensemble_mean_wind_ms >= THRESHOLDS["fixed_wind_primary_ms"]
        and e.max_efi >= THRESHOLDS["efi_primary"]
        and e.max_member_support_fraction >= THRESHOLDS["member_support_threshold"]
        and e.mean_seed_agreement >= THRESHOLDS["prototype_severe_seed_agreement_threshold"]
    )
    moderate = (
        e.max_ensemble_mean_wind_ms >= THRESHOLDS["fixed_wind_sensitivity_ms"]
        and e.max_efi >= THRESHOLDS["efi_sensitivity"]
        and e.mean_seed_agreement >= THRESHOLDS["seed_consensus_threshold"]
        and e.max_sot > 0
    )
    sev = "SEVERE" if severe else ("MODERATE" if moderate else "LOW")
    why: list[str] = []
    if e.max_ensemble_mean_wind_ms >= THRESHOLDS["fixed_wind_primary_ms"]:
        why.append("primary 17.5 m/s wind threshold met")
    elif e.max_ensemble_mean_wind_ms >= THRESHOLDS["fixed_wind_sensitivity_ms"]:
        why.append("12.5 m/s sensitivity threshold met")
    if e.max_efi >= THRESHOLDS["efi_primary"]:
        why.append("primary EFI 0.8 threshold met")
    elif e.max_efi >= THRESHOLDS["efi_sensitivity"]:
        why.append("EFI 0.5 sensitivity threshold met")
    if e.max_member_support_fraction >= THRESHOLDS["member_support_threshold"]:
        why.append(">=40% ensemble-member support")
    if e.mean_seed_agreement >= THRESHOLDS["prototype_severe_seed_agreement_threshold"]:
        why.append("strong five-seed agreement for severe-tier label")
    elif e.mean_seed_agreement >= THRESHOLDS["seed_consensus_threshold"]:
        why.append("five-seed agreement gate met")
    if e.max_sot > 0:
        why.append("positive SOT corroboration")
    if not why:
        why.append("no moderate/severe evidence gate was met")
    return SeverityResponse(
        severity=sev,
        rule=SEVERITY_RULE,
        rule_provenance=RULE_PROVENANCE,
        thresholds=THRESHOLDS,
        explanation=why,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    integrity = provenance_integrity()
    return enrich({"status": "ok", "service": "TRACE alert API", "mode": "research_prototype", "alerts_loaded": len(ALERTS), "frozen_evidence_integrity": integrity["status"]})


@app.get("/api/v1/status")
def status() -> dict[str, Any]:
    return enrich(
        {
            "implemented": [
                "frozen Stage-1 alert ingestion",
                "geographic macro footprint",
                "5 km UI locator ring (not a hazard radius)",
                "LOW/MODERATE/SEVERE prototype severity",
                "Stage1->refinement request contract",
                "Stage-2 evidence-gated route summary",
                "strict generic alert evaluation schemas",
                "verifiable frozen-evidence provenance hashes",
                "generic temporal track-object builder for new caller/model frames",
                "recovered executed FANI sequential wind-footprint tracking evidence (+72 to +120 h)",
                "provider-adapter readiness map plus tested NEPS-G-compatible ingest metadata validator",
                "hazard-readiness matrix plus tested hazard-generic upper/lower-tail footprint evaluator",
                "model-selection defence that avoids overstating graph-message superiority",
                "Stage1->Stage2 integration audit with same-event boundary",
                "fail-closed operational-safety policy with mandatory human review",
                "offline controlled stream replay for saved prototype alerts",
                "official IMD heat/cold event-target registry for future no-test-tuning validation",
            ],
            "severity_rule": SEVERITY_RULE,
            "severity_rule_provenance": RULE_PROVENANCE,
            "thresholds": THRESHOLDS,
            "boundaries": CLAIM_BOUNDARIES,
        }
    )


@app.get("/ready")
def ready() -> JSONResponse:
    report = provenance_integrity()
    code = 200 if report["all_frozen_artifacts_verified"] else 503
    return JSONResponse(status_code=code, content=enrich({"status": "ready" if code == 200 else "not_ready", "integrity": report}))


@app.get("/api/v1/provenance")
def provenance() -> dict[str, Any]:
    return enrich(provenance_integrity())


@app.get("/api/v1/lead-evolution")
def lead_evolution() -> dict[str, Any]:
    ordered = [ALERTS[k] for k in sorted(ALERTS)]
    valid_times = sorted({a["valid_time"] for a in ordered})
    return enrich({
        "schema": "trace-lead-evolution/1.0",
        "event_id": ordered[0]["event_id"],
        "verification_valid_times": valid_times,
        "independent_verification_time_count": len(valid_times),
        "forecast_snapshot_count": len(ordered),
        "shared_valid_time_across_snapshots": len(valid_times) == 1,
        "statistical_note": "The headline BULBUL score summarizes correlated lead forecasts and five frozen seeds; the three leads are not three independent meteorological verification cases.",
        "forecast_snapshots": [
            {
                "forecast_lead_hours": a["forecast_lead_hours"],
                "severity": a["severity"],
                "core": a["anomaly_core_coordinate"],
                "macro_footprint_area_km2": a["macro_footprint_area_km2"],
                "mean_seed_agreement": a["model_evidence"]["mean_seed_agreement_over_footprint"],
            } for a in ordered
        ],
        "interpretation": "These frozen BULBUL snapshots verify the same valid time at different forecast leads. They show lead sensitivity/robustness, NOT a physical storm trajectory across time.",
        "scientific_trajectory_validation": False,
    })


@app.post("/api/v1/tracks/build")
def build_track(req: TrackRequest) -> dict[str, Any]:
    frames = []
    line = []
    for f in req.frames:
        line.append([float(f.core_lon), float(f.core_lat)])
        frames.append({
            "valid_time": f.valid_time.astimezone(timezone.utc).isoformat(),
            "forecast_lead_hours": f.forecast_lead_hours,
            "anomaly_core_coordinate": {"lat": float(f.core_lat), "lon": float(f.core_lon)},
            "temporal_bbox": polygon_bbox(f.macro_footprint_geojson),
            "macro_footprint_geojson": f.macro_footprint_geojson,
        })
    return enrich({
        "schema": "trace-track-object/1.0",
        "event_id": req.event_id,
        "frames": frames,
        "core_trajectory_geojson": {"type": "LineString", "coordinates": line},
        "software_tracking_capability": True,
        "scientific_validation": False,
        "boundary": "This endpoint constructs a time-ordered track object from caller/model frames. The bundled frozen BULBUL alert set does not itself validate a physical trajectory because its three snapshots share one valid time. Separate recovered FANI evidence does contain sequential valid-time wind-footprint tracking, but not validated cyclone-center reconstruction.",
    })


@app.get("/api/v1/tracking/evidence")
def tracking_evidence() -> dict[str, Any]:
    return enrich(copy.deepcopy(FANI_TEMPORAL))


@app.post("/api/v1/provider-ingest/validate")
def provider_ingest_validate(req: ProviderMetadataRequest) -> dict[str, Any]:
    return enrich(validate_provider_metadata(req.model_dump()))


@app.get("/api/v1/model-selection/gate")
def model_selection_gate() -> dict[str, Any]:
    return enrich(MODEL_SELECTION_GATE)


@app.get("/api/v1/screening-brief")
def screening_brief() -> dict[str, Any]:
    return enrich(SCREENING_BRIEF)


@app.get("/api/v1/integration/readiness")
def integration_readiness() -> dict[str, Any]:
    return enrich({
        "schema": "trace-integration-readiness/1.0",
        "software_handoff_ready": True,
        "primary_stage2_route": INTEGRATION_AUDIT["current_route_decision"],
        "same_event_india_meteorological_validation": False,
        "operational_route_allowed": False,
        "research_route_allowed": True,
        "blocking_reason": "No same-event Indian high-resolution reference is bundled; controlled HRRR reconstruction cannot be relabeled as India skill.",
        "frozen_protocol": "validation_protocols/SAME_EVENT_STAGE2_PROTOCOL_V1.json",
    })


@app.get("/api/v1/governance")
def governance_status() -> dict[str, Any]:
    integrity = provenance_integrity()["all_frozen_artifacts_verified"]
    decision = evaluate_operational_safety(
        provenance_ok=integrity,
        research_route_ready=True,
    )
    return enrich({
        "schema": "trace-governance-status/1.0",
        "policy": decision,
        "human_review_required": True,
        "official_warning_release_allowed": False,
        "warning_authority": "EXTERNAL_TO_TRACE",
        "fail_closed_on_provenance_failure": True,
        "authentication_rate_limiting_24x7_monitoring": "NOT_IMPLEMENTED_RESEARCH_PROTOTYPE",
    })


@app.get("/api/v1/operational-safety")
def operational_safety() -> dict[str, Any]:
    integrity = provenance_integrity()["all_frozen_artifacts_verified"]
    research_ready = True
    return enrich(evaluate_operational_safety(
        provenance_ok=integrity,
        research_route_ready=research_ready,
    ))


@app.post("/api/v1/stream/replay")
def controlled_stream_replay(req: ControlledReplayRequest) -> dict[str, Any]:
    integrity = provenance_integrity()["all_frozen_artifacts_verified"]
    records = [r.model_dump() for r in req.records]
    return enrich(replay_stream_records(records, provenance_ok=integrity, research_route_ready=True))


@app.get("/api/v1/hazard-event-targets")
def hazard_event_targets() -> dict[str, Any]:
    return enrich(copy.deepcopy(HAZARD_EVENT_TARGETS))


@app.get("/api/v1/data-blockers")
def data_blockers() -> dict[str, Any]:
    return enrich({
        "schema": "trace-data-blockers/1.0",
        "blockers": [
            {
                "gap": "direct_nepsg_validation",
                "status": "BLOCKED_BY_TARGET_DATA_ACCESS",
                "implemented_readiness": "tested ingest validator + frozen validation protocol",
                "claim_allowed": False,
            },
            {
                "gap": "real_second_hazard_skill",
                "status": "OFFICIAL_EVENT_TARGETS_FROZEN_INPUT_FIELDS_NOT_BUNDLED",
                "implemented_readiness": "tested upper/lower-tail evaluator + official IMD heat/cold event registry",
                "claim_allowed": False,
            },
            {
                "gap": "same_event_india_stage2",
                "status": "BLOCKED_BY_MATCHED_COARSE_FINE_REFERENCE_DATA",
                "implemented_readiness": "software handoff + fail-closed integration readiness + frozen protocol",
                "claim_allowed": False,
            },
        ],
        "principle": "Data-access blockers are exposed explicitly and cannot be converted into positive skill claims by software readiness alone.",
    })


@app.get("/api/v1/weakness-mitigation")
def weakness_mitigation() -> dict[str, Any]:
    return enrich({
        "schema": "trace-weakness-mitigation/1.0",
        "purpose": "One-screen response to the five main national-screening risks. v1.5 preserves the v1.4 evidence upgrades and adds fail-closed governance, mandatory human review, offline controlled stream replay, and frozen official heat/cold event targets without fabricating missing skill.",
        "items": WEAKNESS_MITIGATION,
        "score_effect_boundary": "FANI temporal tracking is executed evidence. Provider/hazard utilities are tested software. Direct NEPS-G, real second-hazard, and same-event India Stage-2 skill still require new data runs.",
    })


@app.get("/api/v1/model-selection/defence")
def model_selection_defence() -> dict[str, Any]:
    return enrich(MODEL_SELECTION_DEFENCE)


@app.get("/api/v1/hazard-readiness")
def hazard_readiness() -> dict[str, Any]:
    return enrich(HAZARD_READINESS)


@app.get("/api/v1/data-adapters")
def data_adapters() -> dict[str, Any]:
    return enrich(DATA_ADAPTERS)


@app.get("/api/v1/integration/audit")
def integration_audit() -> dict[str, Any]:
    return enrich(INTEGRATION_AUDIT)


@app.get("/api/v1/alerts")
def alerts() -> dict[str, Any]:
    out = copy.deepcopy(DATA)
    out["alerts"] = [enrich_alert(a) for a in out["alerts"]]
    out.update(metadata())
    return out


@app.get("/api/v1/alerts/{lead_hours}")
def alert(lead_hours: int) -> dict[str, Any]:
    if lead_hours not in ALERTS:
        raise HTTPException(404, "Available leads: 72, 96, 120")
    return enrich_alert(ALERTS[lead_hours])


@app.get("/api/v1/geojson")
def geojson() -> dict[str, Any]:
    features = []
    for a in DATA["alerts"]:
        base = {"event_id": a["event_id"], "lead_hours": a["forecast_lead_hours"], "severity": a["severity"]}
        features.append({"type": "Feature", "properties": base | {"layer": "macro_threat_footprint"}, "geometry": a["macro_threat_footprint_geojson"]})
        features.append({"type": "Feature", "properties": base | {"layer": "5km_ui_locator_ring", "modeled_hazard_radius": False}, "geometry": a["display_marker_geojson"]})
        c = a["anomaly_core_coordinate"]
        features.append({"type": "Feature", "properties": base | {"layer": "anomaly_core", "meaning": c["meaning"]}, "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]}})
    return {"type": "FeatureCollection", "features": features, **metadata()}


@app.get("/api/v1/refinement-request/{lead_hours}")
def refinement_request(lead_hours: int) -> dict[str, Any]:
    if lead_hours not in ALERTS:
        raise HTTPException(404, "Available leads: 72, 96, 120")
    a = ALERTS[lead_hours]
    return enrich(
        {
            "event_id": a["event_id"],
            "valid_time": a["valid_time"],
            "lead_hours": lead_hours,
            "anomaly_core_coordinate": a["anomaly_core_coordinate"],
            **a["refinement_request"],
        }
    )


@app.get("/api/v1/impact-object/{lead_hours}")
def impact_object(lead_hours: int) -> dict[str, Any]:
    if lead_hours not in ALERTS:
        raise HTTPException(404, "Available leads: 72, 96, 120")
    a = ALERTS[lead_hours]
    return enrich(
        {
            "schema": "trace-impact-object/1.5",
            "event_id": a["event_id"],
            "valid_time": a["valid_time"],
            "forecast_lead_hours": lead_hours,
            "prototype_severity": a["severity"],
            "macro_threat_footprint_geojson": a["macro_threat_footprint_geojson"],
            "macro_footprint_area_km2": a["macro_footprint_area_km2"],
            "anomaly_core_coordinate": a["anomaly_core_coordinate"],
            "display_marker_radius_km": a["display_marker_radius_km"],
            "predicted_impact_radius_km": None,
            "display_marker_meaning": "UI locator only; not modeled hazard extent and not Stage-2 output",
            "model_evidence": a["model_evidence"],
            "stage2_route": STAGE2,
            "stage2_transition_label": "software handoff only; same-event BULBUL 12->5 km skill is not claimed",
            "provenance": a["provenance"],
        }
    )


@app.post("/api/v1/alerts/evaluate")
def evaluate_alert(req: AlertRequest) -> dict[str, Any]:
    s = classify(req.evidence)
    return enrich(
        {
            "schema": "trace-alert-evaluation/1.5",
            "event_id": req.event_id,
            "valid_time": req.valid_time.astimezone(timezone.utc).isoformat(),
            "forecast_lead_hours": req.forecast_lead_hours,
            "severity": s.severity,
            "severity_rule": s.rule,
            "rule_provenance": s.rule_provenance,
            "severity_thresholds": s.thresholds,
            "anomaly_core_coordinate": {
                "lat": float(req.core_lat),
                "lon": float(req.core_lon),
                "meaning": "caller-supplied/model anomaly core; NOT an exact cyclone-center claim",
            },
            "display_marker_radius_km": 5.0,
            "predicted_impact_radius_km": None,
            "display_marker_meaning": "UI locator only; not modeled hazard extent and not Stage-2 output",
            "macro_footprint_geojson": req.macro_footprint_geojson,
            "macro_footprint_area_km2": float(req.macro_footprint_area_km2),
            "evidence": req.evidence.model_dump(),
            "explanation": s.explanation,
            "operational_status": "RESEARCH_PROTOTYPE_NOT_OFFICIAL_WARNING",
        }
    )


@app.get("/api/v1/refinement/benchmark")
def refinement_benchmark() -> dict[str, Any]:
    return enrich(STAGE2)


@app.post("/api/v1/severity", response_model=SeverityResponse)
def severity(e: EvidenceInput) -> SeverityResponse:
    return classify(e)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(HERE / "FINAL_DEMO.html")
