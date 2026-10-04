from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = ROOT / "alert_api"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FROZEN_FILES = [
    "source_evidence/BULBUL_2019_72.npz",
    "source_evidence/BULBUL_2019_96.npz",
    "source_evidence/BULBUL_2019_120.npz",
    "source_evidence/gnn_seed1.pt",
    "source_evidence/gnn_seed2.pt",
    "source_evidence/gnn_seed3.pt",
    "source_evidence/gnn_seed4.pt",
    "source_evidence/gnn_seed5.pt",
    "data/alerts.json",
    "data/stage2_benchmark.json",
]
EXPECTED_ALERTS_SHA256 = "89a92f51c82ba823fb0449314582947e0510abcee8a01a797897642fc0552d73"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str]) -> dict:
    start = time.perf_counter()
    p = subprocess.run(cmd, cwd=API, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "seconds": round(time.perf_counter() - start, 4),
        "output": p.stdout,
    }


def dump(name: str, obj: object) -> str:
    rel = f"outputs/{name}"
    (OUT / name).write_text(json.dumps(obj, indent=2))
    return rel


def main() -> None:
    started = time.perf_counter()
    before_hashes = {rel: sha(API / rel) for rel in FROZEN_FILES}

    rebuild = run([sys.executable, "rebuild_alerts.py"])
    alerts_sha_after = sha(API / "data/alerts.json")
    tests = run([sys.executable, "test_alert_api.py"])

    if rebuild["returncode"] != 0 or tests["returncode"] != 0:
        fail_manifest = {
            "schema": "trace-final-demo-run-manifest/1.8",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "rebuild": rebuild,
            "tests": tests,
        }
        (OUT / "RUN_MANIFEST.json").write_text(json.dumps(fail_manifest, indent=2))
        raise SystemExit("TRACE demo failed; inspect outputs/RUN_MANIFEST.json")
    if alerts_sha_after != EXPECTED_ALERTS_SHA256:
        raise SystemExit("Rebuilt alerts.json hash changed; frozen evidence regression failed")

    sys.path.insert(0, str(API))
    os.chdir(API)
    from fastapi.testclient import TestClient
    import trace_alert_api
    from hazard_footprint import evaluate_percentile_footprint

    client = TestClient(trace_alert_api.app)
    impact_objects = {str(lead): client.get(f"/api/v1/impact-object/{lead}").json() for lead in (72, 96, 120)}
    geojson = client.get("/api/v1/geojson").json()
    stage2 = client.get("/api/v1/refinement/benchmark").json()
    status = client.get("/api/v1/status").json()
    openapi = client.get("/api/v1/openapi.json").json()
    provenance = client.get("/api/v1/provenance").json()
    readiness = client.get("/ready").json()
    lead_evolution = client.get("/api/v1/lead-evolution").json()
    weakness_mitigation = client.get("/api/v1/weakness-mitigation").json()
    model_selection_defence = client.get("/api/v1/model-selection/defence").json()
    hazard_readiness = client.get("/api/v1/hazard-readiness").json()
    data_adapters = client.get("/api/v1/data-adapters").json()
    integration_audit = client.get("/api/v1/integration/audit").json()
    tracking_evidence = client.get("/api/v1/tracking/evidence").json()
    model_selection_gate = client.get("/api/v1/model-selection/gate").json()
    screening_brief = client.get("/api/v1/screening-brief").json()
    integration_readiness = client.get("/api/v1/integration/readiness").json()
    governance = client.get("/api/v1/governance").json()
    operational_safety = client.get("/api/v1/operational-safety").json()
    hazard_event_targets = client.get("/api/v1/hazard-event-targets").json()
    data_blockers = client.get("/api/v1/data-blockers").json()
    stream_replay = client.post("/api/v1/stream/replay", json={"records": [
        {"event_id": "BULBUL_2019", "forecast_lead_hours": 72, "severity": "SEVERE"},
        {"event_id": "BULBUL_2019", "forecast_lead_hours": 96, "severity": "MODERATE"},
        {"event_id": "BULBUL_2019", "forecast_lead_hours": 120, "severity": "MODERATE"},
    ]}).json()
    provider_validation = client.post("/api/v1/provider-ingest/validate", json={
        "provider_name": "NCMRWF NEPS-G candidate metadata fixture",
        "dimensions": {"member": 23, "lat": 31, "lon": 31},
        "coordinates": {"lat": {}, "lon": {}, "valid_time": {}},
        "variables": {"t2m": {"units": "K"}, "u10": {"units": "m s-1"}, "v10": {"units": "m s-1"}},
        "initialization_time": "2020-01-01T00:00:00Z",
        "lead_hours": [72, 96, 120, 144, 168, 192, 216, 240],
    }).json()
    hazard_generic_smoke = evaluate_percentile_footprint(
        np.array([[[31, 33], [28, 30]], [[32, 34], [29, 31]], [[30, 35], [27, 32]]], dtype=float),
        np.array([[32, 34], [28, 31]], dtype=float),
        np.array([[31, 33], [30, 30]], dtype=float),
        polarity="upper",
        support_threshold=0.4,
    )

    # Software-only tracking demonstration. It intentionally does not reuse the frozen
    # BULBUL lead snapshots as a physical storm trajectory because those share one valid time.
    demo_poly = {
        "type": "Polygon",
        "coordinates": [[[87.5, 16.0], [88.0, 16.0], [88.0, 16.5], [87.5, 16.5], [87.5, 16.0]]],
    }
    track_request = {
        "event_id": "SOFTWARE_TRACK_DEMO",
        "frames": [
            {"valid_time": "2026-01-01T00:00:00Z", "forecast_lead_hours": 120, "core_lat": 16.2, "core_lon": 87.7, "macro_footprint_geojson": demo_poly},
            {"valid_time": "2026-01-01T06:00:00Z", "forecast_lead_hours": 114, "core_lat": 16.7, "core_lon": 88.1, "macro_footprint_geojson": demo_poly},
            {"valid_time": "2026-01-01T12:00:00Z", "forecast_lead_hours": 108, "core_lat": 17.1, "core_lon": 88.6, "macro_footprint_geojson": demo_poly},
        ],
    }
    tracking_demo_response = client.post("/api/v1/tracks/build", json=track_request)
    if tracking_demo_response.status_code != 200:
        raise SystemExit("Software tracking demo endpoint failed")
    tracking_demo = tracking_demo_response.json()

    generated = [
        dump("impact_objects.json", impact_objects),
        dump("geojson.json", geojson),
        dump("stage2_route_status.json", stage2),
        dump("api_status.json", status),
        dump("openapi_schema.json", openapi),
        dump("provenance_integrity.json", provenance),
        dump("readiness.json", readiness),
        dump("lead_evolution.json", lead_evolution),
        dump("tracking_software_demo.json", tracking_demo),
        dump("weakness_mitigation.json", weakness_mitigation),
        dump("model_selection_defence.json", model_selection_defence),
        dump("hazard_readiness.json", hazard_readiness),
        dump("data_adapters.json", data_adapters),
        dump("integration_audit.json", integration_audit),
        dump("fani_temporal_tracking_evidence.json", tracking_evidence),
        dump("model_selection_gate.json", model_selection_gate),
        dump("screening_brief.json", screening_brief),
        dump("integration_readiness.json", integration_readiness),
        dump("provider_ingest_validation.json", provider_validation),
        dump("hazard_generic_evaluator_smoke.json", hazard_generic_smoke),
        dump("governance_status.json", governance),
        dump("operational_safety.json", operational_safety),
        dump("controlled_stream_replay.json", stream_replay),
        dump("official_hazard_event_targets.json", hazard_event_targets),
        dump("data_access_blockers.json", data_blockers),
    ]

    test_report = {
        "schema": "trace-test-report/1.6",
        "status": "PASS",
        "api_version": trace_alert_api.API_VERSION,
        "test_command": "python test_alert_api.py",
        "test_returncode": tests["returncode"],
        "test_seconds": tests["seconds"],
        "test_output": tests["output"],
        "rebuild_command": "python rebuild_alerts.py",
        "rebuild_returncode": rebuild["returncode"],
        "rebuild_seconds": rebuild["seconds"],
        "alerts_json_hash": alerts_sha_after,
        "frozen_provenance_integrity": provenance["status"],
        "validation_hardening_checked": [
            "invalid timestamp rejected",
            "invalid medium-range lead rejected",
            "out-of-range EFI rejected",
            "non-finite SOT rejected",
            "extra request fields rejected",
            "invalid GeoJSON rejected",
            "duplicate track times rejected",
            "unordered track times rejected",
            "frozen sample/checkpoint hashes verified",
            "readiness depends on integrity verification",
            "five national-screening weakness mitigations exposed",
            "GNN necessity defence refuses overclaim",
            "multi-hazard readiness separated from multi-hazard validation",
            "NEPS-G adapter contract separated from NEPS-G evidence",
            "Stage1->Stage2 software handoff separated from same-event meteorological skill",
            "executed FANI sequential temporal evidence recovered and hash-verified",
            "NEPS-G-like provider metadata contract accepted while direct skill claim stays false",
            "missing ensemble dimension fails provider-ingest compatibility",
            "hazard-generic upper/lower-tail evaluator unit-tested without multi-hazard skill claim",
            "frozen FANI/BULBUL cross-event architecture gate preserves inconclusive message-passing result",
            "integration readiness refuses operational/same-event claim",
            "provenance failure path blocks output fail-closed",
            "mandatory human review enforced for prototype alert routing",
            "prototype cannot self-authorize official/public warning release",
            "controlled replay is explicitly offline and not live-streaming validation",
            "official IMD heat/cold event targets are frozen without converting them into forecast skill claims",
            "machine-readable data blockers keep access/readiness separate from measured skill",
        ],
    }
    generated.append(dump("TEST_REPORT.json", test_report))

    manifest = {
        "schema": "trace-final-demo-run-manifest/1.8",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python run_trace_demo.py",
        "local_runtime_seconds": round(time.perf_counter() - started, 4),
        "python": sys.version,
        "platform": platform.platform(),
        "status": "PASS",
        "api_version": trace_alert_api.API_VERSION,
        "frozen_evidence_verified": provenance["all_frozen_artifacts_verified"],
        "alerts_json_rebuilt_byte_identical_hash": alerts_sha_after,
        "rebuild": rebuild,
        "tests": tests,
        "input_hashes": before_hashes,
        "generated_output_hashes": {rel: sha(ROOT / rel) for rel in generated},
        "software_capabilities_added": [
            "content-hash provenance verification",
            "readiness endpoint gated by frozen evidence integrity",
            "lead-evolution view explicitly distinguished from physical trajectory",
            "generic time-ordered track-object builder with 4D frame boxes",
            "weakness-mitigation API for national-screening risks",
            "provider-adapter readiness map for NEPS-G transition",
            "hazard-readiness matrix for validated vs unvalidated hazards",
            "model-selection defence endpoint that avoids GNN overclaiming",
            "Stage1->Stage2 integration audit endpoint",
            "hash-verified recovered FANI sequential wind-footprint tracking evidence",
            "tested provider-ingest validator and local file preflight for NetCDF/NPZ",
            "tested hazard-generic upper/lower-tail footprint evaluator",
            "frozen NEPS-G, heat-wave, and same-event Stage-2 validation protocols",
            "cross-event architecture evidence gate derived from frozen FANI/BULBUL aggregates",
            "30-second screening brief endpoint",
            "fail-closed operational-safety governance with mandatory human review",
            "offline controlled stream replay for saved alert records",
            "official 2019 India heat/cold event-target registry for future held-out validation",
            "machine-readable external-data blocker status",
        ],
        "evidence_boundaries": [
            "GEFSv12 is a prototype substitute; direct NEPS-G validation is not included.",
            "Stage-2 HRRR evidence is controlled reconstruction, not same-event BULBUL downscaling skill.",
            "Severity labels are prototype categories, not official warnings or calibrated probabilities.",
            "The anomaly core is a model evidence coordinate, not an exact cyclone-center estimate.",
            "5 km means output/display grid spacing, not guaranteed 5 km forecast accuracy.",
            "The bundled +72/+96/+120 BULBUL snapshots share one valid time, so they are lead-sensitivity evidence rather than a physical trajectory.",
            "The /api/v1/tracks/build endpoint is a generic builder. Separately, recovered FANI evidence contains executed sequential wind-footprint tracking, but large errors remain and cyclone-center/landfall skill is not validated.",
            "The v1.8 stream replay is offline software-operability evidence, not continuous operational NWP streaming.",
            "Official IMD heat/cold event descriptions freeze future test targets; they are not TRACE heat/cold skill results.",
            "Every prototype alert requires human review and cannot self-authorize an official/public warning.",
        ],
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("TRACE_FINAL_DEMO_PASS_V1_8")
    print(json.dumps({"runtime_seconds": manifest["local_runtime_seconds"], "alerts_sha256": alerts_sha_after, "provenance": provenance["status"]}, indent=2))


if __name__ == "__main__":
    main()
