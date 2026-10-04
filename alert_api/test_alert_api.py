from fastapi.testclient import TestClient
from trace_alert_api import app
from provider_ingest import inspect_local_file
from hazard_footprint import evaluate_percentile_footprint
from governance import evaluate_operational_safety
import numpy as np
import tempfile
from pathlib import Path

c = TestClient(app)

FOOTPRINT = {
    "type": "Polygon",
    "coordinates": [[
        [87.8, 17.8], [88.2, 17.8], [88.2, 18.2], [87.8, 18.2], [87.8, 17.8]
    ]],
}
VALID_EVAL = {
    "event_id": "DEMO_EVENT",
    "valid_time": "2026-01-01T00:00:00Z",
    "forecast_lead_hours": 72,
    "core_lat": 18.0,
    "core_lon": 88.0,
    "macro_footprint_area_km2": 20000,
    "macro_footprint_geojson": FOOTPRINT,
    "evidence": {
        "max_ensemble_mean_wind_ms": 18.0,
        "max_efi": 0.85,
        "max_member_support_fraction": 0.4,
        "mean_seed_agreement": 0.9,
        "max_sot": 2.0,
    },
}

# Service health/readiness and evidence integrity.
assert c.get('/health').status_code == 200
h = c.get('/health').json()
assert h['alerts_loaded'] == 3
assert h['api_version'] == '1.8.0-prototype'
assert h['prototype_alert'] is True and h['official_warning'] is False and h['calibrated_probability'] is False
assert h['frozen_evidence_integrity'] == 'PASS'

ready = c.get('/ready')
assert ready.status_code == 200
assert ready.json()['integrity']['all_frozen_artifacts_verified'] is True
prov = c.get('/api/v1/provenance')
assert prov.status_code == 200
pj = prov.json()
assert pj['status'] == 'PASS'
assert len(pj['checks']) == 12
assert all(x['match'] for x in pj['checks'])


# National-screening risk hardening endpoints, upgraded through v1.5.
wm = c.get('/api/v1/weakness-mitigation')
assert wm.status_code == 200
wmj = wm.json()
assert len(wmj['items']) == 5
assert wmj['api_version'] == '1.8.0-prototype'
assert any('NEPS-G' in x['weakness'] for x in wmj['items'])
assert 'without fabricating missing skill' in wmj['purpose']

ms = c.get('/api/v1/model-selection/defence')
assert ms.status_code == 200
msj = ms.json()
assert msj['bulbul_iou_evidence']['wind_spherical_gnn_iou'] == 0.430
assert 'does not claim graph message passing is independently proven necessary' in msj['core_position']

hr = c.get('/api/v1/hazard-readiness')
assert hr.status_code == 200
hrj = hr.json()
assert hrj['validated_now'][0]['hazard'] == 'severe_cyclone_or_extreme_wind'
assert any(x['hazard'] == 'heat_dome' and x['validation_status'] == 'NOT_EXECUTED_IN_BUNDLE' for x in hrj['software_ready_not_validated'])

ad = c.get('/api/v1/data-adapters')
assert ad.status_code == 200
adj = ad.json()
assert adj['implemented_prototype_provider']['status'] == 'IMPLEMENTED_AND_MEASURED'
assert adj['operational_target_provider']['status'].startswith('TESTED_INGEST_CONTRACT')

ia = c.get('/api/v1/integration/audit')
assert ia.status_code == 200
iaj = ia.json()
assert iaj['stage1_to_stage2_contract'] == 'READY_SOFTWARE_HANDOFF'
assert iaj['same_event_india_validation'] is False

# v1.4: recovered executed sequential FANI wind-footprint tracking evidence.
te = c.get('/api/v1/tracking/evidence')
assert te.status_code == 200
tej = te.json()
assert tej['status'] == 'EXECUTED_REAL_TEMPORAL_EVALUATION'
assert tej['time_steps'] == 9 and tej['lead_hours'][0] == 72 and tej['lead_hours'][-1] == 120
assert tej['scientific_evidence']['sequential_valid_time_wind_footprint_tracking_executed'] is True
assert tej['scientific_evidence']['cyclone_center_reconstruction_validated'] is False
assert abs(tej['method_summaries']['member_1']['mean_centroid_error_km'] - 329.1305813265569) < 1e-9

# v1.4: provider-ingest schema validation is tested but explicitly not a skill claim.
provider_meta = {
    'provider_name': 'NCMRWF NEPS-G candidate',
    'dimensions': {'member': 23, 'lat': 31, 'lon': 31},
    'coordinates': {'lat': {}, 'lon': {}, 'valid_time': {}},
    'variables': {'t2m': {'units': 'K'}},
    'initialization_time': '2020-01-01T00:00:00Z',
    'lead_hours': [72, 96, 120, 240],
}
piv = c.post('/api/v1/provider-ingest/validate', json=provider_meta)
assert piv.status_code == 200
pivj = piv.json()
assert pivj['status'] == 'PASS_SCHEMA_COMPATIBLE'
assert pivj['trace_adapter_compatible'] is True
assert pivj['direct_nepsg_skill_claim_supported'] is False
assert pivj['scientific_skill_validation'] is False
missing_member = provider_meta | {'dimensions': {'lat': 31, 'lon': 31}}
assert c.post('/api/v1/provider-ingest/validate', json=missing_member).json()['status'] == 'FAIL_SCHEMA_INCOMPLETE'
bad_provider_time = provider_meta | {'initialization_time': 'not-a-time'}
assert c.post('/api/v1/provider-ingest/validate', json=bad_provider_time).status_code == 422

# v1.4: cross-event architecture gate is derived only from frozen aggregates.
mg = c.get('/api/v1/model-selection/gate')
assert mg.status_code == 200
mgj = mg.json()
assert mgj['decision'] == 'RETAIN_MESSAGE_PASSING_AS_EXPERIMENTAL_BRANCH_NOT_PROVEN_DEFAULT'
assert mgj['frozen_event_evidence']['FANI_2019_all_lead_climate99_iou']['gnn_minus_no_message'] < 0
assert mgj['frozen_event_evidence']['BULBUL_2019_wind_climate99_iou']['gnn_minus_no_message'] > 0
assert mgj['scientific_significance_test'] is False

sb = c.get('/api/v1/screening-brief')
assert sb.status_code == 200 and len(sb.json()['three_measured_proofs']) == 3
ir = c.get('/api/v1/integration/readiness')
assert ir.status_code == 200
irj = ir.json()
assert irj['research_route_allowed'] is True and irj['operational_route_allowed'] is False
assert irj['same_event_india_meteorological_validation'] is False

# Hazard-generic evaluator proves software extension only, for both upper and lower tails.
ens = np.array([[[31, 33], [28, 30]], [[32, 34], [29, 31]], [[30, 35], [27, 32]]], dtype=float)
ref = np.array([[32, 34], [28, 31]], dtype=float)
clim = np.array([[31, 33], [30, 30]], dtype=float)
upper = evaluate_percentile_footprint(ens, ref, clim, polarity='upper', support_threshold=0.4)
assert upper['metrics']['iou'] == 1.0 and upper['scientific_skill_validation'] is False
lower = evaluate_percentile_footprint(-ens, -ref, -clim, polarity='lower', support_threshold=0.4)
assert lower['metrics']['iou'] == 1.0 and lower['scientific_skill_validation'] is False
try:
    evaluate_percentile_footprint(ens[:1], ref, clim)
    raise AssertionError('single-member input should fail')
except ValueError:
    pass

# Local NetCDF provider preflight is executable, not just documentation.
try:
    import xarray as xr
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'synthetic_neps_like.nc'
        ds = xr.Dataset(
            {'t2m': (('member', 'lead_hours', 'lat', 'lon'), np.zeros((3, 2, 2, 2), dtype=np.float32))},
            coords={
                'member': [0, 1, 2],
                'lead_hours': [72, 120],
                'lat': [20.0, 21.0],
                'lon': [78.0, 79.0],
                'forecast_reference_time': np.datetime64('2020-01-01T00:00:00'),
            },
            attrs={'provider': 'synthetic NEPS-G-like test fixture'},
        )
        ds['t2m'].attrs['units'] = 'K'
        ds.to_netcdf(path, engine='scipy')
        inspected = inspect_local_file(path)
        assert inspected['status'] == 'PASS_SCHEMA_COMPATIBLE', inspected
        assert inspected['validation']['direct_nepsg_skill_claim_supported'] is False
except ImportError:
    raise AssertionError('xarray is required by TRACE v1.4 provider preflight tests')


# v1.5: fail-closed governance and controlled stream replay.
gov = c.get('/api/v1/governance')
assert gov.status_code == 200
govj = gov.json()
assert govj['human_review_required'] is True
assert govj['official_warning_release_allowed'] is False
assert govj['fail_closed_on_provenance_failure'] is True
assert govj['policy']['machine_route'] == 'INTERNAL_HUMAN_REVIEW_QUEUE'

osafe = c.get('/api/v1/operational-safety')
assert osafe.status_code == 200
osj = osafe.json()
assert osj['human_review_required'] is True
assert osj['public_warning_release_allowed'] is False
assert osj['official_warning'] is False
fail_closed = evaluate_operational_safety(provenance_ok=False, research_route_ready=True)
assert fail_closed['machine_route'] == 'BLOCKED_FAIL_CLOSED'
assert fail_closed['public_warning_release_allowed'] is False

replay_payload = {'records': [
    {'event_id': 'BULBUL_2019', 'forecast_lead_hours': 72, 'severity': 'SEVERE'},
    {'event_id': 'BULBUL_2019', 'forecast_lead_hours': 96, 'severity': 'MODERATE'},
]}
replay = c.post('/api/v1/stream/replay', json=replay_payload)
assert replay.status_code == 200
rj = replay.json()
assert rj['mode'] == 'OFFLINE_CONTROLLED_REPLAY_NOT_LIVE_STREAMING'
assert rj['record_count'] == 2
assert rj['operational_streaming_validated'] is False
assert all(x['human_review_required'] is True and x['official_warning'] is False for x in rj['records'])
assert c.post('/api/v1/stream/replay', json={'records': []}).status_code == 422

targets = c.get('/api/v1/hazard-event-targets')
assert targets.status_code == 200
tj = targets.json()
assert tj['multi_hazard_skill_validated'] is False
assert len(tj['events']) == 2
assert {x['hazard'] for x in tj['events']} == {'heat_wave_severe_heat_wave', 'cold_wave_severe_cold_wave_cold_day'}
assert all('FORECAST_SKILL_NOT_EXECUTED' in x['trace_status'] for x in tj['events'])

blockers = c.get('/api/v1/data-blockers')
assert blockers.status_code == 200
bj = blockers.json()
assert len(bj['blockers']) == 3
assert all(x['claim_allowed'] is False for x in bj['blockers'])

# Frozen alerts and outputs.
x = c.get('/api/v1/alerts').json()
assert len(x['alerts']) == 3
assert x['evidence_version'].startswith('bulbul2019')

for lead in [72, 96, 120]:
    r = c.get(f'/api/v1/alerts/{lead}')
    assert r.status_code == 200
    j = r.json()
    assert j['severity'] in {'LOW', 'MODERATE', 'SEVERE'}
    assert j['display_marker_radius_km'] == 5.0
    assert j['predicted_impact_radius_km'] is None
    assert j['source_boundary']['five_km_marker_is_ui_locator_not_hazard_radius'] is True
    assert j['source_boundary']['calibrated_probability_claim'] is False
    assert j['prototype_alert'] is True and j['official_warning'] is False
    assert 'severity_thresholds' in j and 'rule_provenance' in j
    rr = c.get(f'/api/v1/refinement-request/{lead}')
    assert rr.status_code == 200
    io = c.get(f'/api/v1/impact-object/{lead}')
    assert io.status_code == 200
    assert io.json()['stage2_transition_label'].startswith('software handoff only')

geo = c.get('/api/v1/geojson').json()
assert geo['type'] == 'FeatureCollection' and len(geo['features']) == 9
assert geo['api_version'] == '1.8.0-prototype'
assert all(f['properties'].get('layer') != '5km_display_impact_ring' for f in geo['features'])
assert sum(f['properties'].get('layer') == '5km_ui_locator_ring' for f in geo['features']) == 3

# Lead evolution must be described honestly: same valid time, not a trajectory.
le = c.get('/api/v1/lead-evolution')
assert le.status_code == 200
lej = le.json()
assert len(lej['forecast_snapshots']) == 3
assert len(lej['verification_valid_times']) == 1
assert lej['scientific_trajectory_validation'] is False
assert 'NOT a physical storm trajectory' in lej['interpretation']

# Generic software track-object builder on explicitly time-varying caller/model frames.
track_req = {
    "event_id": "SOFTWARE_TRACK_DEMO",
    "frames": [
        {"valid_time": "2026-01-01T00:00:00Z", "forecast_lead_hours": 120, "core_lat": 16.0, "core_lon": 87.0, "macro_footprint_geojson": FOOTPRINT},
        {"valid_time": "2026-01-01T06:00:00Z", "forecast_lead_hours": 114, "core_lat": 16.5, "core_lon": 87.5, "macro_footprint_geojson": FOOTPRINT},
        {"valid_time": "2026-01-01T12:00:00Z", "forecast_lead_hours": 108, "core_lat": 17.0, "core_lon": 88.0, "macro_footprint_geojson": FOOTPRINT},
    ],
}
tr = c.post('/api/v1/tracks/build', json=track_req)
assert tr.status_code == 200
trj = tr.json()
assert trj['software_tracking_capability'] is True
assert trj['scientific_validation'] is False
assert trj['core_trajectory_geojson']['type'] == 'LineString'
assert len(trj['core_trajectory_geojson']['coordinates']) == 3
assert len(trj['frames']) == 3
assert all('temporal_bbox' in f for f in trj['frames'])

# Duplicate or unordered times fail safely.
bad_track = track_req | {'frames': [track_req['frames'][0], track_req['frames'][0]]}
assert c.post('/api/v1/tracks/build', json=bad_track).status_code == 422
bad_track = track_req | {'frames': [track_req['frames'][1], track_req['frames'][0]]}
assert c.post('/api/v1/tracks/build', json=bad_track).status_code == 422

sev = c.post('/api/v1/severity', json=VALID_EVAL['evidence'])
assert sev.status_code == 200
sev_j = sev.json()
assert sev_j['schema'] == 'trace-severity-response/1.5'
assert sev_j['severity'] == 'SEVERE'
assert sev_j['official_warning'] is False

assert c.get('/api/v1/refinement/benchmark').status_code == 200
assert c.get('/api/v1/openapi.json').status_code == 200

r = c.post('/api/v1/alerts/evaluate', json=VALID_EVAL)
assert r.status_code == 200
j = r.json()
assert j['severity'] == 'SEVERE'
assert j['valid_time'].endswith('+00:00')
assert j['macro_footprint_geojson']['type'] == 'Polygon'

# Input hardening regressions.
bad = VALID_EVAL | {'valid_time': 'not-a-time'}
assert c.post('/api/v1/alerts/evaluate', json=bad).status_code == 422
bad = VALID_EVAL | {'forecast_lead_hours': 1}
assert c.post('/api/v1/alerts/evaluate', json=bad).status_code == 422
bad = VALID_EVAL | {'evidence': VALID_EVAL['evidence'] | {'max_efi': 99}}
assert c.post('/api/v1/alerts/evaluate', json=bad).status_code == 422
bad = VALID_EVAL | {'evidence': VALID_EVAL['evidence'] | {'max_sot': 'Infinity'}}
assert c.post('/api/v1/alerts/evaluate', json=bad).status_code == 422
bad = VALID_EVAL | {'extra': 'should fail'}
assert c.post('/api/v1/alerts/evaluate', json=bad).status_code == 422
bad = VALID_EVAL | {'macro_footprint_geojson': {'type': 'Point', 'coordinates': [88, 18]}}
assert c.post('/api/v1/alerts/evaluate', json=bad).status_code == 422

assert c.get('/').status_code == 200
print('TRACE_ALERT_API_TEST_PASS_V1_8')
