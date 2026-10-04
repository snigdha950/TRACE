# TRACE Severity Rule V1 - Prototype, Not Official Warning Policy

This rule converts frozen model evidence into LOW / MODERATE / SEVERE **prototype alert tiers** for the dashboard and API. It is not an IMD/NCMRWF warning rule and it is not a calibrated probability.

## Inputs
- Maximum ensemble-mean wind speed inside the candidate footprint.
- Maximum EFI inside the candidate footprint.
- Maximum ensemble-member support fraction.
- Mean five-seed model agreement across the footprint.
- Maximum SOT evidence.

## Moderate tier
All must hold:
- wind >= 12.5 m/s,
- EFI >= 0.5,
- mean seed agreement >= 0.6,
- SOT > 0.

## Severe tier
All must hold:
- wind >= 17.5 m/s,
- EFI >= 0.8,
- ensemble-member support >= 0.4,
- mean seed agreement >= 0.8.

## Provenance note
The 0.6 seed-consensus threshold is the frozen footprint-fusion gate. The 0.8 severe-tier seed agreement is a documented alert-layer category threshold used to require stronger consensus for the most serious prototype label. It is not a calibrated risk probability.

## Safety flags
Every API response includes:
- `prototype_alert=true`
- `official_warning=false`
- `calibrated_probability=false`
