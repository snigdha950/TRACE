from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'PROTOTYPE_MANIFEST.json').read_text())
problems=[]
for rel,meta in manifest['files'].items():
    p=ROOT/rel
    if not p.is_file():
        problems.append(f'missing: {rel}')
        continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=meta['sha256']:
        problems.append(f'hash mismatch: {rel}')
if problems:
    print('TRACE_PROTOTYPE_INTEGRITY_FAIL')
    print('\n'.join(problems))
    raise SystemExit(1)
print(f"TRACE_PROTOTYPE_INTEGRITY_PASS ({len(manifest['files'])} files)")
