from pathlib import Path
import subprocess,sys,json,hashlib
HERE=Path(__file__).resolve().parent
def run(name,cmd,cwd):
 print(f'=== {name} ===',flush=True); r=subprocess.run(cmd,cwd=cwd); return {'name':name,'returncode':r.returncode}
results=[]
results.append(run('Stage-1 frozen BULBUL GNN inference',[sys.executable,'reproduce_bulbul_gnn.py'],HERE/'stage1'))
results.append(run('Stage-1 location-feature occlusion sensitivity',[sys.executable,'run_location_occlusion.py'],HERE/'stage1'))
results.append(run('Stage-2 v2 full train/eval replay',[sys.executable,'run_portable_stage2_v2.py'],HERE/'stage2_v2'))
status='PASS' if all(x['returncode']==0 for x in results) else 'FAIL'
out={'status':status,'runs':results,'scope':'Replays the frozen BULBUL checkpoint headline, an inference occlusion check, and the complete original HRRR Stage-2 v2 train/eval path. The separately executed matched seed-1 retrained no-message location ablation is stored under stage1 and has its own runner. Cross-event v3 diagnostics are archived separately; no claim that this wrapper retrains the later four-fold v3 analysis.'}
(HERE/'SCIENCE_REPRODUCTION_REPORT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));raise SystemExit(0 if status=='PASS' else 1)
