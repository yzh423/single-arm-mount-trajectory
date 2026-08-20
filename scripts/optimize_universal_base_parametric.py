"""Find one upright XYZ base placement for all selected DROID tasks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_parametric_topologies import templates
from scripts.optimize_parametric_6r_droid import MANIFEST, TRACE, seed_bank


def rot(q):
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def tasks(samples: int, trace_path=TRACE, manifest_path=MANIFEST):
    tr = np.load(trace_path)
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    r0 = rot(tr["tcp_quat_wxyz"][0])
    canonical = np.diag([1., -1., -1.])  # tool Z points down, X points forward
    align = r0.T @ canonical
    out = []
    for s in meta["segments"]:
        ids = np.unique(np.linspace(s["start_frame"], s["end_frame"], samples).astype(int))
        out.append({
            "instruction": s["instruction"],
            "p": tr["tcp_xyz"][ids],
            "R": np.stack([rot(q) @ align for q in tr["tcp_quat_wxyz"][ids]]),
        })
    return out


def attempt(model, task, base_xyz, seed):
    q = seed.copy(); pe=[]; re=[]; sg=[]; travel=0.
    for index, (p, R) in enumerate(zip(task["p"], task["R"])):
        target=np.eye(4);target[:3,3]=p-base_xyz;target[:3,:3]=R
        qn,e,s=model.solve(target,q,iters=70 if index==0 else 42,damping=.018)
        travel += float(np.linalg.norm(qn-q));q=qn
        pe.append(np.linalg.norm(e[:3]));re.append(np.linalg.norm(e[3:]));sg.append(s)
    pe=np.asarray(pe);re=np.asarray(re)
    return dict(success_fraction=float(np.mean((pe<.015)&(re<np.radians(5)))),position_rmse_mm=float(1000*np.sqrt(np.mean(pe*pe))),orientation_rmse_deg=float(np.degrees(np.sqrt(np.mean(re*re)))),sigma_min=float(np.min(sg)),joint_travel_rad=travel)


def evaluate(model, taskset, base_xyz, seeds):
    rows=[]
    for task in taskset:
        tries=[attempt(model,task,base_xyz,q) for q in seeds]
        rows.append(min(tries,key=lambda x:(-x['success_fraction'],x['position_rmse_mm']+2*x['orientation_rmse_deg'],-x['sigma_min'])))
    success=np.mean([x['success_fraction'] for x in rows]);pos=np.mean([x['position_rmse_mm'] for x in rows]);ori=np.mean([x['orientation_rmse_deg'] for x in rows]);sigma=np.percentile([x['sigma_min'] for x in rows],10);travel=np.mean([x['joint_travel_rad'] for x in rows])
    score=500*(1-success)+pos+2*ori+30*max(0,.025-sigma)+.08*travel
    return float(score),dict(path_success_fraction=float(success),mean_position_rmse_mm=float(pos),mean_orientation_rmse_deg=float(ori),sigma_min_p10=float(sigma),mean_joint_travel_rad=float(travel),tasks=rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('topology',choices=('doosan','xarm6','ur5','kinova'));ap.add_argument('--trace',type=Path,default=TRACE);ap.add_argument('--manifest',type=Path,default=MANIFEST);ap.add_argument('--samples-per-task',type=int,default=7);ap.add_argument('--branches',type=int,default=10);ap.add_argument('--base-candidates',type=int,default=24);ap.add_argument('--include-refined-centers',action='store_true');ap.add_argument('--geometry-source',default='refined',help='geometry JSON suffix, e.g. refined or local_dense');ap.add_argument('--output-suffix',default='');a=ap.parse_args()
    trace_path=a.trace if a.trace.is_absolute() else ROOT/a.trace;manifest_path=a.manifest if a.manifest.is_absolute() else ROOT/a.manifest
    src=ROOT/'offline_results'/f'geometry_700_750_{a.topology}_{a.geometry_source}.json';saved=json.loads(src.read_text())['best'];model=templates()[a.topology].design(np.asarray(saved['scales']),float(saved['reach_m']));ts=tasks(a.samples_per_task,trace_path,manifest_path);allp=np.vstack([x['p'] for x in ts]);rng=np.random.default_rng(260806)
    # Broad upright-mount search in the DROID robot-base frame.  Pre-rank by
    # spherical reach so expensive full-pose IK is spent only on plausible bases.
    raw=np.column_stack((rng.uniform(-.35,.35,3000),rng.uniform(-.35,.35,3000),rng.uniform(-.35,.12,3000)))
    def geometric(b):
        r=np.linalg.norm(allp-b,axis=1);return 20*np.mean(np.maximum(r-(model.tool_length+saved['reach_m']),0)**2)+np.mean(np.maximum(.14-r,0)**2)
    raw=sorted(raw,key=geometric)[:a.base_candidates]
    if a.include_refined_centers:
        previous=ROOT/'offline_results'/f'geometry_700_750_{a.topology}_universal_base.json'
        anchors=[np.asarray(json.loads(previous.read_text())['best']['base_xyz_droid_m'])]
        per_task=ROOT/'offline_results'/f'geometry_700_750_{a.topology}_per_task_self_collision_base.json'
        rows={int(x['index']):x for x in json.loads(per_task.read_text())['results']}
        for refined in (ROOT/'offline_results').glob(f'geometry_700_750_{a.topology}_per_task_self_collision_base_task*_refined.json'):
            for row in json.loads(refined.read_text())['results']:rows[int(row['index'])]=row
        placements=np.asarray([rows[i]['best']['base_j1_xyz_droid_m'] for i in sorted(rows)])
        anchors.extend((np.mean(placements,axis=0),np.median(placements,axis=0)))
        local=[anchors[1]+rng.uniform([-.18,-.18,-.10],[.18,.18,.10]) for _ in range(max(0,a.base_candidates-len(anchors)))]
        raw=anchors+local
    seeds=seed_bank(model,a.branches,np.random.default_rng(260804+991));rows=[];tic=time.perf_counter()
    for i,b in enumerate(raw):
        score,metrics=evaluate(model,ts,b,seeds);rows.append(dict(base_xyz_droid_m=b.tolist(),score=score,metrics=metrics));print(a.topology,i+1,'/',len(raw),score,flush=True)
    rows.sort(key=lambda x:x['score']);out=dict(topology=a.topology,geometry_source=str(src.relative_to(ROOT)),trace=str(trace_path.relative_to(ROOT)),mapping='upright axes; first orientation aligned to common tool-Z-down frame',samples_per_task=a.samples_per_task,branches=a.branches,compute_s=time.perf_counter()-tic,best=rows[0],top10=rows[:10])
    out['base_coordinate_definition']='DROID-frame position of robot J1 axis origin (upright orientation)'
    dst=ROOT/'offline_results'/f'geometry_700_750_{a.topology}_universal_base{a.output_suffix}.json';dst.write_text(json.dumps(out,indent=2));print(dst)
if __name__=='__main__':main()
