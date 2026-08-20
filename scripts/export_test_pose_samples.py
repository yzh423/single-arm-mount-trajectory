"""Export one held-out episode per task with 20 evenly spaced pose frames."""
from pathlib import Path
import json, numpy as np, sys
ROOT=Path(__file__).resolve().parents[1]; base=ROOT/'data/processed/local_pose_benchmark'; manifest=json.loads((base/'manifest.json').read_text())
sys.path.insert(0,str(ROOT));from design_optimization.local_pose_sampling import relative_pose_sample
rows=[]
for e in sorted(manifest['episodes'],key=lambda row:row['episode_id']):
    if e['split']!='test' or not e.get('trajectory_edge_trim_eligible',True):continue
    d=np.load(base/e['artifact'])
    try:p,rel=relative_pose_sample(d,20)
    except ValueError:continue
    rows.append({'task':e['task'],'hand':e['hand'],'episode_id':e['episode_id'],'edge_trim_seconds':0.15,'relative_position_m':p.tolist(),'relative_quaternion_wxyz':rel.tolist()})
out=base/'test_samples.json'; out.write_text(json.dumps(rows,indent=2),encoding='utf-8'); print({'tasks':len(rows),'frames':sum(len(r['relative_position_m']) for r in rows)})
