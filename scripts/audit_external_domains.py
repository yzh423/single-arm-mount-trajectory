"""Full-frame inventory and leakage-safe episode splits for DROID and EgoDex."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def bucket(text):
 v=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')/2**64;return 'train' if v<.70 else ('validation' if v<.85 else 'test')
droid_files=[];seen={}
for directory in ('selected_5min','selected_20min'):
 npz=next((ROOT/'data/DROID'/directory).glob('*.npz'));manifest=json.loads(next((ROOT/'data/DROID'/directory).glob('*manifest.json')).read_text());z=np.load(npz,allow_pickle=False);segments=[]
 for s in manifest['segments']:
  identity=s.get('uuid',s.get('instruction',''))+'|'+s.get('instruction','');duplicate=identity in seen;seen.setdefault(identity,f'{directory}:{s["start_frame"]}')
  segments.append({**s,'split':bucket(identity),'duplicate_across_droid_files':duplicate})
 droid_files.append({'path':str(npz.relative_to(ROOT)).replace('\\','/'),'frames':len(z['time_s']),'duration_s':float(z['time_s'][-1]-z['time_s'][0]),'segments':segments,'finite_pose_frames':int(np.isfinite(z['tcp_xyz']).all(1).sum())})
ego=ROOT/'data/EgoDex/pose_only_test/egodex_pose_only_test.npz';z=np.load(ego,mmap_mode='r',allow_pickle=False);offsets=z['episode_offsets'];episodes=[]
for i,(task,source) in enumerate(zip(z['episode_task'],z['episode_source'])):
 lo,hi=int(offsets[i]),int(offsets[i+1]);episodes.append({'episode':i,'task':str(task),'source':str(source),'frames':hi-lo,'split':bucket(str(source)),'left_confident_frames':int((z['left_confidence'][lo:hi]>=.8).sum()),'right_confident_frames':int((z['right_confidence'][lo:hi]>=.8).sum())})
payload={'schema_version':1,'domains':{'local':{'manifest':'data/processed/local_pose_benchmark/manifest.json','policy':'kept separate'},'droid':{'files':droid_files,'total_frames':sum(x['frames'] for x in droid_files),'unique_segment_identities':len(seen)},'egodex':{'path':str(ego.relative_to(ROOT)).replace('\\','/'),'frames':int(offsets[-1]),'episodes':len(episodes),'task_classes':len(set(str(x) for x in z['episode_task'])),'episode_records':episodes,'derived_challenge':'data/EgoDex/selected_challenge/egodex_challenge.npz'}}}
out=ROOT/'reports/single_arm/external_domain_audit.json';out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({'droid_frames':payload['domains']['droid']['total_frames'],'droid_unique_segments':len(seen),'egodex_frames':int(offsets[-1]),'egodex_episodes':len(episodes),'egodex_tasks':payload['domains']['egodex']['task_classes']}));print(out)
