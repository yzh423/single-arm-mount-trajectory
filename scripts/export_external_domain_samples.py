"""Create traceable validation/test pose samples for DROID and EgoDex separately."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];outdir=ROOT/'data/processed/external_domains';outdir.mkdir(parents=True,exist_ok=True)
def relative(p,q):
 w,x,y,z=q[0]/np.linalg.norm(q[0]);R=np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),(2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))));p=(p-p[0])@R;inv=q[0]*np.array((1,-1,-1,-1));w1,x1,y1,z1=inv;w2,x2,y2,z2=np.moveaxis(q,-1,0);r=np.stack((w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2),-1);return p,r
def evenly(lo,hi,n=8):return np.linspace(lo,max(lo,hi-1),n).round().astype(int)
droid={'validation':[],'test':[]};seen=set()
for directory in ('selected_5min','selected_20min'):
 base=ROOT/'data/DROID'/directory;z=np.load(next(base.glob('*.npz')),allow_pickle=False);m=json.loads(next(base.glob('*manifest.json')).read_text())
 for number,s in enumerate(m['segments']):
  identity=s.get('uuid','')+'|'+s.get('instruction','');
  if identity in seen:continue
  seen.add(identity);lo,hi=int(s['start_frame']),int(s['end_frame'])+1;cut1=lo+int(.6*(hi-lo));cut2=lo+int(.8*(hi-lo));
  for split,a,b in (('validation',cut1,cut2),('test',cut2,hi)):
   idx=evenly(a,b);p,q=relative(z['tcp_xyz'][idx],z['tcp_quat_wxyz'][idx]);droid[split].append({'task':f'droid_{len(seen):02d}','instruction':s.get('instruction',''),'source':f'{directory}:{number}','indices':idx.tolist(),'hand':'right','relative_position_m':p.tolist(),'relative_quaternion_wxyz':q.tolist()})
ego_path=ROOT/'data/EgoDex/pose_only_test/egodex_pose_only_test.npz';z=np.load(ego_path,mmap_mode='r',allow_pickle=False);offsets=z['episode_offsets'];candidates={'validation':{},'test':{}}
for episode,(task,source) in enumerate(zip(z['episode_task'],z['episode_source'])):
 value=int.from_bytes(hashlib.sha256(str(source).encode()).digest()[:8],'little')/2**64;split='validation' if .70<=value<.85 else ('test' if value>=.85 else 'train')
 if split in candidates:candidates[split].setdefault(str(task),episode)
ego={'validation':[],'test':[]}
for split,by_task in candidates.items():
 for task,episode in sorted(by_task.items()):
  lo,hi=int(offsets[episode]),int(offsets[episode+1]);left=int((z['left_confidence'][lo:hi]>=.8).sum());right=int((z['right_confidence'][lo:hi]>=.8).sum());hand='right' if right>=left else 'left';confidence=z[f'{hand}_confidence'][lo:hi];valid=np.flatnonzero(confidence>=.8)+lo
  if len(valid)<2:continue
  idx=valid[np.linspace(0,len(valid)-1,8).round().astype(int)];p=np.asarray(z[f'{hand}_relative_xyz'][idx]);q=np.asarray(z[f'{hand}_relative_quat_wxyz'][idx]);ego[split].append({'task':task,'episode':episode,'source':str(z['episode_source'][episode]),'indices':idx.tolist(),'hand':hand,'relative_position_m':p.tolist(),'relative_quaternion_wxyz':q.tolist()})
for name,payload in (('droid',droid),('egodex',ego)):
 path=outdir/f'{name}_samples.json';path.write_text(json.dumps({'domain':name,'policy':'temporal_blocked_per_segment' if name=='droid' else 'source_episode_hash','splits':payload},indent=2,ensure_ascii=False),encoding='utf-8');print(name,{k:(len(v),sum(len(x['indices']) for x in v)) for k,v in payload.items()})
