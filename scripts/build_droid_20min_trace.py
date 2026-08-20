from __future__ import annotations
import datetime,json,urllib.parse,urllib.request
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation,Slerp
from build_droid_5min_trace import ROOT,FPS,_resample,_smooth_edges

DATA=ROOT/'data/DROID'; CACHE=DATA/'candidates_20min'; OUT=DATA/'selected_20min'
CATEGORIES={
 'articulated':['microwave','drawer','cupboard','door','coffee maker'],
 'wipe_fold':['wipe','towel','fold','unfold','cloth'],
 'rotation':['pour','stir','twist','rotate','turn the knob','flip'],
 'long_transfer':[' then ',' and then ','finally','lastly'],
}
TARGET_PER_CATEGORY=250.0; BRIDGE_S=1.2

def path_for(uid):
 lab,_,stamp=uid.split('+');dt=datetime.datetime.strptime(stamp,'%Y-%m-%d-%Hh-%Mm-%Ss')
 dirname=dt.strftime('%a_%b_%e_%H:%M:%S_%Y').replace('  ','_').replace(' ','_')
 return f'robotics/droid_raw/1.0.1/{lab}/success/{dt:%Y-%m-%d}/{dirname}/trajectory.h5'
def fetch(uid,ann):
 d=CACHE/uid;d.mkdir(parents=True,exist_ok=True);p=d/'trajectory.h5'
 if not p.exists():
  obj=urllib.parse.quote(path_for(uid),safe='');urllib.request.urlretrieve(f'https://storage.googleapis.com/download/storage/v1/b/gresearch/o/{obj}?alt=media',p)
 (d/'annotation.json').write_text(json.dumps({'uuid':uid,**ann[uid]},indent=2),encoding='utf-8');return p
def duration(p):
 with h5py.File(p,'r') as f:t=np.asarray(f['observation/timestamp/control/step_start'],float)
 return float(t[-1]-t[0])/1000
def main():
 CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 ann=json.load(open(DATA/'aggregated-annotations-030724.json',encoding='utf-8'));chosen=[];used=set()
 for cat,keys in CATEGORIES.items():
  rows=[]
  for uid,a in ann.items():
   s=a.get('language_instruction1','');sl=s.lower();hits=sum(k in sl for k in keys)
   if hits: rows.append((hits*100+len(s),uid,s))
  elapsed=0
  for _,uid,s in sorted(rows,reverse=True):
   if uid in used:continue
   try:p=fetch(uid,ann)
   except Exception:continue
   dur=duration(p)
   if dur<12:continue
   chosen.append((cat,uid,s,dur));used.add(uid);elapsed+=dur
   print(cat,f'{elapsed:.1f}s',uid,s)
   if elapsed>=TARGET_PER_CATEGORY:break
 poses=[];grips=[];segments=[];cursor=0
 for n,(cat,uid,s,dur) in enumerate(chosen):
  with h5py.File(CACHE/uid/'trajectory.h5','r') as f:
   t=np.asarray(f['observation/timestamp/control/step_start'],float);t=(t-t[0])/1000
   raw=np.asarray(f['observation/robot_state/cartesian_position'],float);g=np.asarray(f['observation/robot_state/gripper_position'],float)
  st,pose=_resample(t,raw);pose=_smooth_edges(pose);grip=np.interp(st,t,g)
  if poses:
   count=round(BRIDGE_S*FPS);u=np.linspace(0,1,count+2)[1:-1];u=u*u*(3-2*u);prev=poses[-1][-1]
   bp=prev[:3]+u[:,None]*(pose[0,:3]-prev[:3]);key=Rotation.from_quat(np.vstack((prev[3:][[1,2,3,0]],pose[0,3:][[1,2,3,0]])))
   bq=Slerp([0,1],key)(u).as_quat()[:,[3,0,1,2]];poses.append(np.c_[bp,bq]);grips.append(grips[-1][-1]+u*(grip[0]-grips[-1][-1]));cursor+=count
  poses.append(pose);grips.append(grip);segments.append({'category':cat,'uuid':uid,'instruction':s,'start_frame':cursor,'end_frame':cursor+len(pose)-1,'duration_s':dur});cursor+=len(pose)
 pose=np.concatenate(poses);grip=np.concatenate(grips);time=np.arange(len(pose))/FPS
 np.savez_compressed(OUT/'droid_franka_hard_20min.npz',time_s=time,tcp_xyz=pose[:,:3],tcp_quat_wxyz=pose[:,3:],gripper=grip)
 meta={'fps':FPS,'duration_s':float(time[-1]),'samples':len(time),'segments':segments,'selection':'hard human VR teleoperation, one global robot-base transform'}
 (OUT/'droid_franka_hard_20min_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('TOTAL',time[-1],len(segments))
if __name__=='__main__':main()
