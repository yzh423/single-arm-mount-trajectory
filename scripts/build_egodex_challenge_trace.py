"""Select motion-rich EgoDex episodes and build a portable single-TCP trace."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/EgoDex/pose_only_test/egodex_pose_only_test.npz'
OUTDIR=ROOT/'data/EgoDex/selected_challenge'

def qangle(q):
    return 2*np.arccos(np.clip(np.abs(q[:,0]),0,1))

def resample(pos,q,src_fps=30.,dst_fps=15.):
    t=np.arange(len(pos))/src_fps; td=np.arange(0,t[-1]+1e-9,1/dst_fps)
    pd=np.column_stack([np.interp(td,t,pos[:,j]) for j in range(3)])
    rd=Slerp(t,Rotation.from_quat(q[:,[1,2,3,0]]))(td).as_quat()[:,[3,0,1,2]]
    return pd,rd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=8);ap.add_argument('--seconds',type=float,default=120.);a=ap.parse_args()
    z=np.load(SOURCE,allow_pickle=False); off=z['episode_offsets']; tasks=z['episode_task']
    arrays={side:{field:z[f'{side}_{field}'] for field in ('relative_xyz','relative_quat_wxyz','confidence')} for side in ('left','right')}
    candidates=[]
    for i,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
        n=int(hi-lo)
        if n<90:continue
        # Pick the more dynamic hand; confidence suppresses tracking-loss outliers.
        best=None
        for side in ('left','right'):
            p=arrays[side]['relative_xyz'][lo:hi];q=arrays[side]['relative_quat_wxyz'][lo:hi];conf=arrays[side]['confidence'][lo:hi]
            good=conf>.5
            if good.mean()<.8:continue
            path=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();ang=np.linalg.norm(np.diff(Rotation.from_quat(q[:,[1,2,3,0]]).as_rotvec(),axis=0),axis=1).sum()
            span=np.linalg.norm(np.ptp(p,axis=0));score=path+0.12*ang+2*span
            if best is None or score>best[0]:best=(score,side,p,q,path,ang,span)
        if best:candidates.append((best[0],i,*best[1:]))
    # Diversity: at most one episode per task until necessary.
    chosen=[];seen=set()
    for row in sorted(candidates,reverse=True):
        task=str(tasks[row[1]])
        if task in seen:continue
        chosen.append(row);seen.add(task)
        if len(chosen)>=a.episodes:break
    pose=[];segments=[];cursor=0;budget=int(a.seconds*15)
    # Put every episode centroid at the same workspace centroid used to optimize
    # the four DROID benchmark installations. EgoDex's ARKit origin follows no
    # robot base convention, so retaining its absolute offset would be arbitrary.
    anchor=np.array([.58817996,.05685095,.20327232]);canonical=Rotation.from_euler('xyz',[np.pi,0,np.pi/2])
    for score,idx,side,p,q,path,ang,span in chosen:
        room=budget-cursor
        if room<30:break
        p,rq=resample(p,q);p=p[:room];rq=rq[:room]
        # Preserve relative SE(3), but put every episode at one fair tabletop anchor.
        Rrel=Rotation.from_quat(rq[:,[1,2,3,0]]);R=(canonical*Rrel).as_quat()[:,[3,0,1,2]]
        pp=anchor+(p-p.mean(axis=0))
        pose.append(np.column_stack((pp,R)));segments.append(dict(index=int(idx),task=str(tasks[idx]),instruction=f'EgoDex {tasks[idx]} ({side} hand)',hand=side,start_frame=cursor,end_frame=cursor+len(pp)-1,duration_s=(len(pp)-1)/15,score=float(score),path_m=float(path),rotation_rad=float(ang),span_m=float(span)))
        cursor+=len(pp)
    out=np.concatenate(pose);OUTDIR.mkdir(parents=True,exist_ok=True);t=np.arange(len(out))/15
    np.savez_compressed(OUTDIR/'egodex_challenge.npz',time_s=t,tcp_xyz=out[:,:3],tcp_quat_wxyz=out[:,3:])
    manifest=dict(format='EgoDex relative hand SE(3), portable tabletop anchor',fps=15,duration_s=float(t[-1]),samples=len(t),segments=segments)
    (OUTDIR/'egodex_challenge_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
