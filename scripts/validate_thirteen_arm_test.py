"""Held-out evaluation of frozen global and per-task dense-search winners."""
from pathlib import Path
import json, math, sys, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from design_optimization.topology import load_templates
from design_optimization.kinematics import build_designs, deterministic_joint_samples, joint_world_positions
from design_optimization.ik import deterministic_seeds, solve_multistart
from design_optimization.taskspace import quaternion_wxyz_to_matrix, world_to_base_population
from design_optimization.collision import self_capsule_clearance, table_capsule_clearance
from design_optimization.installation_search_space import mount_transform_from_record

device='cuda' if torch.cuda.is_available() else 'cpu'; dense=json.loads((ROOT/'reports/single_arm/dense_search_results.json').read_text()); samples=json.loads((ROOT/'data/processed/local_pose_benchmark/test_samples.json').read_text()); templates=load_templates(ROOT/'reports/single_arm/model_audit.json',device=device); anchor_r=torch.tensor(((0.,-1.,0.),(-1.,0.,0.),(0.,0.,-1.)),device=device); profile_path=ROOT/'reports/single_arm/collision_proxy_profiles.json'; profiles=json.loads(profile_path.read_text())['robots'] if profile_path.exists() else {}
targets=[]; slices={}
for row in samples:
    lo=len(targets); p=torch.tensor(row['relative_position_m'],device=device); q=torch.tensor(row['relative_quaternion_wxyz'],device=device); anchor=torch.tensor((-.18 if row['hand']=='left' else .18,-.45,.35),device=device); rotations=anchor_r@quaternion_wxyz_to_matrix(q)
    for xyz,r in zip(anchor+p@anchor_r.T,rotations): t=torch.eye(4,device=device);t[:3,:3]=r;t[:3,3]=xyz;targets.append(t)
    slices[row['task']]=(lo,len(targets))
targets=torch.stack(targets); task_names=list(slices); output={'status':'held_out_test','frame_count':len(targets),'robots':{}}
def base_matrix(record):
    return torch.tensor(mount_transform_from_record(record), device=device, dtype=torch.float32)
for name,template in templates.items():
    selected=dense['robots'][name]; records=[selected['global']]+[selected['per_task'][t] for t in task_names]; raw=torch.tensor([r['scale_raw'] for r in records],device=device); designs=build_designs(template,raw,deterministic_joint_samples(template,2048)); bases=torch.stack([base_matrix(r) for r in records]); local=world_to_base_population(targets,bases); ik=solve_multistart(designs,local,deterministic_seeds(template,16),iterations=100); profile=profiles.get(name,{}); radii=profile.get('radii_m',(0.025,)*template.dof); allowed=profile.get('allowed_pairs',[])
    # Select among IK branches with the calibrated collision proxy active,
    # matching the project's collision-aware branch-selection policy.
    self_all=self_capsule_clearance(designs,ik.q,radii=radii,allowed_pairs=allowed); table_all=table_capsule_clearance(designs,ik.q,bases,radii=radii); clearance_all=torch.minimum(self_all,table_all); cost=400*ik.position_error_m+8*ik.orientation_error_rad+1e5*torch.relu(-clearance_all)
    branch=cost.argmin(2); gather=branch[...,None]; q=ik.q.gather(2,gather[...,None].expand(-1,-1,1,template.dof)).squeeze(2); pe=ik.position_error_m.gather(2,gather).squeeze(2);oe=ik.orientation_error_rad.gather(2,gather).squeeze(2); pose=ik.success.gather(2,gather).squeeze(2); self_c=self_all.gather(2,gather).squeeze(2);table_c=table_all.gather(2,gather).squeeze(2); safe=(self_c>=0)&(table_c>=0); final=pose&safe
    def metrics(i,lo,hi):
        pos_bad=pe[i,lo:hi]>.0025; ori_bad=(~pos_bad)&(oe[i,lo:hi]>.0261799388); self_bad=pose[i,lo:hi]&(self_c[i,lo:hi]<0); table_bad=pose[i,lo:hi]&(table_c[i,lo:hi]<0); n=hi-lo
        return {'frames':n,'pose_success_rate':float(pose[i,lo:hi].float().mean()),'success_rate':float(final[i,lo:hi].float().mean()),'frame_coverage':float(final[i,lo:hi].float().mean()),'position_rmse_m':float(torch.sqrt(pe[i,lo:hi].square().mean())),'orientation_rmse_rad':float(torch.sqrt(oe[i,lo:hi].square().mean())),'minimum_clearance_m':float(torch.minimum(self_c[i,lo:hi],table_c[i,lo:hi]).min()),'failure_counts':{'position':int(pos_bad.sum()),'orientation':int(ori_bad.sum()),'self_collision':int(self_bad.sum()),'table_collision':int(table_bad.sum())}}
    per={t:metrics(j+1,*slices[t]) for j,t in enumerate(task_names)}; global_m=metrics(0,0,len(targets)); aggregate={'frames':sum(x['frames'] for x in per.values()),'success_rate':sum(x['success_rate']*x['frames'] for x in per.values())/len(targets),'frame_coverage':sum(x['frame_coverage']*x['frames'] for x in per.values())/len(targets)}; output['robots'][name]={'global':global_m,'per_task':per,'per_task_aggregate':aggregate}
    # Renderer-neutral cache: only the selected per-task design and its 20
    # held-out frames are retained, already transformed into world space.
    world_points=joint_world_positions(designs,q); video_tasks={}
    for j,tname in enumerate(task_names):
        lo,hi=slices[tname]; idx=j+1; rotation=bases[idx,:3,:3]; translation=bases[idx,:3,3]; points=torch.einsum('ij,tkj->tki',rotation,world_points[idx,lo:hi])+translation; video_tasks[tname]={'points_world_m':points.tolist(),'target_world_m':targets[lo:hi,:3,3].tolist(),'safe':final[idx,lo:hi].tolist()}
    cache=ROOT/'videos/single_arm/cache'/f'{name}_all_tasks.json';cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps({'robot':name,'tasks':video_tasks}),encoding='utf-8')
    print(f'{name}: global={global_m["success_rate"]:.2%}, task={aggregate["success_rate"]:.2%}',flush=True)
out=ROOT/'reports/single_arm/held_out_test_results.json';out.write_text(json.dumps(output,indent=2),encoding='utf-8');print(out)
