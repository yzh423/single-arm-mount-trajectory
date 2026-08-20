"""Held-out test of frozen DROID/EgoDex global and task-specific parameters."""
from pathlib import Path
import argparse,json,math,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from design_optimization.topology import load_templates
from design_optimization.kinematics import build_designs,deterministic_joint_samples
from design_optimization.ik import deterministic_seeds,solve_multistart
from design_optimization.taskspace import quaternion_wxyz_to_matrix,world_to_base_population
from design_optimization.collision import self_capsule_clearance,table_capsule_clearance
from design_optimization.installation_search_space import mount_transform_from_record
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=('droid','egodex'),required=True);args=ap.parse_args();device='cuda' if torch.cuda.is_available() else 'cpu';rows=json.loads((ROOT/f'data/processed/external_domains/{args.domain}_samples.json').read_text())['splits']['test'];frozen=json.loads((ROOT/f'reports/single_arm/domains/{args.domain}/dense_results.json').read_text())['robots'];templates=load_templates(ROOT/'reports/single_arm/model_audit.json',device=device);profiles=json.loads((ROOT/'reports/single_arm/collision_proxy_profiles.json').read_text())['robots'];anchor_r=torch.tensor(((0.,-1.,0.),(-1.,0.,0.),(0.,0.,-1.)),device=device);task_targets=[]
 for row in rows:
  p=torch.tensor(row['relative_position_m'],device=device);q=torch.tensor(row['relative_quaternion_wxyz'],device=device);anchor=torch.tensor((-.18 if row.get('hand')=='left' else .18,-.45,.35),device=device);frames=[]
  for xyz,r in zip(anchor+p@anchor_r.T,anchor_r@quaternion_wxyz_to_matrix(q)):t=torch.eye(4,device=device);t[:3,:3]=r;t[:3,3]=xyz;frames.append(t)
  task_targets.append(torch.stack(frames))
 all_targets=torch.cat(task_targets);output={'domain':args.domain,'split':'test','tasks':len(rows),'frames':len(all_targets),'robots':{}}
 def base(record):
  return torch.tensor(mount_transform_from_record(record),device=device,dtype=torch.float32)
 for name,t in templates.items():
  profile=profiles[name]
  def evaluate(records,targets):
   raw=torch.tensor([r['scale_raw'] for r in records],device=device);design=build_designs(t,raw,deterministic_joint_samples(t,2048));bases=torch.stack([base(r) for r in records]);local=world_to_base_population(targets,bases) if targets.ndim==3 else torch.stack([world_to_base_population(x,b[None])[0] for x,b in zip(targets,bases)]);ik=solve_multistart(design,local,deterministic_seeds(t,16),iterations=100);self_c=self_capsule_clearance(design,ik.q,radii=profile['radii_m'],allowed_pairs=profile['allowed_pairs']);table_c=table_capsule_clearance(design,ik.q,bases,radii=profile['radii_m']);clear=torch.minimum(self_c,table_c);cost=400*ik.position_error_m+8*ik.orientation_error_rad+1e5*torch.relu(-clear);branch=cost.argmin(2);g=branch[...,None];pose=ik.success.gather(2,g).squeeze(2);safe=pose&(clear.gather(2,g).squeeze(2)>=0);pe=ik.position_error_m.gather(2,g).squeeze(2);oe=ik.orientation_error_rad.gather(2,g).squeeze(2);return safe,pose,pe,oe
  global_record=frozen[name]['global'];safe,pose,pe,oe=evaluate([global_record],all_targets);global_metrics={'success_rate':float(safe.float().mean()),'pose_success_rate':float(pose.float().mean()),'position_rmse_m':float(torch.sqrt(pe.square().mean())),'orientation_rmse_rad':float(torch.sqrt(oe.square().mean()))}
  records=[];fallback=[]
  for row in rows:
   found=frozen[name]['per_task'].get(row['task']);records.append(found or global_record);fallback.append(found is None)
  safe,pose,pe,oe=evaluate(records,torch.stack(task_targets));per={}
  for i,row in enumerate(rows):per[row['task']]={'success_rate':float(safe[i].float().mean()),'pose_success_rate':float(pose[i].float().mean()),'position_rmse_m':float(torch.sqrt(pe[i].square().mean())),'orientation_rmse_rad':float(torch.sqrt(oe[i].square().mean())),'used_global_fallback':fallback[i]}
  aggregate=float(safe.float().mean());output['robots'][name]={'global':global_metrics,'per_task_aggregate_success_rate':aggregate,'per_task':per,'fallback_task_count':sum(fallback)};print(args.domain,name,f'global={global_metrics["success_rate"]:.2%}',f'task={aggregate:.2%}',f'fallback={sum(fallback)}',flush=True)
 out=ROOT/f'reports/single_arm/domains/{args.domain}/test_results.json';out.write_text(json.dumps(output,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
