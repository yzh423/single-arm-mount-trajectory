"""Independent collision-aware 13-arm search for DROID or EgoDex samples."""
from pathlib import Path
import argparse,json,math,sys,torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from design_optimization.topology import load_templates
from design_optimization.kinematics import build_designs,deterministic_joint_samples
from design_optimization.ik import deterministic_seeds,solve_multistart
from design_optimization.taskspace import quaternion_wxyz_to_matrix,world_to_base_population
from design_optimization.collision import self_capsule_clearance,table_capsule_clearance
from design_optimization.search_policy import dimension_aware_candidate_budget,staged_sobol_candidates
from design_optimization.batched_search import evaluate_candidate_batches,safe_candidate_batch_size
from design_optimization.installation_search_space import first_version_bounds,torch_mount_rotation_matrices
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=('droid','egodex'),required=True);ap.add_argument('--candidates',type=int,default=0,help='0 uses dimension-aware staged search');ap.add_argument('--candidate-batch',type=int,default=64);ap.add_argument('--robot');args=ap.parse_args();device='cuda' if torch.cuda.is_available() else 'cpu';samples=json.loads((ROOT/f'data/processed/external_domains/{args.domain}_samples.json').read_text())['splits']['validation'];templates=load_templates(ROOT/'reports/single_arm/model_audit.json',device=device);profiles=json.loads((ROOT/'reports/single_arm/collision_proxy_profiles.json').read_text())['robots'];anchor_r=torch.tensor(((0.,-1.,0.),(-1.,0.,0.),(0.,0.,-1.)),device=device);targets=[];slices={}
 for row in samples:
  lo=len(targets);p=torch.tensor(row['relative_position_m'],device=device);q=torch.tensor(row['relative_quaternion_wxyz'],device=device);anchor=torch.tensor((-.18 if row.get('hand')=='left' else .18,-.45,.35),device=device)
  for xyz,r in zip(anchor+p@anchor_r.T,anchor_r@quaternion_wxyz_to_matrix(q)):t=torch.eye(4,device=device);t[:3,:3]=r;t[:3,3]=xyz;targets.append(t)
  slices[row['task']]=(lo,len(targets))
 targets=torch.stack(targets);effective_batch=safe_candidate_batch_size(args.candidate_batch,frames=len(targets),seeds=10);space=first_version_bounds(7);search_dimension=len(space.names);candidate_count=args.candidates or dimension_aware_candidate_budget(search_dimension);staged=staged_sobol_candidates(space.lower,space.upper,space.incumbent,budget=candidate_count,seed=750);candidate_vectors=torch.tensor(staged,device=device,dtype=torch.float32);out=ROOT/f'reports/single_arm/domains/{args.domain}/dense_results.json';partial=out.with_name('dense_results.partial.json');metadata={'domain':args.domain,'split':'validation','tasks':len(slices),'frames':len(targets),'search_dimension':search_dimension,'search_parameter_names':list(space.names),'installation_bounds':{'lower':space.lower[-6:].tolist(),'upper':space.upper[-6:].tolist(),'rotation_order':'Rz(yaw) @ Ry(tilt_pitch) @ Rx(roll)'},'candidate_count':candidate_count,'candidate_batch_size':effective_batch,'requested_candidate_batch_size':args.candidate_batch,'maximum_batched_ik_systems':30000,'search_policy':'75% scrambled Sobol global + 25% bounded incumbent-local'};output={**metadata,'robots':{}};out.parent.mkdir(parents=True,exist_ok=True)
 if partial.is_file():
  saved=json.loads(partial.read_text());
  compatibility=('domain','split','tasks','frames','search_dimension','search_parameter_names','installation_bounds','candidate_count','search_policy')
  if all(saved.get(k)==metadata[k] for k in compatibility):
   output=saved;output.update({k:metadata[k] for k in ('candidate_batch_size','requested_candidate_batch_size','maximum_batched_ik_systems')})
 if args.robot:
  if args.robot not in templates:raise ValueError(f'unknown robot: {args.robot}')
  templates={args.robot:templates[args.robot]}
 for name,t in templates.items():
  profile=profiles[name]
  if name in output['robots']:
   print(args.domain,name,'checkpoint-skip',flush=True);continue
  def evaluate(batch,_start):
   raw_batch=batch[:,:7];mount_batch=batch[:,7:];count=len(batch);bases=torch.eye(4,device=device).repeat(count,1,1);bases[:,:3,:3]=torch_mount_rotation_matrices(mount_batch[:,3],mount_batch[:,4],mount_batch[:,5]);bases[:,:3,3]=mount_batch[:,:3];local=world_to_base_population(targets,bases);design=build_designs(t,raw_batch[:,:t.dof],deterministic_joint_samples(t,2048));ik=solve_multistart(design,local,deterministic_seeds(t,10),iterations=60);self_c=self_capsule_clearance(design,ik.q,radii=profile['radii_m'],allowed_pairs=profile['allowed_pairs']);table_c=table_capsule_clearance(design,ik.q,bases,radii=profile['radii_m']);clear=torch.minimum(self_c,table_c);cost=400*ik.position_error_m+8*ik.orientation_error_rad+1e5*torch.relu(-clear);branch=cost.argmin(2);g=branch[...,None];pose=ik.success.gather(2,g).squeeze(2);selected_clear=clear.gather(2,g).squeeze(2);return {'safe':pose&(selected_clear>=0),'pose':pose,'pe':ik.position_error_m.gather(2,g).squeeze(2),'oe':ik.orientation_error_rad.gather(2,g).squeeze(2)}
  evaluated=evaluate_candidate_batches(candidate_vectors,batch_size=effective_batch,evaluate=evaluate);safe,pose,pe,oe=(evaluated[k] for k in ('safe','pose','pe','oe'));rates=safe.float().mean(1);winner=int(rates.argmax())
  def rec(i,lo=0,hi=None):
   hi=len(targets) if hi is None else hi;return {'candidate':i,'success_rate':float(safe[i,lo:hi].float().mean()),'pose_success_rate':float(pose[i,lo:hi].float().mean()),'position_rmse_m':float(torch.sqrt(pe[i,lo:hi].square().mean())),'orientation_rmse_rad':float(torch.sqrt(oe[i,lo:hi].square().mean())),'scale_raw':staged[i,:t.dof].tolist(),'base_xyz_m':staged[i,7:10].tolist(),'tilt_deg':float(staged[i,10]),'yaw_deg':float(staged[i,11]),'roll_deg':float(staged[i,12])}
  per={};
  for task,(lo,hi) in slices.items():rr=safe[:,lo:hi].float().mean(1);per[task]=rec(int(rr.argmax()),lo,hi)
  output['robots'][name]={'active_dof':t.dof,'global':rec(winner),'per_task':per};temp=partial.with_suffix('.tmp');temp.write_text(json.dumps(output,indent=2),encoding='utf-8');temp.replace(partial);print(args.domain,name,f'{rates[winner]:.2%}',flush=True)
 expected=set(load_templates(ROOT/'reports/single_arm/model_audit.json',device='cpu'))
 if set(output['robots'])==expected:out.write_text(json.dumps(output,indent=2),encoding='utf-8');print(out)
 else:print(partial)
if __name__=='__main__':main()
