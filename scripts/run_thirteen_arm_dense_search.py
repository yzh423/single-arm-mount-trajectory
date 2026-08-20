"""Dense deterministic joint geometry/mount search with global and per-task winners."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from design_optimization.topology import TopologyTemplate
from design_optimization.robot_registry import RobotSpec
from design_optimization.urdf_chain import load_native_chain
from design_optimization.kinematics import assemble_from_deltas
from design_optimization.ik import deterministic_seeds, solve_multistart
from design_optimization.taskspace import quaternion_wxyz_to_matrix, world_to_base_population
from design_optimization.collision import self_capsule_clearance, table_capsule_clearance
from design_optimization.search_policy import dimension_aware_candidate_budget, staged_sobol_candidates
from design_optimization.batched_search import evaluate_candidate_batches
from design_optimization.installation_search_space import first_version_bounds, torch_mount_rotation_matrices
from scripts.strict_trajectory_sources import local_task_anchor_z, task_anchor_rotations
from design_optimization.fidelity_funnel import select_diverse_candidates


def select_proxy_shortlist(staged, ranks, *, lower, upper, size, record):
    """Persist traceable proxy candidates while covering distinct mount basins."""
    mounts = np.asarray(staged, dtype=float)
    if not 1 <= int(size) <= len(mounts):
        raise ValueError("shortlist size must fit the coarse candidate population")
    # The proxy is deliberately not a hard rejection gate: historical
    # capsule-to-real-URDF rank correlation is weak, so diversity must see the
    # entire broad population instead of only the proxy top quartile.
    pool_size = len(mounts)
    selected = select_diverse_candidates(
        mounts, ranks, lower=np.asarray(lower), upper=np.asarray(upper),
        count=int(size), pool_size=pool_size)
    rows = []
    for index in selected:
        row = dict(record(int(index)))
        row["candidate_id"] = int(index)
        row["mount"] = mounts[index].tolist()
        rows.append(row)
    return rows


def load_official_search_templates(*, device="cpu", dtype=torch.float32):
    """Build coarse-search chains from the exact strict-model authority."""
    from scripts.strict_urdf_model_audit import MODELS
    templates = {}
    for name, entry in MODELS.items():
        spec = RobotSpec(
            robot_id=name,
            source_model=entry.path.resolve(),
            model_format="urdf",
            base_link=entry.base_link,
            flange_link=entry.tcp_parent,
            active_joints=entry.joints,
            locked_joints_rad=entry.locked_joint_ranges or {},
        )
        chain = load_native_chain(spec)
        tool = torch.eye(4, dtype=dtype, device=device)
        tool[:3, 3] = torch.tensor(entry.tool_translation_m, dtype=dtype, device=device)
        rotation = np.zeros(9)
        __import__("mujoco").mju_quat2Mat(rotation, np.asarray(entry.tool_quaternion_wxyz))
        tool[:3, :3] = torch.tensor(rotation.reshape(3, 3), dtype=dtype, device=device)
        templates[name] = TopologyTemplate(
            name=name,
            axes=torch.tensor(chain.axes, dtype=dtype, device=device),
            deltas=torch.tensor(chain.deltas, dtype=dtype, device=device),
            home_rotation=torch.tensor(chain.home[:3, :3], dtype=dtype, device=device),
            q_min=torch.tensor(chain.q_min, dtype=dtype, device=device),
            q_max=torch.tensor(chain.q_max, dtype=dtype, device=device),
            tool_length_m=entry.tool_offset_m,
            tool_transform=tool,
            first_joint_origin_m=torch.tensor(chain.points[0], dtype=dtype, device=device),
            source_model=entry.path.resolve(),
            tcp_authority=entry.tcp_authority,
        )
    return templates


def build_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=int,default=0,help='0 uses the dimension-aware staged budget')
    parser.add_argument('--candidate-batch',type=int,default=64)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=60)
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--robots',nargs='+')
    parser.add_argument('--episode-artifact',type=Path,
                        help='one processed Local episode for an independent per-episode mount search')
    parser.add_argument('--task',help='task name for --episode-artifact')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--input-fingerprint')
    parser.add_argument('--shortlist-size',type=int,default=0,
                        help='persist this many diverse candidate-level proxy rows per task')
    return parser


def load_search_samples(args):
    if args.episode_artifact is None:
        samples=json.loads((ROOT/'data/processed/local_pose_benchmark/pilot_samples.json').read_text())
        return select_task_samples(samples,args.tasks)
    if not args.task:
        raise ValueError('--task is required with --episode-artifact')
    from scripts.strict_trajectory_sources import load_relative_task_trajectory, resample_trajectory
    trajectory=resample_trajectory(load_relative_task_trajectory(
        'local',args.task,episode_artifact=args.episode_artifact),maximum_frames=300)
    return [{'task':args.task,'hand':trajectory.hand,
             'episode_id':Path(trajectory.source).stem,
             'relative_position_m':trajectory.position_m.tolist(),
             'relative_quaternion_wxyz':trajectory.quaternion_wxyz.tolist()}]


def yaw_only_mount_space():
    """Return the only active coarse-search coordinates: base XYZ and yaw."""
    full_space = first_version_bounds(7)
    mount_names = full_space.names[-6:]
    indices = np.asarray((0, 1, 2, 4), dtype=int)
    return (tuple(mount_names[index] for index in indices),
            full_space.lower[-6:][indices], full_space.upper[-6:][indices],
            full_space.incumbent[-6:][indices])


def expand_yaw_only_mounts(active):
    active = torch.as_tensor(active)
    if active.ndim != 2 or active.shape[1] != 4:
        raise ValueError("active GPU mount candidates must have shape (N, 4)")
    expanded = torch.zeros((len(active), 6), device=active.device, dtype=active.dtype)
    expanded[:, :3] = active[:, :3]
    expanded[:, 4] = active[:, 3]
    return expanded


def select_task_samples(samples, requested_tasks):
    if not requested_tasks:
        return samples
    available = {row['task'] for row in samples}
    missing = [task for task in requested_tasks if task not in available]
    if missing:
        raise ValueError(f"missing tasks: {', '.join(missing)}")
    requested = set(requested_tasks)
    return [row for row in samples if row['task'] in requested]

def select_robots(templates, requested_robots):
    if not requested_robots:
        return templates
    missing=[name for name in requested_robots if name not in templates]
    if missing:
        raise ValueError(f"missing robots: {', '.join(missing)}")
    return {name:templates[name] for name in requested_robots}

def main():
    args=build_parser().parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; dtype=torch.float32
    samples=load_search_samples(args)
    templates=load_official_search_templates(device=device, dtype=dtype)
    templates=select_robots(templates,args.robots)
    profile_path=ROOT/'reports/single_arm/collision_proxy_profiles.json'; profiles=json.loads(profile_path.read_text())['robots']
    targets=[]; slices={}; episode_slices=[]
    for row in samples:
        start=len(targets); rel=torch.tensor(row['relative_position_m'],device=device); quat=torch.tensor(row['relative_quaternion_wxyz'],device=device)
        anchor=torch.tensor((-.18 if row['hand']=='left' else .18,-.45,local_task_anchor_z(row['task'])),device=device)
        rotation=torch.tensor(task_anchor_rotations(row['task'],np.asarray(row['relative_quaternion_wxyz'])),device=device,dtype=dtype)
        for xyz,rot in zip(anchor+rel,rotation):
            t=torch.eye(4,device=device); t[:3,:3]=rot; t[:3,3]=xyz; targets.append(t)
        first = slices[row['task']][0] if row['task'] in slices else start
        slices[row['task']] = (first, len(targets))
        episode_slices.append({'task': row['task'], 'hand': row['hand'], 'lo': start, 'hi': len(targets)})
    targets=torch.stack(targets)
    mount_names,mount_lower,mount_upper,mount_incumbent=yaw_only_mount_space()
    search_dimension=len(mount_names); candidate_count=args.candidates or dimension_aware_candidate_budget(search_dimension)
    staged=staged_sobol_candidates(mount_lower,mount_upper,mount_incumbent,budget=candidate_count,seed=750)
    candidate_vectors=torch.tensor(staged,device=device,dtype=dtype)
    output={'status':'native_length_mount_only_search','input_fingerprint':args.input_fingerprint,'device':device,'search_dimension':search_dimension,'search_parameter_names':mount_names,'installation_bounds':{'lower':mount_lower.tolist(),'upper':mount_upper.tolist(),'rotation_order':'Rz(yaw) @ Ry(tilt_pitch) @ Rx(roll)'},'candidate_count':candidate_count,'candidate_batch_size':args.candidate_batch,'search_policy':'75% scrambled Sobol global + 25% bounded incumbent-local','geometry_policy':'native arm length and audited physical TCP transform; no morphology scaling','seed_count':args.seeds,'iterations':args.iterations,'frame_count':len(targets),'robots':{}}
    outdir=ROOT/'reports/single_arm/dense'; outdir.mkdir(parents=True,exist_ok=True)
    for name,template in templates.items():
        profile=profiles[name]; radii=profile['radii_m']; allowed=profile['allowed_pairs']
        def evaluate(batch, _start):
            mount_batch=expand_yaw_only_mounts(batch); count=len(batch)
            bases=torch.eye(4,device=device).repeat(count,1,1); bases[:,:3,:3]=torch_mount_rotation_matrices(mount_batch[:,3],mount_batch[:,4],mount_batch[:,5]); bases[:,:3,3]=mount_batch[:,:3]
            normal=bases[:,:3,2]; normal_z=normal[:,2]
            feasible=(normal_z>0.10)&(mount_batch[:,0].abs()<=0.95)&(mount_batch[:,1].abs()<=0.72)
            local=world_to_base_population(targets,bases); designs=assemble_from_deltas(template,template.deltas[None].expand(count,-1,-1))
            ik=solve_multistart(designs,local,deterministic_seeds(template,args.seeds),iterations=args.iterations)
            self_all=self_capsule_clearance(designs,ik.q,radii=radii,allowed_pairs=allowed); table_all=table_capsule_clearance(designs,ik.q,bases,radii=radii); clearance_all=torch.minimum(self_all,table_all)
            cost=400*ik.position_error_m+8*ik.orientation_error_rad+1e5*torch.relu(-clearance_all); branch=cost.argmin(2); gather=branch[...,None]
            pose=ik.success.gather(2,gather).squeeze(2); clearance=clearance_all.gather(2,gather).squeeze(2)
            return {'success':pose&(clearance>=0)&feasible[:,None],'pose':pose&feasible[:,None],'pe':ik.position_error_m.gather(2,gather).squeeze(2),'oe':ik.orientation_error_rad.gather(2,gather).squeeze(2),'clearance':clearance}
        evaluated=evaluate_candidate_batches(candidate_vectors,batch_size=args.candidate_batch,evaluate=evaluate)
        success,pose,pe,oe,clearance=(evaluated[k] for k in ('success','pose','pe','oe','clearance'))
        def relevant_episodes(task=None):
            return [row for row in episode_slices if task is None or row['task'] == task]
        def candidate_rank(index, task=None):
            rows = relevant_episodes(task)
            complete = sum(bool(success[index,row['lo']:row['hi']].all()) for row in rows)
            ids = torch.cat([torch.arange(row['lo'],row['hi'],device=success.device) for row in rows])
            failures = []
            for row in rows:
                values=(~success[index,row['lo']:row['hi']]).detach().cpu().numpy(); longest=current=0
                for value in values:
                    current=current+1 if value else 0; longest=max(longest,current)
                failures.append(longest)
            return (complete, float(success[index,ids].float().mean()), -max(failures),
                    -float(torch.sqrt(pe[index,ids].square().mean())),
                    -float(torch.sqrt(oe[index,ids].square().mean())))
        winner=max(range(len(staged)),key=lambda index:candidate_rank(index))
        def record(index,task=None):
            rows=relevant_episodes(task); ids=torch.cat([torch.arange(row['lo'],row['hi'],device=success.device) for row in rows])
            completed=sum(bool(success[index,row['lo']:row['hi']].all()) for row in rows)
            coverage=float(success[index,ids].float().mean())
            return {'candidate':index,'success_rate':completed/len(rows),'episode_success_rate':completed/len(rows),'successful_episodes':completed,'episodes':len(rows),'pose_success_rate':float(pose[index,ids].float().mean()),'frame_coverage':coverage,'position_rmse_m':float(torch.sqrt(pe[index,ids].square().mean())),'orientation_rmse_rad':float(torch.sqrt(oe[index,ids].square().mean())),'minimum_clearance_m':float(clearance[index,ids].min()),'geometry':'native_length_fixed','base_xyz_m':staged[index,:3].tolist(),'tilt_deg':0.0,'yaw_deg':float(staged[index,3]),'roll_deg':0.0}
        task_rows={}
        for task in slices:
            task_ranks=[candidate_rank(index,task) for index in range(len(staged))]
            task_winner=max(range(len(staged)),key=lambda index:task_ranks[index]); task_rows[task]=record(task_winner,task)
            if args.shortlist_size:
                task_rows[task]['diverse_shortlist']=select_proxy_shortlist(
                    staged,task_ranks,lower=mount_lower,upper=mount_upper,
                    size=min(args.shortlist_size,len(staged)),
                    record=lambda index, selected_task=task: record(index,selected_task))
        row={'active_dof':template.dof,
             'mount_selection_priority':'successful whole episodes > frame coverage > shortest failure run > position RMSE > orientation RMSE',
             'global':record(winner),'per_task':task_rows}; output['robots'][name]=row
        (outdir/f'{name}.json').write_text(json.dumps(row,indent=2),encoding='utf-8'); print(f'{name}: {row["global"]["success_rate"]:.2%}',flush=True)
    final=args.output or ROOT/'reports/single_arm/dense_search_results.json'; final.parent.mkdir(parents=True,exist_ok=True); final.write_text(json.dumps(output,indent=2),encoding='utf-8'); print(final)
if __name__=='__main__': main()
