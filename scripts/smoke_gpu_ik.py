from pathlib import Path
import argparse, sys, time
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from design_optimization.kinematics import build_designs, deterministic_joint_samples, fk_tcp
from design_optimization.ik import deterministic_seeds, solve_multistart
from design_optimization.topology import load_templates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--topology',default='xarm6');ap.add_argument('--targets',type=int,default=128);ap.add_argument('--seeds',type=int,default=8);a=ap.parse_args()
    templates=load_templates(ROOT/'reports/parametric_topology_audit.json',device='cuda');t=templates[a.topology]
    calibration=deterministic_joint_samples(t,4096);design=build_designs(t,torch.zeros((1,6),device='cuda'),calibration)
    generator=torch.Generator(device='cuda');generator.manual_seed(19)
    q=t.q_min+torch.rand((a.targets,6),generator=generator,device='cuda')*(t.q_max-t.q_min)
    targets=fk_tcp(design,q)[0]
    seeds=deterministic_seeds(t,a.seeds);torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
    result=solve_multistart(design,targets,seeds,iterations=80)
    torch.cuda.synchronize();elapsed=time.perf_counter()-tic
    solved=result.success.any(dim=-1)
    best=result.position_error_m.min(dim=-1).values
    print(f'{a.topology} targets={a.targets} seeds={a.seeds} solved={solved.float().mean().item()*100:.2f}% '
          f'best_position_rmse_mm={1000*torch.sqrt((best*best).mean()).item():.3f} time={elapsed:.3f}s '
          f'peak={torch.cuda.max_memory_allocated()/2**20:.1f}MiB')


if __name__=='__main__':main()
