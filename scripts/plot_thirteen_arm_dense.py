from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; payload=json.loads((ROOT/'reports/single_arm/dense_search_results.json').read_text()); data=payload['robots']; out=ROOT/'reports/single_arm/figures'; out.mkdir(parents=True,exist_ok=True)
ranked=sorted(data,key=lambda r:data[r]['global']['success_rate'],reverse=True); tasks=sorted(next(iter(data.values()))['per_task'])
global_rate=np.array([data[r]['global']['success_rate'] for r in ranked]); task_rate=np.array([[data[r]['per_task'][t]['success_rate'] for t in tasks] for r in ranked]); task_opt=task_rate.mean(1)
fig,ax=plt.subplots(figsize=(11,6)); y=np.arange(len(ranked)); ax.barh(y+.18,100*task_opt,height=.34,label='Per-task installation/ratio'); ax.barh(y-.18,100*global_rate,height=.34,label='One global installation/ratio'); ax.set(yticks=y,yticklabels=ranked,xlim=(0,101),xlabel='Success / frame coverage (%)'); ax.invert_yaxis(); ax.legend(); fig.tight_layout(); fig.savefig(out/'dense_global_vs_per_task.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(17,7)); im=ax.imshow(100*task_rate,aspect='auto',vmin=0,vmax=100,cmap='magma'); fig.colorbar(im,ax=ax,label='Success rate (%)'); ax.set(yticks=range(len(ranked)),yticklabels=ranked,xticks=range(len(tasks)),xticklabels=tasks); plt.setp(ax.get_xticklabels(),rotation=60,ha='right',fontsize=7); fig.tight_layout(); fig.savefig(out/'dense_task_robot_heatmap.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(10,6)); errors=[[1000*data[r]['per_task'][t]['position_rmse_m'] for t in tasks] for r in ranked]; ax.boxplot(errors,vert=False,tick_labels=ranked,showfliers=True); ax.set_xlabel('Per-task position RMSE (mm)'); ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out/'dense_error_distribution.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,7));
for r in ranked:
    xyz=np.array([data[r]['per_task'][t]['base_xyz_m'] for t in tasks]); ax.scatter(xyz[:,0],xyz[:,2],s=18,alpha=.55,label=r)
ax.set(xlabel='Base x (m)',ylabel='Base z (m)'); ax.legend(ncol=2,fontsize=7); fig.tight_layout(); fig.savefig(out/'dense_optimal_base_distribution.png',dpi=200); plt.close(fig)
lines=['# 13-arm dense screening','',f"Candidates/robot: {payload['candidate_count']}; task frames: {payload['frame_count']}; IK seeds: {payload['seed_count']}",'','|Rank|Robot|DOF|Global|Mean task-opt|Base xyz (m)|Tilt|','|---:|---|---:|---:|---:|---|---:|']
for i,r in enumerate(ranked,1):
 g=data[r]['global']; lines.append(f"|{i}|{r}|{data[r]['active_dof']}|{g['success_rate']:.2%}|{task_opt[i-1]:.2%}|{[round(x,3) for x in g['base_xyz_m']]}|{g['tilt_deg']:.1f}°|")
(ROOT/'reports/single_arm/DENSE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8'); print('\n'.join(lines))
