"""Compile every registered source model in MuJoCo without geometric substitution."""
from pathlib import Path
import json,sys,mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from design_optimization.robot_registry import load_robot_registry
registry=load_robot_registry(ROOT/'configs/robot_registry_13.yaml');rows={}
for name,spec in registry.items():
 try:
  mj=mujoco.MjSpec.from_file(str(spec.source_model));model=mj.compile();missing=[j for j in spec.active_joints if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,j)<0];mesh_geoms=sum(int(model.geom_type[i])==int(mujoco.mjtGeom.mjGEOM_MESH) for i in range(model.ngeom));rows[name]={'status':'pass' if not missing else 'fail','source':str(spec.source_model),'format':spec.model_format,'nbody':model.nbody,'njnt':model.njnt,'ngeom':model.ngeom,'nmesh':model.nmesh,'mesh_geoms':mesh_geoms,'missing_active_joints':missing};print(name,rows[name],flush=True)
 except Exception as e:rows[name]={'status':'fail','source':str(spec.source_model),'format':spec.model_format,'error':f'{type(e).__name__}: {e}'};print(name,rows[name],flush=True)
out=ROOT/'reports/single_arm/mujoco_source_model_audit.json';out.write_text(json.dumps({'status':'pass' if all(x['status']=='pass' for x in rows.values()) else 'fail','robots':rows},indent=2),encoding='utf-8');print(out)
