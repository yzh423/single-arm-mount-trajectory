"""Reference-style 4x4 MuJoCo diagnostic panel comparison for all 13 arms."""
from pathlib import Path
import json, math, subprocess, time
import numpy as np
import mujoco, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1];CACHE=json.loads((ROOT/'videos/single_arm/cache/open_box_chains.json').read_text());NAMES=list(CACHE['robots']);TW,TH,COLS,ROWS=400,225,4,4;W,H,FPS,FRAMES=TW*COLS,TH*ROWS,10,300
COLORS=[(.16,.47,.82,1),(.10,.63,.43,1),(.88,.43,.16,1),(.48,.32,.75,1),(.12,.62,.68,1),(.82,.25,.35,1),(.35,.58,.22,1),(.65,.46,.20,1),(.25,.45,.62,1),(.56,.32,.54,1),(.30,.62,.50,1),(.72,.35,.18,1),(.38,.42,.48,1)]
def vals(x):return ' '.join(f'{float(v):.8g}' for v in x)
def build(name,row,color):
 base=row['base_xyz_m'];tilt=math.radians(row['tilt_deg']);quat=(math.cos(tilt/2),0,math.sin(tilt/2),0);indent='    ';parts=[f'<mujoco model="{name}"><compiler angle="radian"/><option gravity="0 0 -9.81"/><visual><global offwidth="{TW}" offheight="{TH}" fovy="48"/><quality shadowsize="2048" offsamples="4"/><headlight ambient=".35 .35 .35" diffuse=".62 .62 .62"/></visual><asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".84 .88 .94" rgb2=".48 .58 .70" width="256" height="1024"/><texture name="grid" type="2d" builtin="checker" rgb1=".75 .76 .74" rgb2=".64 .65 .63" width="256" height="256"/><material name="floor" texture="grid" texrepeat="8 8"/></asset><worldbody><light pos="0 -1 2" dir="0 .3 -1" directional="true"/><geom type="plane" size="2 2 .05" material="floor"/><body name="root" pos="{vals(base)}" quat="{vals(quat)}">']
 for i,(axis,delta,lo,hi) in enumerate(zip(row['axes'],row['deltas_m'],row['q_min_rad'],row['q_max_rad'])):
  if i:parts.append(f'{indent}<body name="link{i}" pos="{vals(row["deltas_m"][i-1])}">');indent+='  '
  parts.append(f'{indent}<joint name="j{i+1}" type="hinge" axis="{vals(axis)}" range="{lo:.8g} {hi:.8g}" limited="true" damping="1"/>');length=np.linalg.norm(delta);radius=max(.014,min(.04,.11*length));parts.append(f'{indent}<geom type="capsule" fromto="0 0 0 {vals(delta)}" size="{radius:.6g}" rgba="{vals(color)}"/>' if length>1e-6 else f'{indent}<geom type="sphere" size="{radius:.6g}" rgba="{vals(color)}"/>')
 parts.append(f'{indent}<site name="tcp" pos="{vals(row["deltas_m"][-1])}" size=".028" rgba="0.02 .8 .45 1"/>')
 for _ in row['axes']:indent=indent[:-2];parts.append(f'{indent}</body>')
 parts.append('<body name="target" mocap="true"><geom type="sphere" size=".035" rgba=".9 .08 .08 .95" contype="0" conaffinity="0"/></body></worldbody></mujoco>');model=mujoco.MjModel.from_xml_string('\n'.join(parts));data=mujoco.MjData(model);adr=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'j{i+1}')]) for i in range(len(row['axes']))];mid=int(model.body_mocapid[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'target')]);renderer=mujoco.Renderer(model,height=TH,width=TW);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[base[0],base[1]-.18,max(.28,base[2]+.28)];cam.distance=1.55;cam.azimuth=145;cam.elevation=-24;return model,data,adr,mid,renderer,cam
contexts=[build(n,CACHE['robots'][n],c) for n,c in zip(NAMES,COLORS)];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);output=ROOT/'videos/single_arm/mujoco/thirteen_arm_open_box_mujoco_panels_30s.mp4';ff=imageio_ffmpeg.get_ffmpeg_exe();proc=subprocess.Popen([ff,'-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE);started=time.time()
for frame in range(FRAMES):
 phase=frame*19/(FRAMES-1);lo=int(phase);hi=min(lo+1,19);a=phase-lo;canvas=Image.new('RGB',(W,H),(235,238,241))
 for index,(name,ctx) in enumerate(zip(NAMES,contexts)):
  model,data,addresses,mocap,renderer,cam=ctx;row=CACHE['robots'][name];q=(1-a)*np.asarray(row['q_rad'][lo])+a*np.asarray(row['q_rad'][hi]);
  for address,value in zip(addresses,q):data.qpos[address]=value
  target=(1-a)*np.asarray(row['target_world_m'][lo])+a*np.asarray(row['target_world_m'][hi]);data.mocap_pos[mocap]=target;mujoco.mj_forward(model,data);renderer.update_scene(data,camera=cam);tile=Image.fromarray(renderer.render());draw=ImageDraw.Draw(tile,'RGBA');draw.rounded_rectangle((8,7,TW-8,49),8,fill=(250,250,250,220));safe=bool(row['safe'][lo]);draw.text((17,10),name,font=font,fill=(22,27,33,255));draw.text((17,31),f't={frame/FPS:4.1f}s  '+('SAFE' if safe else 'FAILED'),font=small,fill=((22,135,78,255) if safe else (205,48,48,255)));canvas.paste(tile,((index%COLS)*TW,(index//COLS)*TH))
 draw=ImageDraw.Draw(canvas,'RGBA');draw.rounded_rectangle((3*TW+8,3*TH+8,W-8,H-8),10,fill=(250,250,250,245));draw.text((3*TW+22,3*TH+20),'MuJoCo 3.3.7',font=font,fill=(25,30,36,255));draw.text((3*TW+22,3*TH+48),'qpos trajectory following',font=small,fill=(65,72,80,255));draw.text((3*TW+22,3*TH+68),'red target / green TCP',font=small,fill=(65,72,80,255));draw.text((3*TW+22,3*TH+88),'Local held-out: open-box',font=small,fill=(65,72,80,255));proc.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
proc.stdin.close();code=proc.wait()
for *_,renderer,_ in contexts:renderer.close()
if code:raise RuntimeError(f'ffmpeg exited with {code}')
meta={'renderer':'MuJoCo Renderer panel composite','mujoco_version':mujoco.__version__,'qpos_following':True,'mj_forward_each_frame':True,'width':W,'height':H,'fps':FPS,'frames':FRAMES,'duration_s':30,'robots':NAMES,'elapsed_s':time.time()-started};output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(output);print(meta)
