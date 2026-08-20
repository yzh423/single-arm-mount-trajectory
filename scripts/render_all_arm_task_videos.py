"""Batch-render one compact MP4 for every robot/task cached trajectory."""
from pathlib import Path
import hashlib, json, re
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];cache_dir=ROOT/'videos/single_arm/cache';out_root=ROOT/'videos/single_arm/per_arm_task';width,height=640,480;fps,duration=5,5;frame_count=fps*duration
def slug(x):return re.sub(r'[^a-zA-Z0-9_-]+','_',x)
def project(points):
    x=(points[:,0]+.8)/1.6*(width-80)+40;z=(1.05-points[:,2])/1.2*(height-80)+40;return np.stack((x,z),1)
manifest=[]
for cache_path in sorted(cache_dir.glob('*_all_tasks.json')):
 data=json.loads(cache_path.read_text());robot=data['robot'];directory=out_root/robot;directory.mkdir(parents=True,exist_ok=True)
 for task,row in data['tasks'].items():
  output=directory/f'{slug(task)}.mp4';points=np.asarray(row['points_world_m']);targets=np.asarray(row['target_world_m']);safe=row['safe'];writer=imageio_ffmpeg.write_frames(str(output),(width,height),fps=fps,codec='libx264',quality=7,pix_fmt_in='rgb24',output_params=['-pix_fmt','yuv420p','-movflags','+faststart']);writer.send(None)
  for frame in range(frame_count):
   phase=frame*(len(points)-1)/(frame_count-1);lo=int(phase);hi=min(lo+1,len(points)-1);a=phase-lo;p=(1-a)*points[lo]+a*points[hi];target=(1-a)*targets[lo]+a*targets[hi];image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image);draw.rectangle((40,40,width-40,height-40),outline=(205,210,218),width=2);draw.line((40,project(np.array([[0,0,0]]))[0,1],width-40,project(np.array([[0,0,0]]))[0,1]),fill=(150,150,150),width=2);xy=project(p);color=(38,150,83) if safe[lo] else (210,55,55);draw.line([tuple(v) for v in xy],fill=color,width=6,joint='curve');
   for x,y in xy:draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
   tx,tz=project(target[None])[0];draw.line((tx-8,tz,tx+8,tz),fill=(240,170,0),width=4);draw.line((tx,tz-8,tx,tz+8),fill=(240,170,0),width=4);draw.text((50,50),f'{robot} | {task}',fill=(20,25,35));draw.text((50,72),f't={frame/fps:4.1f}s  '+('SAFE' if safe[lo] else 'FAILED'),fill=color);writer.send(np.asarray(image).tobytes())
  writer.close();digest=hashlib.sha256(output.read_bytes()).hexdigest();manifest.append({'robot':robot,'task':task,'path':str(output.relative_to(ROOT)).replace('\\','/'),'duration_s':duration,'fps':fps,'sha256':digest});print(robot,task,flush=True)
manifest_path=ROOT/'videos/single_arm/per_arm_task_manifest.json';manifest_path.write_text(json.dumps({'count':len(manifest),'videos':manifest},indent=2),encoding='utf-8');print(manifest_path,len(manifest))
