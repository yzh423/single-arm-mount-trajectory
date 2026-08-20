# Episode 3D 轨迹查看器

双击 `index.html`，或在项目根目录运行：

```powershell
python -m http.server 8000 --directory tools/episode-3d-viewer
```

然后打开 `http://localhost:8000`。查看器依赖已放在 `vendor/`，可以完全离线使用；CSV 内容始终只在浏览器本地解析，不会上传。

## 使用方式

1. 将一个或多个 episode CSV 拖入虚线区域，或点击“选择 CSV”。
2. 在图中按住鼠标左键旋转，右键平移，滚轮缩放。
3. 点击图例可隐藏/显示单条轨迹；点击轨迹查看时间和 XYZ 坐标。
4. 点击“重置视角”恢复自动相机。

查看器自动识别以下 XYZ 字段组：

- `right_controller_pos_x/y/z`
- `right_tcp_pos_x/y/z`
- `left_controller_pos_x/y/z`
- `left_tcp_pos_x/y/z`

缺失一整组 XYZ、数值无效或没有可用轨迹时，左侧会显示错误信息。超过 12,000 个有效点的单条轨迹会仅在显示时均匀降采样，CSV 原文件不会改变。
