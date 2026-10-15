# Smoke inference LiDAR PCD với checkpoint BEVFusion nuScenes

Công cụ `tools/infer_pcd.py` chạy checkpoint LiDAR-only nuScenes trực tiếp trên
các frame PCD binary chưa có label. Đây là kiểm tra inference single-frame,
không phải đánh giá accuracy và không cần dựng bộ dữ liệu nuScenes giả.

## Input contract

- PCD `DATA binary`, các field scalar `x y z intensity` kiểu float32.
- Model nhận `[x, y, z, intensity, sweep_time_lag]`; công cụ tự thêm time lag 0.
- Hệ trục phải là LiDAR/ego kiểu nuScenes: x trước, y trái, z lên, đơn vị mét.
- Checkpoint phải đi cùng config 10-class gốc
  `configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml`.

Nếu PCD chưa ở đúng hệ trục, truyền ma trận 4x4 row-major bằng `--transform`.
Phép đổi được áp dụng theo `xyz_out = R @ xyz_in + t`.

## Kiểm tra môi trường GPU

Repository dùng legacy stack trong README/Dockerfile: Python 3.8, PyTorch
1.10.1 + CUDA 11.3, MMCV/MMCV-full 1.4.0, MMDetection 2.20.0, torchpack và
nuscenes-devkit. Sau khi cài và compile CUDA extensions:

```bash
python setup.py develop
python -c "import torch, mmcv, mmdet, mmdet3d; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

## Chạy ba frame trước

Chạy bằng Python thường; công cụ này không dùng distributed runner:

```bash
python tools/infer_pcd.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  checkpoint/lidar-only-det.pth \
  --pcd-dir data/20260916_134800_lidar \
  --out-dir runs/pcd-pretrained-smoke \
  --score-threshold 0.2 \
  --max-frames 3
```

Điều kiện qua:

- checkpoint nạp `strict=True` không báo missing/unexpected/shape mismatch;
- cả ba frame chạy xong không CUDA error hoặc NaN;
- mỗi frame có JSON trong `predictions/` và PNG trong `bev/`;
- `summary.json` ghi số point, detection và thời gian inference.

Sau đó chạy toàn bộ 204 frame:

```bash
python tools/infer_pcd.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  checkpoint/lidar-only-det.pth \
  --pcd-dir data/20260916_134800_lidar \
  --out-dir runs/pcd-pretrained-all \
  --score-threshold 0.2
```

Thêm `--no-bev` nếu chỉ muốn JSON. Dùng `--stride 10` để lấy mỗi frame thứ
10. Không ghép các frame thành multisweep nếu chưa có ego pose để motion
compensation.

## Diễn giải kết quả

Class output theo đúng thứ tự nuScenes:

`car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone`.

Box JSON có thứ tự `[x, y, z_bottom, length, width, height, yaw, vx, vy]` theo
quy ước nội bộ của checkpoint. Dữ liệu mẫu hiện có intensity bằng 0 toàn bộ;
điều này không chặn smoke inference nhưng prediction có domain shift lớn và
không nên dùng để kết luận chất lượng model.
