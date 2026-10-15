# Chạy LiDAR-only BEVFusion trên RTX 5060 (`sm_120`)

Đây là môi trường riêng cho GPU Blackwell. Không dùng `docker/Dockerfile` cũ:
nó bị khóa ở CUDA 11.3/PyTorch 1.10 và chỉ biên dịch CUDA extension đến `sm_86`.

## 1. Cho terminal quyền dùng Docker

Lệnh `usermod` đã thêm user vào group `docker`, nhưng group chỉ có hiệu lực sau
khi đăng nhập lại. Làm **một** trong hai cách dưới đây:

```bash
# Cách bền vững: logout rồi login lại desktop/SSH, sau đó mở terminal mới.
```

Hoặc chỉ cho terminal hiện tại:

```bash
newgrp docker
docker run --rm hello-world
```

Nếu vẫn báo lỗi socket sau khi login lại, nhờ IT kiểm tra Docker daemon/group;
không chạy mọi lệnh Docker bằng `sudo` lâu dài.

## 2. Build image Blackwell

Từ thư mục repository:

```bash
cd ~/bevfusion-adapter
# Cần có image base do team đã xác thực: bevfusion-blackwell:base.
docker build -f docker/Dockerfile.rtx5060-deps -t bevfusion-rtx5060-deps:latest .
docker build -f docker/Dockerfile.rtx5060 -t bevfusion:rtx5060 .
```

Image dependency dùng `mmcv-full 1.7.2` build source + `mmdet 2.28.2`, phù hợp
với API MMDetection 2.x của repository. Laptop 8 GB VRAM là đủ cho
**single-frame inference**, nhưng compile sẽ tốn RAM CPU và có thể mất nhiều
phút.

## 3. Xác thực CUDA trong container

```bash
docker run --rm --gpus all bevfusion:rtx5060 \
  python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list()); assert 'sm_120' in torch.cuda.get_arch_list()"
```

Kết quả phải có CUDA `12.8` và `sm_120`.

## 4. Chạy smoke inference ba PCD frames

```bash
cd ~/bevfusion-adapter
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD:/workspace/bevfusion-adapter" \
  -w /workspace/bevfusion-adapter \
  bevfusion:rtx5060 \
  python tools/infer_pcd.py \
    configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
    checkpoint/lidar-only-det.pth \
    --pcd-dir data/20260916_134800_lidar \
    --out-dir runs/pcd-pretrained-smoke \
    --score-threshold 0.2 \
    --max-frames 3
```

Output nằm trên máy host tại `runs/pcd-pretrained-smoke/`: `summary.json`,
`predictions/*.json` và `bev/*.png`.

## Nếu build fail

Đây là codebase OpenMMLab 2022. Phần dễ vỡ nhất là `mmcv-full==1.4.0` và custom
C++/CUDA extensions khi compile cùng PyTorch 2.7. Nếu lỗi, lưu **toàn bộ log từ
dòng `ERROR` đầu tiên trở lên** rồi gửi lại; đừng hạ CUDA/PyTorch xuống, vì như
vậy sẽ mất `sm_120`. `setup.py` đã nhận `TORCH_CUDA_ARCH_LIST=12.0` để sinh
kernel đúng kiến trúc GPU.
