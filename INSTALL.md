# Install

## Clone

```
git clone https://github.com/jiaqixuac/MAP-Net.git
cd MAP-Net
```

## Install libraries (uv-based)

```
uv pip install torch torchvision av einops facexlib lmdb numpy opencv-python Pillow tensorboard tqdm --torch-backend=cu118
uv pip install openmim
mim install mmcv-full
uv pip install -r requirements/runtime.txt
uv pip install -e .
```

