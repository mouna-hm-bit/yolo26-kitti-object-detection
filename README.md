# yolo26-kitti-object-detection

Object detection on the **KITTI** dataset using transfer learning with **YOLO26n** (`ultralytics`).

## Repository Contents

- `kiti-model-ipynb.ipynb` — Kaggle notebook covering Ultralytics installation, COCO-pretrained `yolo26n.pt` transfer learning on KITTI, validation metrics, and loading the trained `best.pt` checkpoint.

## Dataset & Classes

Trained on Ultralytics' `kitti.yaml` split (**5,985** training images, **1,496** validation images) across **8 classes**:

`car` · `van` · `truck` · `pedestrian` · `person_sitting` · `cyclist` · `tram` · `misc`

## Training Setup

- **Base model:** `yolo26n.pt` (COCO pretrained)
- **Framework:** `ultralytics 8.4.202`, PyTorch `2.11.0+cu130`
- **Hardware:** 2× NVIDIA Tesla T4 GPUs (`device="0,1"`, DDP + `amp=True`)
- **Hyperparameters:** `epochs=10`, `imgsz=640`, `batch=128`, `patience=20`, `workers=8`

## Validation Results (`best.pt`)

Evaluated on 1,496 validation images (8,128 instances):

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all** | **1496** | **8128** | **0.779** | **0.618** | **0.698** | **0.426** |
| car | 1322 | 5716 | 0.879 | 0.831 | 0.909 | 0.655 |
| van | 419 | 555 | 0.823 | 0.655 | 0.762 | 0.517 |
| truck | 212 | 223 | 0.907 | 0.788 | 0.848 | 0.627 |
| pedestrian | 377 | 921 | 0.715 | 0.528 | 0.609 | 0.273 |
| person_sitting | 17 | 36 | 0.645 | 0.505 | 0.554 | 0.249 |
| cyclist | 252 | 352 | 0.720 | 0.474 | 0.566 | 0.267 |
| tram | 77 | 117 | 0.704 | 0.726 | 0.800 | 0.469 |
| misc | 158 | 208 | 0.835 | 0.433 | 0.539 | 0.350 |

- **Speed per image:** `0.0ms` preprocess, `1.4ms` inference, `0.7ms` postprocess.

## Usage

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.train(
    data="kitti.yaml",
    epochs=10,
    imgsz=640,
    batch=128,
    device="0,1",
    amp=True,
    patience=20,
    workers=8,
    project="KITTI_Project",
    name="YOLO26n_Transfer_Learning",
)
```
