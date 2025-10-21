<div align="center">

# 🎯 VSRD

### Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection

[![Python](https://img.shields.io/badge/Python-3.10-3670A0?style=for-the-badge&logo=Python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=%23EE4C2C)](https://pytorch.org/)
[![CVPR](https://img.shields.io/badge/CVPR-2024-4b44ce?style=for-the-badge)](https://arxiv.org/abs/2404.00149)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2404.00149-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2404.00149)

**[📄 Paper](https://arxiv.org/abs/2404.00149)** | **[🌐 Project Page](http://www.ok.sc.e.titech.ac.jp/res/VSRD/index.html)** | **[🎬 Demo Video](https://www.bilibili.com/video/BV1mD421p7k9/)** | **[📺 Bilibili](https://www.bilibili.com/video/BV1mD421p7k9/)**

---


https://github.com/skmhrk1209/VSRD/assets/29158616/fc64e7dd-2bb2-4719-b662-cb1e16ce764

<p align="center">
  <a href="http://www.ok.sc.e.titech.ac.jp/res/VSRD/index.html"><img src="https://img.shields.io/badge/Bilibili-Video-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white" alt="Bilibili"></a>
  <br>

</p>

</div>

---

## 📋 Table of Contents

- [✨ Highlights](#-highlights)
- [🚀 Installation](#-installation)
- [📦 Data Preparation](#-data-preparation)
- [🎓 Multi-View 3D Auto-Labeling](#-multi-view-3d-auto-labeling)
- [🏷️ Pseudo Label Preparation](#️-pseudo-label-preparation)
- [📄 License](#-license)
- [📖 Citation](#-citation)

---

## ✨ Highlights

> 🎯 **Weakly Supervised 3D Detection**: Learn 3D bounding boxes from only 2D instance masks
> 
> 🔮 **Volumetric Rendering**: Novel volumetric silhouette rendering approach
> 
> ⚡ **Efficient Auto-Labeling**: ~15 minutes per frame on V100
> 
> 🎨 **KITTI-360 Dataset**: State-of-the-art results on challenging autonomous driving scenes

---

## 🚀 Installation

<details open>
<summary><b>Quick Start</b></summary>

### 1️⃣ Setup the Conda Environment

```bash
conda env create -f environment.yaml
```

### 2️⃣ Install this Repository

```bash
pip install -e .
```

</details>

## 📦 Data Preparation

### 1️⃣ Download KITTI-360 Dataset

📥 Download the [KITTI-360 dataset](https://www.cvlibs.net/datasets/kitti-360/download.php).

**Required Data:**

| Component | Size | Description |
|-----------|------|-------------|
| 📸 Left perspective images | 124 GB | RGB camera images |
| 🎭 Left instance masks | 2.2 GB | 2D segmentation masks |
| 📦 3D bounding boxes | 420 MB | Ground truth boxes |
| 📷 Camera parameters | 28 KB | Intrinsic parameters |
| 📍 Camera poses | 28 MB | Extrinsic parameters |

**Directory Structure:**

```bash
KITTI-360
├── 📁 calibration         # camera parameters
├── 📁 data_2d_raw         # perspective images
├── 📁 data_2d_semantics   # instance masks
├── 📁 data_3d_bboxes      # 3D bounding boxes
└── 📁 data_poses          # camera poses
```

---

### 2️⃣ Generate Annotation Files

Create a JSON annotation file for each frame:

```bash
python tools/kitti_360/make_annotations.py \
    --root_dirname ROOT_DIRNAME \
    --num_workers NUM_WORKERS
```

✅ **Result:** A new `annotations` directory will be created:

```bash
KITTI-360
├── 📁 annotations         # per-frame annotations ✨
├── 📁 calibration         # camera parameters
├── 📁 data_2d_raw         # perspective images
├── 📁 data_2d_semantics   # instance masks
├── 📁 data_3d_bboxes      # 3D bounding boxes
└── 📁 data_poses          # camera poses
```

> ⚠️ **Note:** The following frames are excluded:
> - Frames without camera poses
> - Frames without instance masks

---

### 3️⃣ (Optional) Visualize Annotations

Verify the previous step completed successfully:

```bash
python tools/kitti_360/visualize_annotations.py \
    --root_dirname ROOT_DIRNAME \
    --out_dirname OUT_DIRNAME \
    --num_workers NUM_WORKERS
```

---

### 4️⃣ Sample Target & Source Frames

Sample frames for VSRD optimization:

```bash
python tools/kitti_360/sample_annotations.py \
    --root_dirname ROOT_DIRNAME \
    --num_workers NUM_WORKERS
```

✅ **Result:** A new `filenames` directory will be created:

```bash
KITTI-360
├── 📁 annotations         # per-frame annotations
├── 📁 calibration         # camera parameters
├── 📁 data_2d_raw         # perspective images
├── 📁 data_2d_semantics   # instance masks
├── 📁 data_3d_bboxes      # 3D bounding boxes
├── 📁 data_poses          # camera poses
└── 📁 filenames           # sampled filenames ✨
```

<details>
<summary><b>📖 Sampling Strategy Details</b></summary>

For efficiency, we use only selected frames as target frames for VSRD optimization:

1. 🔄 Frames with the same set of instance IDs are grouped
2. 🎯 Only one frame is sampled as a target frame for each instance group
3. 🏷️ Pseudo labels for each target frame are shared with all frames in the same instance group

> 💡 Please refer to the supplementary material for details on source frame sampling.

</details>

**Dataset Splits:**

| Sequence | Split | 🎯 Target Frames | 🏷️ Labeled Frames |
| :------- | :---- | ---------------: | -----------------: |
| `2013_05_28_drive_0000_sync` | 🔵 Training | 2,562 | 9,666 |
| `2013_05_28_drive_0002_sync` | 🔵 Training | 748 | 7,569 |
| `2013_05_28_drive_0003_sync` | 🟡 Validation | 32 | 238 |
| `2013_05_28_drive_0004_sync` | 🔵 Training | 658 | 5,608 |
| `2013_05_28_drive_0005_sync` | 🔵 Training | 408 | 4,103 |
| `2013_05_28_drive_0006_sync` | 🔵 Training | 745 | 6,982 |
| `2013_05_28_drive_0007_sync` | 🟡 Validation | 64 | 877 |
| `2013_05_28_drive_0009_sync` | 🔵 Training | 1,780 | 10,250 |
| `2013_05_28_drive_0010_sync` | 🟢 Test | 908 | 2,459 |

---

## 🎓 Multi-View 3D Auto-Labeling

VSRD optimizes **3D bounding boxes** and **residual signed distance fields (RDF)** for each target frame. The optimized 3D bounding boxes can be used as pseudo labels for training any 3D object detector.

### ⚡ Performance

> ⏱️ **Speed:** ~15 minutes per frame on V100 GPU

### 🚀 Distributed Training

Target frames are split and distributed across multiple processes for parallel processing. Each process operates independently on its assigned chunk.

> ⚠️ **Important:** Unlike typical distributed training, gradients are **not** averaged between processes.

Run [`main.py`](scripts/main.py) with the configuration file for your sequence:

<details>
<summary><b>🖥️ Using Slurm</b></summary>

Perfect for HPC clusters with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager:

```bash
python -m vsrd.distributed.slurm.launch \
    --partition PARTITION \
    --num_nodes NUM_NODES \
    --num_gpus NUM_GPUS \
    scripts/main.py \
        --launcher slurm \
        --config CONFIG \
        --train
```

</details>

<details>
<summary><b>🔥 Using Torchrun</b></summary>

Perfect for multi-node/multi-GPU training with [PyTorch Elastic](https://pytorch.org/docs/stable/elastic/run.html):

```bash
torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint HOST_NODE_ADDR \
    --nnodes NUM_NODES \
    --nproc_per_node NUM_GPUS \
    scripts/main.py \
        --launcher torchrun \
        --config CONFIG \
        --train
```

</details>

---

## 🏷️ Pseudo Label Preparation

### 1️⃣ Generate Pseudo Labels

Extract pseudo labels from trained checkpoints:

```bash
python tools/kitti_360/make_predictions.py \
    --root_dirname ROOT_DIRNAME \
    --ckpt_dirname CKPT_DIRNAME \
    --num_workers NUM_WORKERS
```

---

### 2️⃣ (Optional) Visualize Pseudo Labels

Verify the generated pseudo labels:

```bash
python tools/kitti_360/visualize_predictions.py \
    --root_dirname ROOT_DIRNAME \
    --ckpt_dirname CKPT_DIRNAME \
    --out_dirname OUT_DIRNAME \
    --num_workers NUM_WORKERS
```

---

### 3️⃣ Convert to KITTI Format

Convert pseudo labels to KITTI format for compatibility with existing frameworks like [MMDetection3D](https://github.com/open-mmlab/mmdetection3d):

```bash
python tools/kitti_360/convert_predictions.py \
    --root_dirname ROOT_DIRNAME \
    --ckpt_dirname CKPT_DIRNAME \
    --num_workers NUM_WORKERS
```

> 💡 **Tip:** This enables seamless integration with popular 3D detection training pipelines!

---

## 📄 License

VSRD is released under the [MIT License](LICENSE). Feel free to use it in your research and projects! 🎉

---

## 📖 Citation

If you find VSRD useful in your research, please consider citing our paper:

```bibtex
@article{liu2024vsrd,
    title={VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection},
    author={Liu, Zihua and Sakuma, Hiroki and Okutomi, Masatoshi},
    journal={arXiv preprint arXiv:2404.00149},
    year={2024}
}
```


