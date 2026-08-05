# Code for ICML 2026 paper "DiLA: Disentangled Latent Action World Models"

<p align="center">
  <a href="https://disentangled-latent-action-world-models.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.15725"><img src="https://img.shields.io/badge/arXiv-2605.15725-b31b1b" alt="arXiv"></a>
  <a href="https://huggingface.co/senngadaisuki/disentangled-latent-action-world-models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Hugging%20Face-yellow" alt="Hugging Face Checkpoint"></a>
  <a href="https://huggingface.co/datasets/senngadaisuki/omni-primitive-transforms"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow" alt="Hugging Face Dataset"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-green" alt="Citation"></a>
</p>

<p align="center">
  <a href="https://ztqakita.github.io/">Tianqiu Zhang</a><sup>*</sup> &nbsp;·&nbsp;
  <a href="https://senngadaisuki.github.io/">Muyang Lyu</a><sup>*</sup> &nbsp;·&nbsp;
  Yufan Zhang &nbsp;·&nbsp;
  Fang Fang &nbsp;·&nbsp;
  <a href="https://www.psy.pku.edu.cn/english/people/faculty/professor/wusi/index.htm">Si Wu</a>
</p>
<p align="center">Peking University</p>
<p align="center"><sup>*</sup> Equal contribution</p>

<p align="center">
  <img src="https://disentangled-latent-action-world-models.github.io/assets/model.png" width="95%" alt="DiLA architecture">
</p>
<p align="center">
  <em>DiLA decouples video features into a structure pathway for latent action dynamics
  and a content pathway for appearance-preserving world-model prediction.</em>
</p>

---

## Overview

DiLA is a **Disentangled Latent Action world model** for learning from
unlabeled videos. Latent Action Models infer action-like variables between
frames, but they often trade off abstract, transferable actions against
high-fidelity prediction. DiLA addresses this tension through
**content-structure disentanglement**.

- **Structure pathway** learns dynamics-relevant spatial layouts and abstract latent actions.
- **Content pathway** stores appearance, texture, and slowly revealed scene details with a [Mamba](https://github.com/state-spaces/mamba) memory.
- **Fusion decoder** recombines predicted structure, content memory, and initial-frame features to predict future [DINOv2](https://github.com/facebookresearch/dinov2) embeddings.
- **Latent rollouts** are performed autoregressively in structure space for action transfer, manifold analysis, and visual planning.

Video frames are encoded with frozen [DINOv2](https://huggingface.co/facebook/dinov2-with-registers-base) features. DiLA is trained
self-supervised from observation sequences, without ground-truth action labels.

> See the [paper](https://arxiv.org/abs/2605.15725) and the
> [project page](https://disentangled-latent-action-world-models.github.io/)
> for the full method, visualizations, and results.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/senngadaisuki/disentangled-latent-action-world-models
cd disentangled-latent-action-world-models
```

### 2. Set up the environment

The recommended way is to create a Conda environment from the provided
[`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate dila
```

The environment file includes the dependencies needed for training,
VP<sup>2</sup> adaptation, and the inference notebook. It installs PyTorch with
CUDA 12.1 by default; if your CUDA version is different, edit the
`pytorch-cuda` line in `environment.yml` before creating the environment.
+
### 3. Prepare pretrained RAE

For training, DiLA uses the [DINOv2 encoder](https://huggingface.co/facebook/dinov2-with-registers-base)
from [RAE](https://github.com/bytetriper/RAE). It is downloaded automatically
on first use into `RAE.encoder_cache_dir: ./pretrained`. To pre-download it for
offline runs, use the same cache directory:

```bash
mkdir -p pretrained
hf download facebook/dinov2-with-registers-base \
  --cache-dir ./pretrained
```

Training also needs the RAE latent normalization statistics:

```bash
hf download nyu-visionx/RAE-collections \
  stats/dinov2/wReg_base/imagenet1k/stat.pt \
  --local-dir ./pretrained
```

For testing and qualitative visualization, download the RAE decoder used to
reconstruct generated latent embeddings back into image space:

```bash
hf download nyu-visionx/RAE-collections \
  decoders/dinov2/wReg_base/ViTXL/dinov2_decoder.pt \
  --local-dir ./pretrained
```

After downloading, the expected local files are:

```text
pretrained/models--facebook--dinov2-with-registers-base/
pretrained/stats/dinov2/wReg_base/imagenet1k/stat.pt
pretrained/decoders/dinov2/wReg_base/ViTXL/dinov2_decoder.pt
```

The decoder architecture config is included at
[`configs/decoder/ViTXL/config.json`](configs/decoder/ViTXL/config.json). If you
use a different RAE variant, update `decoder_config_path`,
`pretrained_decoder_path`, and `normalization_stat_path` in
[`configs/train_model.yaml`](configs/train_model.yaml).

---

## Data Preparation

The default training config uses a [Hydra](https://hydra.cc/) multi-dataset
loader. Following the paper, training uses
[Something-Something-V2 (SSv2)](https://www.qualcomm.com/developer/software/something-something-v-2-dataset),
[RT-1](https://robotics-transformer1.github.io/),
[RECON](https://rail.eecs.berkeley.edu/datasets/recon-navigation/), and
[LoopNav](https://arxiv.org/abs/2505.22976). For faster PyTorch dataloading,
this repository expects the video datasets to be preprocessed into per-frame
image folders.
Dataset paths and sampling ratios are configured in
[`configs/data/multidataset.yaml`](configs/data/multidataset.yaml).

### Something-Something-V2 (SSv2)

[SSv2](https://arxiv.org/abs/1706.04261) is a large-scale human-object
interaction video dataset. Download it following the
[official instructions](https://www.qualcomm.com/developer/software/something-something-v-2-dataset),
extract frames from each video, and place the processed raw frames and metadata
at:

```text
data/ssv2/rawframes/
  {video_id}/
    img_00001.jpg
    img_00002.jpg
    ...
data/ssv2/labels/
  train.json
  validation.json
  test.json
```

For training, we use a filtered SSv2 subset that removes static clips and
strong camera-motion clips to emphasize clear physical interactions.
The filtered metadata files should be placed in the same `labels/` directory,
for example `training-clean.json` and `validation-clean.json`.

### RT-1

[RT-1](https://robotics-transformer1.github.io/) is used for robot
manipulation videos. In the released configs, this dataset is the
`fractal20220817_data` subset from the Open X-Embodiment/RT-X collection.
Convert the RLDS/TFDS episodes into processed frame folders before training.
The default config expects:

```text
data/rtx/
  fractal20220817_data_videos/
    fractal20220817_data.txt      # optional instructions
    video_0/
      frame_0.png
      frame_1.png
      ...
    video_1/
      frame_0.png
      ...
```

By default, the enabled RT-1 subset is `fractal20220817_data`. The config key
is still named `openx` because the shared loader can read processed
Open X-Embodiment-style frame folders. Update
`data.multidataset.openx.dataset_names` in
[`configs/data/multidataset.yaml`](configs/data/multidataset.yaml) if you use
a different processed RT-1 folder. The loader searches one directory level
under `data/rtx/`, so placing `fractal20220817_data_videos/` directly under
`data/rtx/` also works.

### RECON

[RECON](https://rail.eecs.berkeley.edu/datasets/recon-navigation/) is used for
outdoor ground navigation videos with continuous robot motion. In the config,
`nwm` refers to the dataset     adapted from
[Navigation World Models](https://www.cs.cmu.edu/~aarnab/nwm/), which we use to
read RECON (`dataset_name: recon`); we do not use all datasets from the NWM
work.
Prepare the RECON data in the format expected by this loader:

```text
data/recon/
  data/
    {trajectory_name}/
      traj_data.pkl
      0.jpg
      1.jpg
      ...
  data_splits/
    train/
      traj_names.txt
    test/
      traj_names.txt
```

### LoopNav

[LoopNav](https://arxiv.org/abs/2505.22976) is used for Minecraft navigation videos with discrete step-by-step controls. Due to the extensive size of the original dataset, we utilize a subset for our experiments:
* **Villages:** Out of the 6 village types (20 instances each), we only use the first 2 instances per type (`xxxx_1`, `xxxx_2`).
* **Biomes & Locations:** From the 26 available environments (18 biomes + 8 locations), we sample 4 specific areas: forest, river, stony shore, and end city.

The loader recursively scans for run folders that contain `meta.json` and numbered frame images:

```text
data/loopnav/frames/
  ABA/                         # pattern
    desert_village_1/          # scene
      15/                      # path length
        05-04_17-55-03/        # run id
          meta.json
          00000.jpg
          00001.jpg
          ...

To train with the full dataset mixture from the paper, enable all four datasets
and set matching sampling ratios in
[`configs/data/multidataset.yaml`](configs/data/multidataset.yaml), for example:

```yaml
multidataset:
  dataset_names: [ssv2, openx, nwm, loopnav]
  ratios: [1, 1, 1, 1]
```

Evaluation-only datasets and auxiliary benchmarks are described in the
[Evaluation](#evaluation) section.

---

## Training

Training uses [Accelerate](https://github.com/huggingface/accelerate)
(multi-GPU / mixed precision) and [Hydra](https://hydra.cc/) configs.
Following the paper, DiLA is first trained from scratch and then finetuned with
latent rollouts to reduce autoregressive error accumulation.

```bash
# One-time accelerate setup (multi-GPU / mixed precision)
accelerate config
```

```bash
# Training
accelerate launch main.py phase=1 \
  num_train_steps=30000 \
  work_dir=./checkpoints \
  batch_size=8 \
  exp_name=DiLA-training

# Latent-rollout finetuning
accelerate launch main.py phase=2 \
  model.phase1_ckpt=./checkpoints/phase1/DiLA-training.pt \
  num_train_steps=1000 \
  work_dir=./checkpoints \
  exp_name=DiLA-finetune
```

Useful Hydra overrides:

```bash
accelerate launch main.py batch_size=8 seq_len=16 work_dir=./checkpoints
```

`batch_size` is the per-GPU batch size. The effective global batch size is
`batch_size * num_gpus * grad_accum_every`; for example, `batch_size=8` on four
GPUs with `grad_accum_every=1` gives a global batch size of 32 sequences.

See [`configs/train_model.yaml`](configs/train_model.yaml)
for the full training config. Checkpoint filenames are based on `exp_name` and
are saved under `work_dir/phase{N}/`. With `exp_name=DiLA-training`, training
saves:

```text
checkpoints/phase1/DiLA-training.pt
checkpoints/phase1/DiLA-training_milestone_5000.pt
checkpoints/phase1/DiLA-training_final.pt
```

The main model and optimization parameters are configured in:

- [`configs/train_model.yaml`](configs/train_model.yaml)
- [`configs/world_model/inverse_world_model.yaml`](configs/world_model/inverse_world_model.yaml)
- [`configs/structure_encoder/st_transformer.yaml`](configs/structure_encoder/st_transformer.yaml)
- [`configs/content_fusion/separate_fusion.yaml`](configs/content_fusion/separate_fusion.yaml)
- [`configs/optimizer/optimizer.yaml`](configs/optimizer/optimizer.yaml)

---

## Evaluation

### Checkpoints

Pretrained weights: https://huggingface.co/senngadaisuki/disentangled-latent-action-world-models

```bash
huggingface-cli download senngadaisuki/disentangled-latent-action-world-models \
  --local-dir ./checkpoints
```

### Notebook

For interactive visualization and evaluation of generation, action transfer and rebinding, run **[`test.ipynb`](test.ipynb)**.

### Omni-Primitive-Transforms (our evaluation benchmark for latent action analysis)

[Omni-Primitive-Transforms](https://huggingface.co/datasets/senngadaisuki/omni-primitive-transforms)
is a 3D object primitive-transformation dataset used to probe structural
generalization. It contains 3D object sequences under controlled single or
compositional primitive transformations, such as rotation, translation, and
scaling, rendered from high-quality scanned meshes from
[OmniObject3D](https://omniobject3d.github.io/) with Blender.

The Hugging Face dataset repository provides both the rendered data used in our
experiments and the rendering/generation code for creating the dataset. It is
**optional and used for evaluation only**; it is not required for DiLA training.

```bash
huggingface-cli download senngadaisuki/omni-primitive-transforms \
  --repo-type dataset --local-dir dataset/omni-primitive-transforms
```

### Visual planning on VP<sup>2</sup>

We evaluate visual planning on the
[VP<sup>2</sup> benchmark](https://github.com/s-tian/vp2), following the
action-adaptation and planning protocol used by
[AdaWorld](https://github.com/Little-Podi/AdaWorld/blob/main/docs/PLANNING.md).
In this setting, DiLA is adapted from a latent-action world model into an
action-conditioned dynamics model for model-predictive control.

The adaptation has three steps:

1. **Initialize an action-to-latent MLP.** For each VP<sup>2</sup>
   environment, sample 100 trajectories and train a lightweight MLP that maps
   ground-truth robot actions to DiLA latent actions. In this repository, the
   initialization script is [`fast_init_mlp.py`](fast_init_mlp.py).
2. **Fine-tune DiLA with action conditioning.** Replace the inverse dynamics
   latent-action inference with the initialized action MLP, then fine-tune DiLA
   on environment-specific RoboSuite or RoboDesk trajectories.
3. **Use DiLA as the VP<sup>2</sup> world model.** The fine-tuned model predicts
   candidate futures inside the VP<sup>2</sup> MPC/MPPI planner. We use the same
   VP<sup>2</sup> cost functions and planning protocol as described in the paper.

Example fine-tuning command for RoboSuite:

```bash
accelerate launch main.py \
  phase=3 \
  data=robotask \
  model.phase2_ckpt=/path/to/phase2_dila.pt \
  model.action_decoder_ckpt=/path/to/robosuite/mlp_init_weights.pth \
  data.root_dir=/path/to/robosuite_frames \
  num_train_steps=1000 \
  work_dir=./checkpoints \
  exp_name=DiLA-vp2-robosuite-finetune
```

For RoboDesk, use the RoboDesk frames and set the action dimensionality to 5:

```bash
accelerate launch main.py \
  phase=3 \
  data=robotask \
  model.phase2_ckpt=/path/to/phase2_dila.pt \
  model.action_decoder_ckpt=/path/to/robodesk/mlp_init_weights.pth \
  data.root_dir=/path/to/robodesk_frames \
  decoder.num_actions=5 \
  num_train_steps=1000 \
  work_dir=./checkpoints \
  exp_name=DiLA-vp2-robodesk-finetune
```

We can now employ the fine-tuned model for VP<sup>2</sup> planning. To begin, configure the following paths:

1. **RAE initialization checkpoint**

   Set the RAE initialization path in:

   [`DiLA_for_vp2/vp2/vp2/models/dila.py`](DiLA_for_vp2/vp2/vp2/models/dila.py)

2. **World model checkpoint**

   Replace `/path/to/world_model_checkpoint.pth` with the path to the fine-tuned
   DiLA world model checkpoint in:

   [`DiLA_for_vp2/vp2/vp2/scripts/configs/model/dila.yaml`](DiLA_for_vp2/vp2/vp2/scripts/configs/model/dila.yaml)

3. **Goal dataset**

   Configure the path to the corresponding goal dataset for both RoboDesk and
   RoboSuite in the environment configuration files under:

   [`DiLA_for_vp2/vp2/vp2/scripts/configs/env`](DiLA_for_vp2/vp2/vp2/scripts/configs/env)

After configuring these paths, run planning with:

```bash
bash DiLA_for_vp2/vp2/vp2/run_plan.sh
```

See Appendix D of the paper for the full protocol, including the number of
adaptation trajectories, fine-tuning data, and MPC/MPPI planning settings.

---

## Citation

```bibtex
@inproceedings{zhang2026dila,
  title     = {{DiLA}: Disentangled Latent Action World Models},
  author    = {Zhang, Tianqiu and Lyu, Muyang and Zhang, Yufan and Fang, Fang and Wu, Si},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=BRBHruBDkb}
}
```

---

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for
details.

Third-party datasets, pretrained encoders/decoders, and external benchmark
assets are subject to their own licenses.
