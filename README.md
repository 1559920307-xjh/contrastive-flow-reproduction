
```markdown
# Contrastive Flow Matching (ΔFM) on Imagenette

本项目基于 [DeltaFM](https://github.com/gstoica27/DeltaFM) 官方代码，针对 **Imagenette2-320** 数据集（10类，256×256）在 **SiT-S/2** 模型上复现和扩展了 **对比流匹配 (Contrastive Flow Matching)** 方法。  
原始论文：[Contrastive Flow Matching](https://arxiv.org/abs/2506.05350)

## 特性

- 支持 **纯 Flow Matching 基线** 和 **ΔFM 对比训练**，无需外部视觉编码器（如 DINOv2）
- 自动适配小模型，修复了维度错配、投影层冲突等问题
- 提供训练、评估脚本，支持多卡加速
- 评估时自动检测模型结构，无需手动配置参数

## 环境配置

推荐使用 Python 3.10+ 和 CUDA 11.8+。  
主要依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install accelerate omegaconf einops diffusers huggingface_hub torchmetrics pytorch-fid wandb
```

如果网络受限，可使用国内镜像加速 Hugging Face 下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

（训练和评估前均建议设置该环境变量）

## 数据准备

1. 下载 [imagenette2-320](https://github.com/fastai/imagenette) 数据集。
2. 解压后预处理（如需要），本项目使用 `CustomDataset`，会自动从预处理过的 VAE 缓存加载。
3. 确保数据目录结构为：
   ```
   /root/autodl-tmp/imagenette2-320-preprocessed/
   ├── train/
   │   ├── n01440764/
   │   └── ...
   └── val/
       ├── n01440764/
       └── ...
   ```
   （或者使用原 `imagenette2-320` 结构，数据集加载时会自动处理）

## 预训练 VAE 下载

训练需要 Stable Diffusion 的 VAE 模型，脚本会自动从 HuggingFace 下载（需联网）。  
如果下载失败，可提前手动缓存：

```bash
python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"
```

## 训练

### 1. 纯 Flow Matching 基线

```bash
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

accelerate launch train.py \
  --data-dir /path/to/imagenette2-320-preprocessed \
  --model SiT-S/2 \
  --resolution 256 \
  --num-classes 10 \
  --batch-size 64 \
  --loss-type mean \
  --proj-coeff 0 \
  --enc-type none \
  --output-dir ./baseline_experiment \
  --max-train-steps 60000 \
  --checkpointing-steps 5000 \
  --mixed-precision fp16 \
  --path-type linear \
  --prediction v \
  --weighting uniform \
  --contrastive-weight 0.0 \
  --is-class-conditioned
```

### 2. ΔFM 对比训练（λ=0.01，推荐）

```bash
accelerate launch train.py \
  --data-dir /path/to/imagenette2-320-preprocessed \
  --model SiT-S/2 \
  --resolution 256 \
  --num-classes 10 \
  --batch-size 64 \
  --loss-type contrastive \
  --proj-coeff 0 \
  --enc-type none \
  --output-dir ./contrastive_experiment \
  --max-train-steps 100000 \
  --checkpointing-steps 10000 \
  --mixed-precision fp16 \
  --path-type linear \
  --prediction v \
  --weighting uniform \
  --contrastive-weight 0.01 \
  --is-class-conditioned \
  --dont-contrast-on-unconditional
```

> 若需要继续训练，添加 `--resume` 即可自动加载最新检查点。

## 评估 FID

使用 `eval_FID_auto.py` 脚本，它会自动从检查点推断模型结构（`z_dims`、`decoder_hidden_size` 等）。

```bash
python eval_FID_auto.py \
  --checkpoint ./contrastive256_lamda0.01/sits2-vanilla-bs64-cccontrastiveTemp0p01-res256/checkpoints/0080000.pt \
  --output_dir ./fid_contrastive_lamda0.01_80k \
  --batch_size 20 \
  --num_samples 5000 \
  --cfg_scale 2.0

## 主要修改说明

我们对原始仓库进行了以下调整，使其能在**无外部编码器**的条件下运行，并支持 ΔFM 的纯 flow 对比：

- **`models/sit.py`**  
  - `decoder_hidden_size` 自动适配：当未指定时设为 `hidden_size`，避免维度错配  
  - 条件初始化投影层：仅当 `z_dims` 非空时才创建 `projectors`，否则为空列表  
  - `forward` 返回 `(x, zs, labels)`，供对比损失使用

- **`train.py`**  
  - 默认 `--enc-type` 设为 `'none'`，跳过 DINOv2 加载  
  - 编码器加载逻辑改为：当 `enc_type` 为空或 `'none'` 时，`z_dims=[]`，避免创建投影层  
  - 恢复检查点时自动过滤 `projectors.*` 键，使旧检查点与新结构兼容  
  - 日志记录使用 `.get()` 安全获取键，避免 `KeyError`

- **`loss.py`**  
  - 忽略模型返回的第三个值 `labels`  
  - 安全处理空 `zs` 列表，返回完整字典（含 `flow_loss` 和 `contrastive_loss`）

- **`triplet_loss.py`**（ΔFM 对比损失）  
  - 新增逻辑：当 `zs` 为空时，直接对 **flow 向量**（预测的 flow 与目标 flow）计算归一化余弦相似度损失，实现真正的“对比流”  
  - 兼容原有编码器特征对比方式  
  - 正确处理分类器无关令牌（CFG 丢弃标签时）的对比策略

- **`eval_FID_auto.py`**  
  - 自动从检查点提取 `z_dims` 和 `decoder_hidden_size`，无需手动指定  
  - 支持自适应模型结构加载

## 结果

| 模型        | 训练步数 | FID  | 备注                |
|-------------|----------|------|---------------------|
| 基线 (FM)   | 110k     | ~52  | SiT-S/2, bs=64     |
| ΔFM (λ=0.01)| 100k     | 待评估 | 纯 flow 对比，无投影 |

> 注：ΔFM 结果正在训练中，后续会更新。

## 致谢

代码基于 [DeltaFM](https://github.com/gstoica27/DeltaFM) 和 [SiT](https://github.com/willisma/SiT) 仓库开发，感谢作者的开源贡献。

## 引用

```bibtex
@article{stoica2025contrastive,
  title={Contrastive Flow Matching},
  author={Stoica, George and Ramanujan, Vivek and Fan, Xiang and Farhadi, Ali and Krishna, Ranjay and Hoffman, Judy},
  journal={arXiv preprint arXiv:2506.05350},
  year={2025}
}
```

保存后，即可将其与其他源代码文件一同上传到 GitHub。
