<div align="center">

# Mantis-Alpha 🦗

**高频端侧视觉-语言-动作（VLA）策略栈 · 独立 SmolVLA 实现**

*基于 SmolVLA / SmolVLM2 架构，自包含代码库，运行时不依赖 LeRobot。*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![Arch: SmolVLA](https://img.shields.io/badge/Architecture-SmolVLA_450M-purple.svg)](https://huggingface.co/papers/2506.01844)
[![Benchmark: LIBERO](https://img.shields.io/badge/Benchmark-LIBERO_40_Tasks-brightgreen.svg)](https://github.com/Lifelong-Robot-Learning/LIBERO)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/huggingface/lerobot)

</div>

---

## 概述 (Overview)

**Mantis-Alpha** 是一个面向边缘设备的高频具身操作策略栈。模型代码（SmolVLA 架构：
SigLIP 视觉编码器 + SmolVLM2 语言模型骨干 + 流匹配动作专家）已**完整内联**到本仓库
`src/mantis_alpha/` 中，训练、评测、推理服务均为自有实现，**任何运行路径都不导入
LeRobot**（可用 `grep -rnE "^\s*(import lerobot|from lerobot)" src/ scripts/` 验证）。

核心特性：

- **独立运行**：仅依赖 torch / transformers / pyarrow / pandas / pillow / safetensors 等基础库；
- **SmolVLA 架构**：~450M 参数（16 层 VLM 骨干冻结 + 99.9M 可训练动作专家），Flow-Matching 10 步 Euler 采样；
- **LeRobot v3 数据集兼容**：直接读取 parquet（内嵌 JPEG 图像）+ meta/info.json + stats.json，
  与 LeRobot 训练产出的 checkpoint（config.json + model.safetensors）双向兼容；
- **RTX 4090 24GB 可训**：256px 输入、batch 256 的完整训练配置已内置；
- **RTC 就绪**：Real-Time Chunking 推理支持已随模型代码内联，配置 `rtc_config` 即可启用。

## 架构 (Architecture)

```text
图像 (256×256×2) ──> SigLIP 视觉塔 ──> Pixel Shuffle ──> 图像 Tokens ─┐
                                                                      │  16 层 VLM 骨干 (冻结)
语言指令 ──────────> SmolVLM2 Tokenizer ──> 语言 Tokens ──────────────┤  (SmolVLM2-500M 前 16 层)
                                                                      │
机器人状态 (8 维) ──> State Proj ──> 状态 Token ──────────────────────┘
                                      │ prefix KV Cache
                                      ▼
                 ┌─────────────────────────────────┐
                 │  动作专家 (16 层, 720 hidden)    │  self-attn / cross-attn 交替
                 │  + Flow-Matching 时间条件        │
                 └─────────────────────────────────┘
                                      │ 10 步 Euler 积分
                                      ▼
                          动作块 (50 步 × 7 维)
```

## 仓库结构

```text
mantis_alpha/
├── scripts/
│   ├── train.py           # 独立训练脚本（自有训练循环）
│   ├── eval_libero.py     # 离线 action-MSE 评测
│   └── serve_policy.py    # FastAPI 推理服务
└── src/mantis_alpha/
    ├── modeling.py            # SmolVLA 策略 + VLAFlowMatching 模型（内联实现）
    ├── smolvlm_with_expert.py # VLM/动作专家双流注意力 wrapper（内联实现）
    ├── config.py              # SmolVLAConfig（含特征定义与反序列化）
    ├── policy_base.py         # 精简 PreTrainedPolicy（checkpoint 存取）
    ├── dataset.py             # LeRobot v3 数据集读取器
    ├── processor.py           # 分词 / 归一化 / batch 组装
    ├── flow_matching.py       # Euler 积分与时间采样
    ├── vla_utils.py           # 2D attention mask / RoPE 辅助 / pad / resize
    ├── utils.py               # 常量、依赖守卫、队列、dtype 工具
    ├── ensemble.py            # ACT 风格时序集成（推理平滑，可选）
    └── rtc/                   # Real-Time Chunking（配置 + 推理逻辑）
```

## 安装

```bash
conda activate lerobot    # 或任意含 torch>=2.5, transformers 5.4.x 的环境
pip install -e /home/adminroot/Desktop/vla/mantis_alpha --no-deps
```

## 训练

```bash
python scripts/train.py \
    --dataset_root /home/adminroot/Desktop/vla/datasets/libero \
    --vlm_model_name /home/adminroot/Desktop/vla/models/smolvlm2 \
    --batch_size 256 \
    --steps 40000 \
    --save_freq 4000 \
    --output_dir /home/adminroot/Desktop/vla/outputs/mantis_alpha

# 最小冒烟（单个 episode）
python scripts/train.py --episodes 0 --batch_size 2 --steps 3 --num_workers 0 \
    --output_dir /tmp/mantis_smoke
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch_size` | 256 | 4090 24GB 上 SmolVLA expert-only 的复现配方 |
| `--steps` | 40000 | cosine 衰减至 `--scheduler_decay_lr`，horizon 默认等于 `--steps`，warmup 500 步 |
| `--policy_path` | 无 | 热启动 checkpoint 目录（兼容 LeRobot 产出的 `pretrained_model/`） |
| `--no_vlm_weights` | 关 | 不加载 SmolVLM2 预训练权重（从零训练专家） |
| `--train_vlm` | 关 | 解冻 VLM 语言层 |
| `--episodes` | 全部 | 逗号分隔的 episode 下标，例如 `0` 或 `0,1,2` |

日志：控制台显示 tqdm 进度条（`Training:  92%|█████████▏| 18356/20000 [20:02:13<1:57:04, 4.27s/step]`），
完整记录（命令行、全参数 dump、每步 loss/lr/grad_norm、存档事件）追加写入
`<output_dir>/train.log`，另存结构化 `train_params.json` / `train_args.json`。

## 评测 / 推理

```bash
# 离线 action-MSE（归一化空间）
python scripts/eval_libero.py --checkpoint <ckpt_dir> --num_samples 64

# HTTP 服务
python scripts/serve_policy.py --checkpoint <ckpt_dir> --port 8000
# POST /act  {"task": "...", "image_main_b64": "...", "state": [...]}
```

## 与 LeRobot 的关系

模型实现遵循 Hugging Face SmolVLA（Apache 2.0）设计并内联于此；checkpoint
布局（`config.json` + `model.safetensors`）与 LeRobot 训练产出一致，可互相加载。
本仓库运行时不依赖 LeRobot 包。

## License

Apache 2.0（模型实现版权归 Hugging Face Inc.，见各文件头声明）。
