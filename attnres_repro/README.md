# Attention Residuals Reproduction

此项目复现论文《Attention Residuals》（arXiv:2603.15031）中提出的 residual 机制，用以探索 PreNorm transformer 中传统 residual 的稀释问题，并提供一个可扩展的 PyTorch 工程框架。

## 论文核心思想
- **标准 Residual（baseline）**：使用传统残差连接：`x = x + attn(norm(x))` 与 `x = x + mlp(norm(x))`，存在 `PreNorm dilution`（残差信息被稀释）现象。
- **Full Attention Residuals**：使用 depth-wise attention（attention 在 depth 维度做 softmax）对历史表达进行加权，而不是简单相加，embedding 与每层状态都参与聚合，key 侧使用 RMSNorm 统一归一化。
- **Block Attention Residuals**：将 history 拆成 `blocks`（已完成 block 表示）+ `partial_block`（当前 block 累积），分别在 attention/MLP 前对两个列表做 depth attention，paper Figure 2 中给出具体流程。

## 设计模块差异
| 模式 | 残差形式 | 聚合维度 | 结构说明 |
|------|---------|---------|---------|
| baseline | 直接加法 residual | 无 | 经典 PreNorm Transformer Block |
| full_attnres | Depth attention over full history | depth | 使用 RMSNorm + learnable scoring projection 聚合 embedding + past states |
| block_attnres | Block + partial depth attention | source | 模拟论文 Figure 2，maintains `blocks` 与 `partial_block`，在 attention 与 mlp 前各做一次 depth attention |

## 工程结构
```
attnres_repro/
├── configs/            # 所有训练配置（baseline/full/block toy）
├── models/             # 模型层：norm, layers, attention residuals, transformer variants
│   ├── norms.py        # RMSNorm
│   ├── layers.py       # MHA + MLP + BasePreNormBlock
│   └── attnres.py      # DepthAttnResidual / BlockDepthAttnResidual + BlockState
│   └── transformer.py  # Baseline / Full / Block transformers + shared BaseTransformer
├── utils/              # 通用工具（logging, seed, metrics）
│   ├── seed.py         # 随机数/设备种子设置
│   ├── logging_utils.py# 简单的 logging 配置
│   └── metrics.py      # hidden norm、grad norm、depth attention mean
├── tests/              # PyTorch 单元测试（shape/attention/block checks）
├── train.py            # Toy 训练脚本（baseline/full/block），打印 loss + depth attention mean
├── evaluate.py         # 评估脚本，输出 hidden/grad norm 与 depth attention mean
├── README.md           # 项目说明 + 运行/测试指南
└── requirements.txt    # pip/install 依赖（torch>=2.0, numpy, pyyaml, pytest）
```

## 目录用途与协作指南
- `models/` 负责论文核心业务逻辑：norm、attention、残差聚合以及 transformer 变体。后续可在这里引入更大 decoder block、Flash Attention 或其他注意力替代。
- `utils/` 提供可重复使用的日志/seed/metric 工具，供训练与评估脚本引用；可拓展入 `wandb`/`tensorboard` hook。
- `configs/` 以 YAML 描述各模式训练参数；想跑新实验时只需复制模板并修改 `model` 部分、`trainer` 的 `steps`/`batch_size` 或 `dataset.text_file`。
- `tests/` 保持轻量 smoke tests，验证 shape、attention distribution、block 状态；可扩充到 `torch.jit` 兼容或更多模块测试。
- 脚本 `train.py` 与 `evaluate.py` 均可通过 `python -m attnres_repro.train configs/...` 方式调用；训练脚本打印 loss & depth mean，评估脚本打印 hidden/grad stats，便于论文分析与日志归档。

## 设计与使用步骤
1. 在 `attnres_repro` 根目录创建 conda/venv 环境并安装依赖：
   ```bash
   conda create -n attnres python=3.11 pip
   conda activate attnres
   pip install -r attnres_repro/requirements.txt
   ```
2. 选择配置并训练：`python -m attnres_repro.train attnres_repro/configs/<mode>_toy.yaml`，例如 `full_attnres_toy.yaml`。
3. 评估或获取分析指标：`python -m attnres_repro.evaluate attnres_repro/configs/full_attnres_toy.yaml` 输出 hidden norm、grad norm 与 depth mean。
4. 扩展时修改 `models/` 或 `configs/`，`tests/` 中添加新 case，运行 `pytest attnres_repro/tests` 验证。
5. 推送前常规 git: `git status -sb`, `git diff`, `git commit`, `git push origin master`。

### 核心模块说明
- `models/norms.py`：RMSNorm
- `models/layers.py`：MultiHeadSelfAttention、MLP、BasePreNormBlock
- `models/attnres.py`：`DepthAttnResidual`（Full）与 `BlockDepthAttnResidual`（Block）以及 `BlockState`
- `models/transformer.py`：`BaselineTransformer` / `FullAttnResTransformer` / `BlockAttnResTransformer`，并在 stats 中记录 depth attention 权重与 hidden state
- `utils/metrics.py`：hidden norm / grad norm / depth attention mean 等分析工具

## 运行方式
1. 进入项目根目录并创建 Python 虚拟环境：
   ```bash
   cd attnres_repro
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 训练示例（baseline）：
   ```bash
   cd ..  # 或在项目根目录运行，确保 Python path 包含 attnres_repro
   python -m attnres_repro.train attnres_repro/configs/baseline_toy.yaml
   ```
3. 全 residual 模式训练：
   ```bash
   python -m attnres_repro.train attnres_repro/configs/full_attnres_toy.yaml
   python -m attnres_repro.train attnres_repro/configs/block_attnres_toy.yaml
   ```
4. 评估（hidden norm / grad norm / depth attention）：
   ```bash
   python -m attnres_repro.evaluate attnres_repro/configs/full_attnres_toy.yaml
   ```

## 日志与监控
- `train.py` 每步打印 `loss` 与 depth attention mean，便于观察 Full/Block 模式的 depth 贡献。
- `evaluate.py` 会输出 hidden norm、grad norm 与 depth attention mean，配合论文中“缓解 PreNorm dilution”的趋势分析。

## 测试
```bash
pytest tests
```

## 扩展到更大模型
1. 将 `configs/xxx.yaml` 中的 `model` 部分替换为更大的 `embed_dim` / `num_layers`，并根据数据集切换 `dataset.text_file`。
2. 使用 `utils.metrics.hidden_norms` / `grad_norms` / `depth_attention_mean` 收集每层指标。
3. 训练脚本可直接替换为更大数据集：只需修改 `ToyTokenDataset` 中的 `text_file` 或 dataset loader。
4. 后续可在 Transformer 外层包装 Pipeline Parallel / Flash Attention 插件，只要保留 `DepthAttnResidual` 和 `BlockDepthAttnResidual` 的接口即可。

## 已知简化与差异
- 目前仅提供 toy dataset，真实语料需要自行替换 `dataset.text_file` 或 custom DataLoader。
- 未实现论文中提到的 pipeline parallel / two-phase kernel 优化，仅在代码注释中预留扩展点。
- Block Attention Residual 中的 `block_size` 表示每 block 的 layer 数，不含跨分布式缓存。
