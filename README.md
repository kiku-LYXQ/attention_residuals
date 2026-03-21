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
.
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
├── attnres_hf_patch/    # 聚合 Hugging Face Llama AttnRes 模块
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
- 脚本 `train.py` 与 `evaluate.py` 均在仓库根目录，直接运行 `python train.py configs/...` 或 `python evaluate.py configs/...` 即可；训练脚本打印 loss & depth mean，评估脚本打印 hidden/grad stats，便于论文分析与日志归档。

## 设计与使用步骤
1. 在仓库根目录创建 conda/venv 环境并安装依赖：
   ```bash
   conda create -n attnres python=3.11 pip
   conda activate attnres
   pip install -r requirements.txt
   ```
2. 选择配置并训练：`python train.py configs/<mode>_toy.yaml`（如 `configs/full_attnres_toy.yaml`）。
3. 评估或获取分析指标：`python evaluate.py configs/full_attnres_toy.yaml` 输出 hidden norm、grad norm 与 depth mean。
4. 扩展时修改 `models/` 或 `configs/`，`tests/` 中添加新 case，运行 `pytest tests` 验证。
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
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 训练示例（baseline）：
   ```bash
   python train.py configs/baseline_toy.yaml
   ```
3. 全 residual 模式训练：
   ```bash
   python train.py configs/full_attnres_toy.yaml
   python train.py configs/block_attnres_toy.yaml
   ```
4. 评估（hidden norm / grad norm / depth attention）：
   ```bash
   python evaluate.py configs/full_attnres_toy.yaml
   ```

## 日志与监控
- `train.py` 每步打印 `loss` 与 depth attention mean，便于观察 Full/Block 模式的 depth 贡献。
- `evaluate.py` 会输出 hidden norm、grad norm 与 depth attention mean，配合论文中“缓解 PreNorm dilution”的趋势分析。

## 测试
```bash
pytest tests
```

## W&B Logging
1. 安装：`pip install -r requirements.txt`（其中已包含 `wandb>=0.15.0`）。
2. 如果要使用 online 模式，先执行 `wandb login` 并输入 API Key；offline/disabled 模式不需要联网。
3. 使用示例：
   ```bash
   python train.py configs/full_attnres_toy.yaml --model_type full_attnres_llama \
     --wandb_mode offline --wandb_project attention_residuals --wandb_run_name depth-stats
   ```
4. `--wandb_mode` 可选：
   - `disabled`：完全不初始化 W&B（默认），仅保留 console 日志；
   - `offline`：在本地 `wandb/` 目录记录，之后可用 `wandb sync wandb/offline-run-*/` 上传；
   - `online`：实时上传到 W&B，需要登录。
5. 已记录的指标包括 `train/loss`、`train/depth_mean`、`train/depth_entropy`、`train/depth_weight_count`、`train/depth_mean_layer_i`（每层聚合）、`train/depth_entropy_layer_i`、`train/depth_weights_hist`、`train/depth_entropy_hist`、`train/learning_rate`、`train/step`，底层 console 仍然输出 `loss` + `depth_mean`，保证现有监控不变。

## 扩展到更大模型
1. 将 `configs/xxx.yaml` 中的 `model` 部分替换为更大的 `embed_dim` / `num_layers`，并根据数据集切换 `dataset.text_file`。
2. 使用 `utils.metrics.hidden_norms` / `grad_norms` / `depth_attention_mean` 收集每层指标。
3. 训练脚本可直接替换为更大数据集：只需修改 `ToyTokenDataset` 中的 `text_file` 或 dataset loader。
4. 后续可在 Transformer 外层包装 Pipeline Parallel / Flash Attention 插件，只要保留 `DepthAttnResidual` 和 `BlockDepthAttnResidual` 的接口即可。

## WikiText-103 & paper16 准备流程
1. 运行 `scripts/download_wikitext103.py --limit N`（N 代表每个 split 保留的行数）从 Hugging Face datasets 里抓取 WikiText-103 的 train/validation，对原始行做简单清洗并写入 `data/wikitext103/train.txt`、`data/wikitext103/validation.txt`。
2. 使用 `scripts/prepare_text_dataset.py --tokenizer_name Qwen/Qwen2-0.5B --text_file data/wikitext103/<split>.txt --output_path cache/qwen_wikitext103_<split>_tokens.pt` 生成 1D token cache；文件名已包含 tokenizer/split 信息，可直接在 config 中引用。
3. `configs/` 下带 `_smoke` 或 `_formal_small` 的 YAML 继续服务于本地快速验证，`*_paper16.yaml` 则是真正的 16 层、CUDA、seq_len=512 配置，且统一指向 `cache/qwen_wikitext103_*_tokens.pt`。
4. 在正式训练前可先跑 `*_smoke.yaml` 配置验证 pipeline，再在 GPU 上运行 `*_paper16.yaml` 以完成论文级实验。

## OpenWebText2 大语料准备
1. 脚本 `scripts/download_openwebtext2.py` 使用 `datasets.load_dataset("openwebtext2")` 拉取 OpenWebText2，默认参数 `--limit 1000` 只会写入每个 split 的前 1,000 条；要下载完整语料（几十 GB）可以将 `--limit` 设为数据总大小或直接删掉 slicing，如 `python scripts/download_openwebtext2.py --limit 200000 --hf_token $HF_TOKEN`。该脚本会写入 `data/openwebtext2/train.txt` 和 `data/openwebtext2/validation.txt`，每行自动去换行符并分割为一句。
2. 仍然使用 `scripts/prepare_text_dataset.py --tokenizer_name Qwen/Qwen2-0.5B --text_file data/openwebtext2/<split>.txt --output_path cache/qwen_openwebtext2_<split>_tokens.pt` 来生成 cache；提示：如果下载大语料时需要 HF_TOKEN，请设置 `--hf_token` 或环境变量；本地测试可在 `--limit 2000` 继续验证 pipeline。
3. 生成的 `cache/qwen_openwebtext2_<split>_tokens.pt` 可直接在 `*_paper16.yaml` 配置里替换 `train_tokens_path`/`val_tokens_path`，train.py 会自动加载 tokenizer、对齐 vocab_size，并跑 validation。
4. `trainer` block 现在也支持 weight_decay / warmup_ratio / max_grad_norm 等大模型超参，请在配置中添加这些字段以开启 AdamW + cosine warmup + clipping 的标准流程。

## 已知简化与差异
- 目前仅提供 toy dataset，真实语料需要自行替换 `dataset.text_file` 或 custom DataLoader。
- 未实现论文中提到的 pipeline parallel / two-phase kernel 优化，仅在代码注释中预留扩展点。
- Block Attention Residual 中的 `block_size` 表示每 block 的 layer 数，不含跨分布式缓存。

## Hugging Face Llama AttnRes Patch
1. 新增子模块 `attnres_hf_patch/`，其中：
   - `attnres_state.py` 定义 `FullAttnResState`、`BlockAttnResState` 以及 `AttnResStats`（记录 `depth_mean`/`depth_entropy`/`attn_depth_weights`）;
   - `attnres_adapter.py` 复用 `models/attnres.py` 的聚合算子，并在 attention/MLP 前把深度 attention 权重交给 stats；
   - `modeling_llama_attnres.py` 复制并修改 Hugging Face `LlamaDecoderLayer` 和 `LlamaModel`，在 attention/MLP 前插入 depth-wise residual 聚合，并在 `AttnResLlamaModelBase`/`HfAttnResCausalLM` 中可选输出 stats；
   - `config.py` 提供 `HfAttnResMode` 枚举。
2. `train.py` 支持 `--model_type` 参数：
   - `baseline_llama`：原始 Llama；
   - `full_attnres_llama`：Full Attention Residuals（统计 `depth_mean`/`depth_entropy`）；
   - `block_attnres_llama`：Block Attention Residuals（同样输出 stats）；
   通过 CLI 传 `--model_type full_attnres_llama` 便可在 HF 版本上运行实验，训练日志自动打印 `depth_mean`（依赖 HF stats）并继续支持 `depth_attention_mean` 的多层合并。
3. 新增 `tests/test_hf_attnres.py` 验证 HF patch（shape、depth softmax、embedding 历史、block 状态、stat 统计、backward）。
