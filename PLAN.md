# 中文 AI 文本检测器（Laya 式）—— 分阶段计划

> 目标：训一个小编码器 + 分类头（0.1–0.6B），对中文文本（新闻/社交/问答）判「人写 vs AI 写」，
> 单次前向出**校准过的概率**。后续可选：把检测器当奖励，驱动 Qwen 改写以降 AI 味。
> 最后更新：2026-09-29

## 0. 定位与边界（别自欺）

- 这是**分类任务**，和 Laya 的 `noul`/`choice` 同源；不是生成模型。
- 「去 AI 味」是**生成/改写**任务，编码器做不了 → 只能「检测器当打分器 + Qwen 改写」组合。
- AI 检测是**移动靶**：换生成器/换领域极易崩。公开 "99%" 多半 OOD 就失效。
  本项目只求在**中文通用**这一窄带上可靠，并**如实报告跨生成器/跨领域的掉点**。

## 1. 硬件 / 软件

- GPU：RX 7900 XT 20GB（gfx1100）。⚠️ 桌面也在这块卡上（见 AGENTS.md 铁律）。
- 训练前必须**腾显存**：现在 18.5GB 被 KVMem Qwen 占着。
  - 停：`fuser -k 18200/tcp`（KVMem 非自启，训练后手动 `~/LLM/kvmem/run-kvmem.sh` 恢复）
  - 400M 编码器 + bs32/seq512 约需 6–10GB，够。
- 环境：新建 venv（原 `bonsai-draft/train/env` 已丢）。
  - torch：`download.pytorch.org/whl/rocm7.2` 的 `torch==2.14.0+rocm7.2`（cp312），实测直连 24MB/s
  - 其余：transformers、datasets、accelerate、scikit-learn、numpy、peft/trl（可选）
  - HF 走 `hf-mirror.com`（`HF_ENDPOINT=https://hf-mirror.com`）

## 2. 数据（决定一切）

### 2.1 现成数据集（hf-mirror）
- **HC3-Chinese**（`Hello-SimpleAI/HC3-Chinese`）：人 vs ChatGPT 问答，~40k 对 → 问答域
- **M4**（多生成器多领域，含 zh）：生成器多样性关键
- **SemEval-2024 Task 8 Subtask B**（人机混写，多语含中文）
- 备选：MAGE、AuTexTification、中文新闻语料（人写侧）

### 2.2 自造（本地模型补生成器 + 对齐领域）
- 用本地 **Qwen3.8-27B**（11434 或 18200）在中文**新闻/社交/问答**题材上生成 AI 文本
  - 多 prompt × 多温度 × 多长度，覆盖「AI 改过（AI-edited）」这一类
- 人写侧：新闻语料 + 论坛/问答语料（注意隐私与许可，仅本地训练）

### 2.3 划分（红线）
- **按生成器切分**，留 1–2 个生成器完全不参与训练 → 测泛化
- **按领域切分**，留一个领域（如社交）做 OOD 测试
- 禁止随机切分冒充泛化

## 3. 模型

| 选项 | 基座 | 参数量 | 备注 |
|---|---|---|---|
| A 首选 | `hfl/chinese-roberta-wwm-ext` | 102M | 中文稳、快、省 |
| B | `hfl/chinese-macbert-large` | 326M | 更强，慢些 |
| C 对照 | multilingual ModernBERT（mmBERT-base） | 322M | Laya 同源，8192 ctx |

- 头：`AutoModelForSequenceClassification`，2 类（人/AI）或 3 类（人/AI/AI改）。
- 也可学 Laya 出**连续 AI 味分数**（回归 / 期望值）。

## 4. 训练

- 交叉熵；class weight 平衡；early stop 看 OOD 验证集。
- **温度校准必做**（Laya 的核心卖点），报 ECE / Brier / AUROC。
- 从 A（102M）先跑通全流程，再上 B/C。

## 5. 评测（诚实版）

- 指标：accuracy / F1 / AUROC / **ECE**，**分生成器、分领域**报。
- 必测：留出生成器的泛化；真·OOD 领域；短文本；人工改写冲击。
- 输出 `results/` 下的报告，含「哪里会崩」的明示。

## 6. 阶段与闸门

- **T0 环境**：venv + ROCm torch 可 `torch.cuda.is_available()==True`
- **T1 数据**：≥5万条人/AI 标注样本，跨 ≥3 领域、≥3 生成器
- **T2 基线**：A 基座在**留出生成器**上 AUROC ≥ 0.90（达不到就说明数据/任务问题）
- **T3 泛化报告**：跨生成器/跨领域完整表 + ECE
- **T4（可选）**：检测器当 reward，Qwen 生成多版改写选最低分；或造偏好对 DPO

## 7. 风险

- 跨生成器崩 → 靠数据多样性 + 持续补新生成器
- 中文短文本（社交）难 → 可能需专门子模型
- 自训检测器只对本领域可靠，别当通用产品
- GPU 与桌面/LLM 争用 → 训练时停 KVMem，训完恢复
