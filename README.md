# ArtEmis + OLA：实验代码

本项目实现一条轻量、可解释、完全本地运行的研究路线：

1. 根据艺术风格、作品名称和 OLA 客观描述，预测九类受众情绪分布；
2. 根据客观描述和指定情绪，生成对应情绪视角的艺术解释；
3. 预测 Top-3 情绪并分别生成三个版本，同时用简单规则减少无依据叙事。

所有模型均通过 Hugging Face Transformers 在本地运行，不调用 ChatGPT 等外部 API。首次使用预训练模型时需要下载权重，下载完成后可以使用本地缓存离线运行。

## 文件用途

### `prepare_data.py`

负责数据清洗和训练集构造：

- 读取原始 ArtEmis 和 OLA CSV；
- 删除 ArtEmis 中完全相同的重复行；
- 根据 `art_style + painting` 精确连接两个数据集；
- 为每幅作品计算九类情绪比例；
- 构造“客观描述 + 目标情绪 → 情感解释”训练样本；
- 按作品划分训练集、验证集和测试集，防止同一幅作品跨集合泄漏；
- 原始 CSV 永远不会被修改。

运行：

```powershell
python prepare_data.py
```

输出：

- `processed/artwork_emotion_distribution.csv`：作品级九维情绪分布；
- `processed/narrative_pairs.csv`：情绪控制生成训练对；
- `processed/metadata.json`：样本数量、类别分布和划分统计。

如果需要重新生成：

```powershell
python prepare_data.py --overwrite
```

### `train_emotion_distribution.py`

训练情绪分布预测模型。默认使用 `microsoft/deberta-v3-base`，模型最后输出九维 Softmax 概率。

- 损失函数：Jensen–Shannon divergence；
- 指标：JSD、Top-1 Accuracy、Recall@3；
- 最优模型保存在 `models/emotion_distribution/`。

运行：

```powershell
python train_emotion_distribution.py --device cuda --epochs 5 --batch-size 16
```

显存不足时可以改用更小模型：

```powershell
python train_emotion_distribution.py --model-name distilroberta-base --batch-size 8
```

### `train_narrative_generator.py`

训练情绪控制艺术解释生成模型，默认使用 `google/flan-t5-base`。

- 输入：客观描述、艺术风格、作品名称和目标情绪；
- 输出：与目标情绪相符的艺术解释；
- 指标：BLEU-4、ROUGE-L 和无依据声明比例；
- 最优模型保存在 `models/narrative_generator/`；
- 测试集生成结果保存在 `test_generations.csv`，方便论文案例分析。

运行：

```powershell
python train_narrative_generator.py --device cuda --precision auto --epochs 5 --batch-size 4
```

训练结束后默认评价完整测试集（全部 2,673 条测试记录）。如只想快速检查流程，
可以显式限制评价条数，例如：

```powershell
python train_narrative_generator.py --eval-max-samples 500
```

在支持 BF16 的 NVIDIA 显卡（例如 RTX 4070）上，`--precision auto` 会自动
选择 BF16。BF16 比 FP16 更适合 T5/FLAN-T5，可避免训练损失因数值溢出变成
`NaN`。脚本检测到非有限损失时会立即停止，不再浪费后续 epoch。若 BF16
仍不稳定，可使用 `--precision fp32 --batch-size 2`，但训练会更慢。

显存不足时建议：

```powershell
python train_narrative_generator.py --model-name google/flan-t5-small --batch-size 4
```

该脚本还包含 Top-3 多视角演示模式。它先加载情绪分布模型，选择概率最高的三种情绪，再分别生成艺术解释；每种情绪生成三个候选，并通过客观描述词覆盖度和禁用声明规则选出一个结果。

```powershell
python train_narrative_generator.py --mode demo `
  --style Realism `
  --painting sample-painting `
  --description "A woman sits alone beside a table."
```

### `evaluate_narrative_generator.py`

该脚本只加载已经保存的最佳生成模型，不会重新训练。它会：

- 使用完整测试集，而不是前 500 条；
- 按唯一输入提示分组，将同一提示对应的多条 ArtEmis 解释作为多参考答案；
- 为每个提示生成 3 个 Beam 候选；
- 同时计算原始最佳 Beam 和轻量可信重排序结果；
- 报告多参考 BLEU-4、最大/平均 ROUGE-L、依据得分、禁用声明比例、文本多样性和重排序改写比例。

完整评估：

```powershell
python evaluate_narrative_generator.py --device cuda --precision bf16
```

首次运行前可用 20 个唯一提示进行冒烟测试：

```powershell
python evaluate_narrative_generator.py --device cuda --precision bf16 --max-prompts 20
```

如需重新生成已有评估结果，增加 `--overwrite`。输出保存在：

- `models/narrative_generator/full_test_evaluation/unique_prompt_predictions.csv`
- `models/narrative_generator/full_test_evaluation/full_test_metrics.json`

## 安装依赖

建议在独立 Python 环境中安装：

```powershell
python -m pip install -r requirements.txt
```

如果电脑有 NVIDIA 显卡，必须确认安装的是 CUDA 版 PyTorch，而不是名称带
`+cpu` 的 CPU 版。程序启动后会打印显卡名称和是否启用 AMP；使用
`--device cuda` 时，如果 CUDA 不可用，程序会直接给出错误，不会悄悄改用 CPU。

当前项目电脑的 NVIDIA 驱动适合使用 PyTorch CUDA 12.6 版本。在 `ziyue`
环境中可以这样替换 CPU 版 PyTorch：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一条命令应显示 `True` 和 NVIDIA 显卡名称。训练时建议明确指定：

```powershell
python train_emotion_distribution.py --device cuda --batch-size 16
```

训练模型前建议使用带 CUDA 的 NVIDIA GPU。没有 GPU 也能运行，但 FLAN-T5-base 在 CPU 上训练会非常慢。

## 推荐实验顺序

```text
prepare_data.py
        ↓
train_emotion_distribution.py
        ↓
train_narrative_generator.py
        ↓
train_narrative_generator.py --mode demo
```

EI 论文中可以将前两个作为主要实验，将 Top-3 多视角结果作为系统展示和案例分析。轻量可信控制只作为辅助模块，不需要单独训练复杂模型。
