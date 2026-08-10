# ArtEmis + OLA: Experimental Code

This project implements a lightweight, interpretable, and fully local research pipeline:

1. Predict the distribution of nine audience emotion categories based on art style, artwork title, and OLA objective descriptions.
2. Generate emotion-specific artwork explanations based on objective descriptions and target emotions.
3. Predict the Top-3 emotions and generate three corresponding narrative versions, while using simple rules to reduce unsupported storytelling.

All models are run locally through Hugging Face Transformers, without calling external APIs such as ChatGPT. When pretrained models are used for the first time, their weights need to be downloaded. After downloading, the local cache can be used for offline execution.

## File Descriptions

### `prepare_data.py`

This script is responsible for data cleaning and training set construction:

- Reads the original ArtEmis and OLA CSV files;
- Removes exact duplicate rows from ArtEmis;
- Precisely joins the two datasets using `art_style + painting`;
- Computes the nine-dimensional emotion distribution for each artwork;
- Constructs training samples in the form of “objective description + target emotion → affective explanation”;
- Splits the dataset into training, validation, and test sets at the artwork level to prevent leakage across splits;
- The original CSV files are never modified.

Run:

```powershell
python prepare_data.py
```

Outputs:

- `processed/artwork_emotion_distribution.csv`: artwork-level nine-dimensional emotion distributions;
- `processed/narrative_pairs.csv`: emotion-controlled generation training pairs;
- `processed/metadata.json`: sample counts, category distributions, and split statistics.

To regenerate the processed data:

```powershell
python prepare_data.py --overwrite
```

### `train_emotion_distribution.py`

This script trains the emotion distribution prediction model. By default, it uses `microsoft/deberta-v3-base`, and the model outputs a nine-dimensional Softmax probability distribution.

- Loss function: Jensen-Shannon divergence;
- Metrics: JSD, Top-1 Accuracy, and Recall@3;
- The best model is saved in `models/emotion_distribution/`.

Run:

```powershell
python train_emotion_distribution.py --device cuda --epochs 5 --batch-size 16
```

If GPU memory is insufficient, a smaller model can be used:

```powershell
python train_emotion_distribution.py --model-name distilroberta-base --batch-size 8
```

### `train_narrative_generator.py`

This script trains the emotion-controlled artwork explanation generation model. By default, it uses `google/flan-t5-base`.

- Input: objective description, art style, artwork title, and target emotion;
- Output: an artwork explanation aligned with the target emotion;
- Metrics: BLEU-4, ROUGE-L, and unsupported claim rate;
- The best model is saved in `models/narrative_generator/`;
- Test set generation results are saved in `test_generations.csv` for case analysis in the paper.

Run:

```powershell
python train_narrative_generator.py --device cuda --precision auto --epochs 5 --batch-size 4
```

After training, the script evaluates the full test set by default, including all 2,673 test records. If you only want to quickly check the workflow, you can explicitly limit the number of evaluation samples, for example:

```powershell
python train_narrative_generator.py --eval-max-samples 500
```

On NVIDIA GPUs that support BF16, such as RTX 4070, `--precision auto` will automatically select BF16. BF16 is more suitable than FP16 for T5/FLAN-T5 and can help avoid `NaN` training loss caused by numerical overflow. If the script detects a non-finite loss, it will stop immediately instead of wasting later epochs. If BF16 is still unstable, use `--precision fp32 --batch-size 2`, although training will be slower.

If GPU memory is insufficient, it is recommended to use:

```powershell
python train_narrative_generator.py --model-name google/flan-t5-small --batch-size 4
```

This script also includes a Top-3 multi-perspective demonstration mode. It first loads the emotion distribution model, selects the three emotions with the highest probabilities, and then generates artwork explanations for each emotion. For each emotion, three candidates are generated, and one final result is selected based on objective-description word coverage and prohibited-claim rules.

```powershell
python train_narrative_generator.py --mode demo `
  --style Realism `
  --painting sample-painting `
  --description "A woman sits alone beside a table."
```

### `evaluate_narrative_generator.py`

This script only loads the saved best generation model and does not retrain it. It will:

- Use the full test set instead of only the first 500 samples;
- Group samples by unique input prompts and use multiple ArtEmis explanations corresponding to the same prompt as multi-reference answers;
- Generate 3 beam candidates for each prompt;
- Evaluate both the original best-beam output and the lightweight reliability-based reranked output;
- Report multi-reference BLEU-4, maximum/mean ROUGE-L, grounding score, unsupported claim rate, text diversity, and reranking change rate.

Full evaluation:

```powershell
python evaluate_narrative_generator.py --device cuda --precision bf16
```

Before the first full run, you can perform a smoke test using 20 unique prompts:

```powershell
python evaluate_narrative_generator.py --device cuda --precision bf16 --max-prompts 20
```

If you need to regenerate existing evaluation results, add `--overwrite`. Outputs are saved in:

- `models/narrative_generator/full_test_evaluation/unique_prompt_predictions.csv`
- `models/narrative_generator/full_test_evaluation/full_test_metrics.json`

## Dependency Installation

It is recommended to install dependencies in an independent Python environment:

```powershell
python -m pip install -r requirements.txt
```

If your computer has an NVIDIA GPU, make sure that the CUDA version of PyTorch is installed, rather than the CPU version with `+cpu` in its name. When the program starts, it will print the GPU name and whether AMP is enabled. When using `--device cuda`, if CUDA is unavailable, the program will directly report an error instead of silently switching to CPU.

The NVIDIA driver on the current project computer is suitable for the PyTorch CUDA 12.6 version. In the `ziyue` environment, the CPU version of PyTorch can be replaced as follows:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last command should display `True` and the name of the NVIDIA GPU. During training, it is recommended to explicitly specify:

```powershell
python train_emotion_distribution.py --device cuda --batch-size 16
```

Before training the models, it is recommended to use an NVIDIA GPU with CUDA support. The code can also run without a GPU, but training FLAN-T5-base on CPU will be very slow.

## Recommended Experimental Order

```text
prepare_data.py
        ↓
train_emotion_distribution.py
        ↓
train_narrative_generator.py
        ↓
train_narrative_generator.py --mode demo
```

In an EI conference paper, the first two scripts can be used as the main experiments, while the Top-3 multi-perspective results can be presented as a system demonstration and case analysis. The lightweight reliability control module can be treated as an auxiliary component and does not require training an additional complex model.