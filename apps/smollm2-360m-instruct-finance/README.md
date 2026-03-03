# FlowerTune LLM on Finance Dataset

This directory conducts federated instruction tuning with a pretrained [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model on a [Finance dataset](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Methodology

This experiment performs federated LLM fine-tuning with [DoRA](https://arxiv.org/abs/2402.09353) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.

The clients' models are aggregated with `FedAvg` strategy.


### SmolLM2-360M-Instruct

- **Precision**: `bf16` for model weights.
- **Quantization**: `4-bit` quantization for reduced memory usage.
- **Optimizer**: `paged_adamw_8bit`
- **[DoRA](https://arxiv.org/abs/2402.09353) Configuration**:
  - Rank (r): `16`
  - Alpha: `64`
  - Target Modules:
    - `down_proj`
    - `up_proj`
    - `gate_proj`
- **Training Configuration**:
  - Batch size: `16`
  - Maximum number of steps: `8`
  - Total number of rounds: `1`
  - Fraction fit per round: `0.1`
- **Learning Rate Scheduler**:
  - Cosine Annealing over rounds, where:
    - Maximum LR: `2e-4`
    - Minimum LR: `6e-6`
  - Constant learning rate scheduler over steps
- **Strategy**: `FedAvg`

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

To run this on [AMD ROCm](https://rocm.docs.amd.com/en/latest/), install with:
```shell
pip install -e ".[rocm]" --extra-index-url https://download.pytorch.org/whl/rocm7.1/
```


## Running the experiment

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

## Model saving

The global PEFT model checkpoints are saved every 1 round after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

---
*Flower App by [ethicalabs.ai](https://ethicalabs.ai) - AI/ML research and development - [HuggingFace](https://huggingface.co/ethicalabs)*
