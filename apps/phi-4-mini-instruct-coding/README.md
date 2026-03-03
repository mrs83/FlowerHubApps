# FlowerTune LLM Labs - microsoft/Phi-4-mini-instruct

This directory conducts federated instruction tuning with a pretrained [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) model on a [Code dataset](https://huggingface.co/datasets/flwrlabs/code-alpaca-20k).

We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.

[Flower](https://flower.ai/)'s Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.


## Methodology

This app performs federated LLM fine-tuning with [DoRA](https://arxiv.org/abs/2402.09353) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.

The clients' models are aggregated with `FedAvg` strategy.

### microsoft/Phi-4-mini-instruct

- **Precision**: `bf16` for model weights.
- **Quantization**: `4-bit` quantization for reduced memory usage.
- **Optimizer**: `paged_adamw_8bit`
- **[DoRA](https://arxiv.org/abs/2402.09353) Configuration**:
  - Rank (r): `16`
  - Alpha: `64`
  - Target Modules:
    - `down_proj`,
    - `gate_up_proj`,
    - `o_proj`,
    - `qkv_proj`,
- **Training Configuration**:
  - Batch size: `8`
  - Maximum number of steps: `10`
  - Accumulation steps: `1`
  - Total number of rounds: `1`
  - Fraction fit per round: `0.2`
- **Learning Rate Scheduler**:
  - Cosine Annealing over rounds, where:
    - Maximum LR: `5e-6`
    - Minimum LR: `5e-5`
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
