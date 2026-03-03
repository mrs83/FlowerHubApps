# FlowerTune LLM Labs - Qwen/Qwen3-4B-Coding

This directory conducts federated instruction tuning with a pretrained [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) model on a [Code dataset](https://huggingface.co/datasets/flwrlabs/code-alpaca-20k).

We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.

[Flower](https://flower.ai/)'s Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Methodology

This app performs federated LLM fine-tuning with [DoRA](https://arxiv.org/abs/2402.09353) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with `FedAvg` strategy.

### Qwen/Qwen3-4B

- **Precision**: `bf16` for model weights.
- **Quantization**: `4-bit` quantization for reduced memory usage.
- **Optimizer**: `paged_adamw_8bit`
- **[DoRA](https://arxiv.org/abs/2402.09353) Configuration**:
  - Rank (r): `8`
  - Alpha: `16`
  - Target Modules:
    - `down_proj`
    - `up_proj`
    - `o_proj`
    - `q_proj`
    - `k_proj`
    - `v_proj`
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
- **Strategy**: `FedAvg` and `FlexLora`

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

You can run the experiment with default config values by running the following command:

```bash
flwr run
```

The default configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

## Model saving

The global PEFT model checkpoints are saved every 1 round after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

---
*Flower App by [ethicalabs.ai](https://ethicalabs.ai) - AI/ML research and development - [HuggingFace](https://huggingface.co/ethicalabs)*
