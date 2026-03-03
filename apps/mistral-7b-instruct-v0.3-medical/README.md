# FlowerTune LLM on Medical Dataset

This directory conducts federated instruction tuning with a pretrained [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).

We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## PEFT Adapter

The fine-tuning results have been submitted as a PEFT adapter and can be accessed here:

[FlowerTune-Mistral-7B-Instruct-v0.3-Medical-PEFT](https://huggingface.co/mrs83/FlowerTune-Mistral-7B-Instruct-v0.3-Medical-PEFT)

## Methodology

This app performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.

The clients' models are aggregated with `FedProx` strategy.

### Mistral-7B-Instruct-v0.3

For the **Mistral-7B-Instruct-v0.3** model, we adopted the following fine-tuning methodology:

- **Precision**: bf16 for model weights
- **Quantization**: 4-bit quantization for reduced memory usage.
- **Optimizer**: Paged AdamW 8-bit for effective optimization under constrained resources.
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 32
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head
- **Training Configuration**:
  - Batch size: 8
  - Maximum number of steps: 6
  - Warmup steps: 2
  - Total number of rounds: 100
  - Fraction fit per round: 0.15
- **Learning Rate Scheduler**: Constant learning rate scheduler with warmup steps, where:
  - Maximum LR: 5e-5
  - Minimum LR: 1e-6
- **Strategy**: FedProx

When bf16 and tf32 are enabled, model weights are stored in bf16 format, while gradients are computed in half-precision and converted to full 32-bit precision for updates.

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

First make sure that you have got the access to [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model with your Hugging-Face account.

You can request access directly from the Hugging-Face website.

Then, follow the instruction [here](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) to log in your account.

Note you only need to complete this stage once in your development machine:

```bash
huggingface-cli login
```

You can run the experiment with default config values by running the following command:

```bash
flwr run
```

The default configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

## Model saving

The global PEFT model checkpoints are saved every 1 round after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

---
*Flower App by [ethicalabs.ai](https://ethicalabs.ai) - AI/ML research and development - [HuggingFace](https://huggingface.co/ethicalabs)*