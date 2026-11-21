# GPU setup: vLLM on the Windows PC (RTX 5060 Ti 16 GB)

Code, tests and scripts run natively on Windows. vLLM runs inside WSL2 and serves an OpenAI-compatible API.
WSL2 forwards `localhost` to Windows, so the code reaches the server at `http://localhost:8000/v1`.
The server is not exposed to the local network, so no firewall rules or API key are needed.

```
Windows (repo, uv .venv, pytest, scripts)  ──HTTP localhost:8000──►  WSL2 Ubuntu-22.04 ─ vLLM ─ RTX 5060 Ti
```

Tested with: NVIDIA driver 610.88, WSL 2.5.10, vLLM 0.34.0, torch 2.14.0+cu130.

## 1. Check the driver and WSL

Use the normal Windows NVIDIA driver. Don't install a Linux NVIDIA driver inside WSL: the Windows driver provides CUDA to WSL.
```powershell
nvidia-smi
wsl -l -v
wsl -d Ubuntu-22.04 -- nvidia-smi
```
Both `nvidia-smi` calls must show the RTX 5060 Ti.

## 2. Install vLLM (inside WSL)

Open Ubuntu (`wsl -d Ubuntu-22.04`). Needs [uv](https://docs.astral.sh/uv/).
```bash
mkdir -p ~/vllm && cd ~/vllm
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
python -c "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
```
`--torch-backend=auto` picks the PyTorch build that matches the driver's CUDA version.
The last line must print a CUDA version ≥ 12.8 and capability `(12, 0)` (Blackwell).

## 3. Download the model (inside WSL)

About 9 GB. If the download breaks, run the same command again and it continues.
```bash
uvx --from huggingface_hub hf download Qwen/Qwen3-8B-FP8
```

## 4. Start the server (inside WSL)

```bash
cd ~/vllm && source .venv/bin/activate
vllm serve Qwen/Qwen3-8B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes
```
Startup takes about 2 minutes (torch.compile + CUDA graphs). It is ready when it prints `Application startup complete.`
`0.85` leaves room for the ~1.2 GB that the Windows desktop uses on the GPU. With these settings the model takes 8.8 GiB and the KV cache about 21k tokens.

## 5. Smoke test (Windows, from the repo root)

```powershell
uv sync --group dev
uv run python scripts/vllm_smoke.py --n 32
```
Expected: three lines starting `[1/3] server OK`, `[2/3] … tokens/s`, `[3/3] tool call OK`.
First result on this PC: 32 parallel requests, 1284 tokens/s, tool call OK.
