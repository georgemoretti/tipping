# GPU setup: vLLM on the Windows PC (RTX 5060 Ti 16 GB)

Goal: vLLM runs inside WSL2 on the Windows PC and serves an OpenAI-compatible API. The Mac sends requests to it over the local network.

```
Mac (code, tests)  ──HTTP──►  Windows PC ─ WSL2 Ubuntu ─ vLLM :8000 ─ RTX 5060 Ti
```

## 1. Windows side

1. Update the **NVIDIA Windows driver** to the latest Game Ready / Studio driver. Don't install a Linux NVIDIA driver inside WSL: the Windows driver provides CUDA to WSL.
2. In PowerShell (as administrator):
   ```powershell
   wsl --install -d Ubuntu-24.04
   wsl --update
   ```
3. Turn on **mirrored networking**, so the Mac can reach WSL on the PC's own IP. Create `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   Then run `wsl --shutdown` and open Ubuntu again.
4. Allow inbound port 8000. In PowerShell (as administrator):
   ```powershell
   New-NetFirewallHyperVRule -Name vLLM8000 -DisplayName "vLLM 8000" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8000
   New-NetFirewallRule -DisplayName "vLLM 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Private
   ```
   Your home network must be set to **Private** in Windows network settings.
5. Note the PC's local IP: `ipconfig` → IPv4 address (e.g. `192.168.1.50`).

## 2. WSL2 (Ubuntu) side

```bash
nvidia-smi                       # must show the RTX 5060 Ti
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
mkdir -p ~/vllm && cd ~/vllm
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
The last line must print a CUDA version ≥ 12.8 and capability `(12, 0)`. If it doesn't, the installed vLLM/PyTorch build doesn't support Blackwell (RTX 50xx) yet. Tell Claude what it printed.

## 3. Start the server

Pick a long random key and keep it private. It stops other devices on your network from using the server.
```bash
export VLLM_KEY=$(openssl rand -hex 16); echo $VLLM_KEY
vllm serve Qwen/Qwen3-8B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --api-key $VLLM_KEY \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes
```
The first start downloads about 9 GB. If it runs out of memory, lower `--max-model-len` to 8192.

## 4. Test from the Mac

```bash
export TIPPING_LLM_BASE_URL=http://<windows-pc-ip>:8000/v1
export TIPPING_LLM_API_KEY=<the VLLM_KEY value>
uv run python scripts/vllm_smoke.py --n 32
```
Expected: three lines starting `[1/3] server OK`, `[2/3] … tokens/s`, `[3/3] tool call OK`.
