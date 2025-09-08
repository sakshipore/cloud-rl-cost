## Cloud RL Cost — Windows Setup & Run Guide

This guide shows how to set up and run the project on Windows, generate the trained model, metrics, and plots for your report/demo.

### 1) Prerequisites
- Install Python 3.11 (recommended). Use the Windows installer from `https://www.python.org/downloads/` and check “Add Python to PATH”.
- Git (optional) if you’re cloning the repo.

If you already have Python 3.10–3.12, that should be fine. Avoid 3.13 on Windows as some ML wheels may lag behind.

### 2) Get the code
If you cloned with Git:
```bash
git clone <your-repo-url>
cd cloud-rl-cost
```

Or place this folder on disk and open it in Terminal/PowerShell.

### 3) Create and activate a virtual environment

PowerShell (recommended):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
```

Command Prompt (cmd.exe):
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -V
```

You should see `Python 3.11.x` after activation.

### 4) Install dependencies
Upgrade pip, then install packages (CPU-only PyTorch):

PowerShell or cmd:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install numpy matplotlib gymnasium==0.29.1 stable-baselines3==2.3.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Notes:
- We use CPU-only PyTorch; no CUDA required.
- If installation is slow, ensure your internet connection and try again.

### 5) Quick smoke test (random policy)
This verifies the Gym environment runs end-to-end.

PowerShell:
```powershell
$env:PYTHONPATH = (Get-Location).Path
python rl/test_env.py
```

Command Prompt (cmd.exe):
```bat
set PYTHONPATH=%CD%
python rl\test_env.py
```

Expected: it should print a final total reward (negative is okay for random actions).

### 6) Train and evaluate the RL agent
This trains a DQN agent and saves outputs to the `outputs/` folder.

PowerShell:
```powershell
$env:PYTHONPATH = (Get-Location).Path
python rl/train_dqn.py
```

Command Prompt (cmd.exe):
```bat
set PYTHONPATH=%CD%
python rl\train_dqn.py
```

You’ll see training logs. After it finishes, check `outputs/`:
- `outputs/dqn_cloud_cost.zip` — trained model
- `outputs/eval_metrics.json` — total reward, cost, SLA violations, steps
- `outputs/eval_history.csv` — time series of demand, instances, latency, cost
- `outputs/rl_performance.png` — plot (saved; no GUI required)

### 7) Tips for your demo/report
- Open `outputs/rl_performance.png` to show capacity vs demand, latency vs target, and cumulative cost.
- Quote metrics from `outputs/eval_metrics.json` for quantitative results.
- You can re-run training multiple times to compare metrics after hyperparameter changes in `rl/train_dqn.py`.

### Troubleshooting
- ModuleNotFoundError: No module named 'envs'
  - Ensure you set `PYTHONPATH` to the project root before running scripts (see step 5/6).
  - Alternatively, run from the project root and use module-style execution:
    - PowerShell: `python -m rl.train_dqn`
    - cmd: `python -m rl.train_dqn`

- Torch install issues
  - Stick to Python 3.11 and use the provided CPU index URL.
  - If corporate network blocks downloads, try a different network.

- Plot windows do not appear
  - That’s fine; plots are saved to `outputs/rl_performance.png` non-interactively.

### Clean up
To deactivate the virtual environment:

PowerShell:
```powershell
deactivate
```

cmd:
```bat
deactivate
```

If you want to start fresh, delete the `.venv` folder and `outputs/` directory, then repeat from step 3.


