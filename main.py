"""
Supply Chain 5.0
Local VS Code Version
Real AI using local GGUF (llama.cpp CPU)
Federated Simulation + Optimization + Human Override + AI Impact
"""

import os
import gc
import glob
import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================
# Configuration
# =====================================================
@dataclass
class SCConfig:
    MODEL_NAME: str = "SupplyChain-Qwen (GGUF via llama.cpp CPU)"
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    LLM_GGUF_PATH: str = os.getenv(
        "SC_LLM_GGUF_PATH",
        os.path.join(PROJECT_ROOT, "models", "SupplyChain-Qwen.gguf")
    )
    LLAMA_CPP_CLI_PATH: str = os.getenv(
        "SC_LLAMA_CPP_CLI_PATH",
        os.path.join(PROJECT_ROOT, "Llama-cpp-cpu", "llama-completion.exe")
    )
    PRODUCT_NAME: str = "Milk"  # Fixed Product
    NUM_CLIENTS: int = 3
    NUM_ROUNDS: int = 2
    CARBON_CAP: float = 500.0  # Adjusted for Milk (e.g. per batch)
    LOG_DIR: str = "sc50_logs"
    DATASET_DIR: str = "DATASETS"
    DP_EPSILON: float = 5.0  # Privacy Budget (Lower = More Privacy/Noise)
    
    # Financials (Per Unit)
    SELLING_PRICE: float = 4.0
    COST_PRICE: float = 1.5
    WASTE_COST: float = 0.5  # Cost of disposal/spoilage


# Create logs folder if missing
os.makedirs(SCConfig.LOG_DIR, exist_ok=True)


# =====================================================
# Utility
# =====================================================
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj


# =====================================================
# Load Local GGUF Model (llama.cpp CPU)
# =====================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _resolve_gguf_path() -> str:
    candidates = [
        SCConfig.LLM_GGUF_PATH,
        os.getenv("LLM_GGUF_PATH", ""),
        os.path.join(SCConfig.PROJECT_ROOT, "models", "SupplyChain-Qwen.gguf"),
        os.path.join("models", "SupplyChain-Qwen.gguf"),
        os.path.join("models", "supplychain-qwen.gguf"),
    ]

    hf_named_files = sorted(glob.glob(os.path.join("**", "*SupplyChain*Qwen*.gguf"), recursive=True))
    generic_gguf_files = sorted(glob.glob(os.path.join("**", "*.gguf"), recursive=True))
    candidates.extend(hf_named_files)
    candidates.extend(generic_gguf_files)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "GGUF model file not found. Put your local SupplyChain-Qwen .gguf in the project "
        "(e.g. models/SupplyChain-Qwen.gguf) or set SC_LLM_GGUF_PATH."
    )

def _resolve_llama_cli_path() -> str:
    candidates = [
        SCConfig.LLAMA_CPP_CLI_PATH,
        os.getenv("LLAMA_CPP_CLI_PATH", ""),
        os.path.join(SCConfig.PROJECT_ROOT, "Llama-cpp-cpu", "llama-completion.exe"),
        os.path.join(SCConfig.PROJECT_ROOT, "llama-completion.exe"),
        os.path.join(SCConfig.PROJECT_ROOT, "Llama-cpp-cpu", "llama-cli.exe"),
        os.path.join(SCConfig.PROJECT_ROOT, "llama-cli.exe"),
        os.path.join("Llama-cpp-cpu", "llama-completion.exe"),
        os.path.join("Llama-cpp-cpu", "llama-cli.exe"),
        "llama-completion.exe",
        "llama-cli.exe",
    ]
    candidates.extend(sorted(glob.glob(os.path.join("**", "llama-completion.exe"), recursive=True)))
    candidates.extend(sorted(glob.glob(os.path.join("**", "llama-cli.exe"), recursive=True)))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""

class LlamaCppCLIModel:
    """Minimal wrapper to call local llama.cpp CLI while preserving current app flow."""
    def __init__(self, cli_path: str, model_path: str, n_ctx: int = 2048):
        self.cli_path = cli_path
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = max(1, (os.cpu_count() or 2) - 1)

    def create_chat_completion(self, messages, max_tokens=200, temperature=0.7, top_p=0.9, repeat_penalty=1.1):
        prompt = _messages_to_prompt(messages)
        exe_name = os.path.basename(self.cli_path).lower()

        cmd = [
            self.cli_path,
            "-m", self.model_path,
            "-ngl", "0",  # force CPU
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--repeat-penalty", str(repeat_penalty),
            "--no-warmup",
            "--simple-io",
            "-p", prompt,
        ]

        if "llama-completion" in exe_name:
            cmd.extend(["-no-cnv", "--no-display-prompt"])
        else:
            cmd.extend(["--single-turn", "--no-display-prompt"])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("llama-cli generation timed out after 180 seconds.") from exc

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "Unknown llama-cli error"
            raise RuntimeError(f"llama-cli failed: {err}")

        content = (proc.stdout or "").strip()
        if content.startswith(prompt):
            content = content[len(prompt):].strip()
        content = content.replace("[end of text]", "").strip()
        if content.lower().startswith("assistant:"):
            content = content[len("assistant:"):].strip()
        if content.lower().startswith("assistant\n"):
            content = content.split("\n", 1)[-1].strip()
        if not content:
            err = proc.stderr.strip()
            raise RuntimeError(f"llama-cli returned empty output. stderr: {err[:500]}")

        return {"choices": [{"message": {"content": content}}]}

def load_model():
    model_path = _resolve_gguf_path()
    cli_path = _resolve_llama_cli_path()
    log(f"Using GGUF model: {model_path}")

    model = None
    try:
        from llama_cpp import Llama
        n_threads = max(1, (os.cpu_count() or 2) - 1)
        log("Loading model via llama-cpp-python (CPU mode)...")
        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=0,
            chat_format="chatml",
            use_mmap=True,
            verbose=False,
        )
    except ImportError:
        if not cli_path:
            raise ImportError(
                "Could not import llama-cpp-python and no local llama.cpp executable was found. "
                "Set SC_LLAMA_CPP_CLI_PATH or install llama-cpp-python."
            )
        log(f"llama-cpp-python not found. Falling back to local llama.cpp executable: {cli_path}")
        model = LlamaCppCLIModel(cli_path=cli_path, model_path=model_path, n_ctx=2048)

    log("Model Loaded Successfully.")
    return None, model


def _normalize_messages(prompt) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": f"You are a helpful Supply Chain Assistant optimized for {SCConfig.PRODUCT_NAME}."},
            {"role": "user", "content": prompt},
        ]
    return prompt


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").strip().lower()
        content = str(m.get("content", "")).strip()
        lines.append(f"{role.title()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from Qwen3 model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def llm_generate(prompt, tokenizer, model, max_tokens=200, temperature=0.7):
    if model is None:
        raise ValueError("Model is not loaded. Please load the GGUF model first.")

    messages = _normalize_messages(prompt)
    try:
        out = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        content = out["choices"][0]["message"]["content"]
        content = content.strip() if isinstance(content, str) else str(content)
        return _strip_think_tags(content)
    except Exception:
        # Fallback for runtimes that do not expose chat completion helpers
        text_prompt = _messages_to_prompt(messages)
        out = model(
            text_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["\nUser:", "\nSystem:"],
        )
        return _strip_think_tags(out["choices"][0]["text"].strip())


# =====================================================
# Differential Privacy
# =====================================================
class DifferentialPrivacy:
    @staticmethod
    def add_noise(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
        """Adds Laplacian noise for Differential Privacy."""
        if epsilon <= 0: return value # No privacy
        beta = sensitivity / epsilon
        noise = np.random.laplace(0, beta)
        return value + noise

    @staticmethod
    def clip_gradients(value: float, clip_norm: float = 5.0) -> float:
        """Clips the update to bound sensitivity."""
        return max(min(value, clip_norm), -clip_norm)


# =====================================================
# Dataset Manager
# =====================================================
class SupplyChainDataManager:
    def __init__(self, num_clients: int, weeks: int = 52, dataset_dir: str = SCConfig.DATASET_DIR):
        self.num_clients = num_clients
        self.weeks = weeks
        self.dataset_dir = dataset_dir
        self.client_data = {}
        self.load_data()

    def load_data(self):
        dataset_files = self._discover_dataset_files()
        mapped_files = self._map_dataset_files(dataset_files)
        loaded_from_dataset = 0

        for cid in range(self.num_clients):
            client_key = str(cid)
            if client_key in mapped_files:
                self.client_data[client_key] = self._load_client_dataset(mapped_files[client_key], cid)
                loaded_from_dataset += 1
            else:
                self.client_data[client_key] = self._generate_synthetic_client(cid)

        if loaded_from_dataset == 0:
            log(f"No CSV/XLS/XLSX files found in '{self.dataset_dir}'. Using synthetic data for all clients.")
        elif loaded_from_dataset < self.num_clients:
            log(
                f"Loaded {loaded_from_dataset} client dataset(s) from '{self.dataset_dir}'. "
                f"Generated synthetic data for {self.num_clients - loaded_from_dataset} extra client(s)."
            )
        else:
            log(f"Loaded client datasets from '{self.dataset_dir}' for all {self.num_clients} clients.")

    def _discover_dataset_files(self):
        patterns = ["*.csv", "*.xls", "*.xlsx"]
        dataset_files = []
        for pattern in patterns:
            dataset_files.extend(glob.glob(os.path.join(self.dataset_dir, pattern)))
        return sorted(set(dataset_files))

    def _map_dataset_files(self, dataset_files):
        mapped_files = {}
        remaining_files = []

        for file_path in dataset_files:
            file_name = os.path.basename(file_path).lower()
            upload_match = re.search(r"client_(\d+)__upload__current\.(csv|xls|xlsx)$", file_name)
            if upload_match:
                mapped_files[upload_match.group(1)] = file_path
            else:
                remaining_files.append(file_path)

        unassigned_clients = [str(cid) for cid in range(self.num_clients) if str(cid) not in mapped_files]
        for cid, file_path in zip(unassigned_clients, remaining_files):
            mapped_files[cid] = file_path

        return mapped_files

    def _load_client_dataset(self, csv_path: str, cid: int) -> pd.DataFrame:
        df = self._read_tabular_file(csv_path)
        required_cols = ["demand", "disruption_prob", "emission_factor"]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            log(f"{os.path.basename(csv_path)} missing {missing}; falling back to synthetic client {cid}.")
            return self._generate_synthetic_client(cid)

        data = pd.DataFrame()
        if "week" in df.columns:
            week_series = pd.to_numeric(df["week"], errors="coerce")
            week_series = week_series.ffill().bfill()
            data["week"] = week_series.fillna(pd.Series(np.arange(len(df)))).astype(int)
        else:
            data["week"] = np.arange(len(df))

        data["demand"] = pd.to_numeric(df["demand"], errors="coerce")
        data["disruption_prob"] = pd.to_numeric(df["disruption_prob"], errors="coerce")
        data["emission_factor"] = pd.to_numeric(df["emission_factor"], errors="coerce")

        for col in ["demand", "disruption_prob", "emission_factor"]:
            if data[col].isna().all():
                log(f"{os.path.basename(csv_path)} has invalid '{col}' values; using synthetic client {cid}.")
                return self._generate_synthetic_client(cid)
            data[col] = data[col].interpolate(limit_direction="both").fillna(data[col].median())

        data["demand"] = data["demand"].clip(lower=0).astype(int)
        data["disruption_prob"] = data["disruption_prob"].clip(lower=0, upper=1)
        data["emission_factor"] = data["emission_factor"].clip(lower=0.01)
        data = data[["week", "demand", "disruption_prob", "emission_factor"]].dropna().reset_index(drop=True)

        if len(data) < 6:
            log(f"{os.path.basename(csv_path)} has too few rows ({len(data)}); using synthetic client {cid}.")
            return self._generate_synthetic_client(cid)

        return data

    def _read_tabular_file(self, file_path: str) -> pd.DataFrame:
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(file_path)
        raise ValueError(f"Unsupported dataset file type: {file_path}")

    def _generate_synthetic_client(self, cid: int) -> pd.DataFrame:
        rng = np.random.default_rng(42 + cid)
        t = np.arange(self.weeks)
        trend = 100 + (t * 0.5)
        seasonality = 20 * np.sin(2 * np.pi * t / 12)
        noise = rng.normal(0, 5, self.weeks)
        firm_shift = rng.integers(-10, 20)

        demand = trend + seasonality + noise + firm_shift
        disruption_prob = np.clip(rng.beta(2, 10, self.weeks), 0, 1)
        emission_factor = np.full(self.weeks, 1.5 + rng.random() * 0.5)

        return pd.DataFrame({
            "week": t,
            "demand": demand.astype(int),
            "disruption_prob": disruption_prob,
            "emission_factor": emission_factor
        })

    def get_client_data(self, cid: str):
        return self.client_data[str(cid)]


# =====================================================
# Federated LSTM Model
# =====================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def federated_average(models_state_dict):
    """Averages the weights of multiple models."""
    global_dict = models_state_dict[0].copy()
    for k in global_dict.keys():
        for i in range(1, len(models_state_dict)):
            global_dict[k] += models_state_dict[i][k]
        global_dict[k] = torch.div(global_dict[k], len(models_state_dict))
    return global_dict


# =====================================================
# Federated Simulation (LSTM)
# =====================================================
class FedSim:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.input_size = 1
        self.sequence_length = 5
        self.max_val = self._compute_normalization_scale()
        # Initialize Global Model
        self.global_model = LSTMModel(input_size=self.input_size)
        self.metrics = {"rounds": [], "mae": [], "rmse": [], "loss": []}

    def _compute_normalization_scale(self) -> float:
        demand_max = 1.0
        for cid in range(self.data_manager.num_clients):
            df = self.data_manager.get_client_data(str(cid))
            demand_max = max(demand_max, float(df["demand"].max()))
        return demand_max

    def train_client(self, cid, global_weights, epochs=5, lr=0.01):
        """Trains a local model on client data."""
        # Load local model with global weights
        local_model = LSTMModel(input_size=self.input_size)
        local_model.load_state_dict(global_weights)
        local_model.train()
        
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare Data
        df = self.data_manager.get_client_data(str(cid))
        data = df["demand"].values.astype(np.float32)
        
        # Normalize using the actual demand scale across participating clients.
        data_norm = data / self.max_val

        # Create Sequences
        X, y = [], []
        for i in range(len(data_norm) - self.sequence_length):
            X.append(data_norm[i:i+self.sequence_length])
            y.append(data_norm[i+self.sequence_length])
            
        X = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(-1) # (Batch, Seq, Feature)
        y = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(-1) # (Batch, 1)
        
        # Local Training Loop
        epoch_loss = 0
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        return local_model.state_dict(), epoch_loss / epochs

    def run(self, tokenizer=None, model=None, epsilon=SCConfig.DP_EPSILON):
        # NOTE: tokenizer/model args kept for compatibility but not used for LSTM training
        log("Starting Federated LSTM Simulation")
        
        self.metrics = {"rounds": [], "mae": [], "rmse": [], "loss": []}
        
        for r in range(SCConfig.NUM_ROUNDS):
            log(f"--- Round {r+1} ---")
            local_weights = []
            round_loss = 0
            
            # Broadcast Global Weights
            global_weights = self.global_model.state_dict()
            
            for cid in range(SCConfig.NUM_CLIENTS):
                # Train Client
                w, loss = self.train_client(cid, global_weights)
                
                # --- Differential Privacy (Add Noise to Weights) ---
                # Simple implementation: Add noise to each weight tensor
                if epsilon > 0:
                    for k in w.keys():
                        noise = torch.tensor(np.random.laplace(0, 0.01 / epsilon, w[k].shape)).float()
                        w[k] += noise
                # ---------------------------------------------------
                
                local_weights.append(w)
                round_loss += loss
                
            # Aggregation (FedAvg)
            new_global_weights = federated_average(local_weights)
            self.global_model.load_state_dict(new_global_weights)
            
            # Validation (Metrics on all clients)
            # Use the new global model to predict last known data point
            total_mae = 0
            total_rmse = 0
            
            self.global_model.eval()
            with torch.no_grad():
                for cid in range(SCConfig.NUM_CLIENTS):
                    df = self.data_manager.get_client_data(str(cid))
                    data = df["demand"].values.astype(np.float32)
                    
                    # Predict last week using previous sequence
                    last_seq = data[-self.sequence_length-1:-1] / self.max_val
                    true_val = data[-1]
                    
                    inp = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    pred_norm = self.global_model(inp).item()
                    pred = max(0, int(pred_norm * self.max_val))
                    
                    err = abs(true_val - pred)
                    total_mae += err
                    total_rmse += err**2
                    
            avg_loss = round_loss / SCConfig.NUM_CLIENTS
            mae = total_mae / SCConfig.NUM_CLIENTS
            rmse = np.sqrt(total_rmse / SCConfig.NUM_CLIENTS)
            
            self.metrics["rounds"].append(r+1)
            self.metrics["mae"].append(mae)
            self.metrics["rmse"].append(rmse)
            self.metrics["loss"].append(avg_loss)

            log(f"Round {r+1} | Loss: {avg_loss:.4f} | MAE: {mae:.2f}")

        return self.global_model, self.metrics


# =====================================================
# Optimization
# =====================================================
def optimize(
    forecast,
    inventory,
    emission_factor,
    risk,
    override_qty=None,
    carbon_cap=None,
    selling_price=None,
    cost_price=None,
    waste_cost=None,
):
    carbon_cap = SCConfig.CARBON_CAP if carbon_cap is None else carbon_cap
    selling_price = SCConfig.SELLING_PRICE if selling_price is None else selling_price
    cost_price = SCConfig.COST_PRICE if cost_price is None else cost_price
    waste_cost = SCConfig.WASTE_COST if waste_cost is None else waste_cost

    # Safety Stock includes risk buffer
    safety_stock = int(forecast * (0.1 + risk))
    
    # Order Qty logic
    if override_qty is not None:
        qty = override_qty
    else:
        qty = max(0, forecast + safety_stock - inventory)

    # Emissions
    emissions = float(qty * emission_factor)
    feasible = emissions <= carbon_cap
    
    # Financials (Projected)
    # Scenario: We sell everything we forecast (up to available stock)
    # Available for sale = Inventory + Qty
    available_stock = inventory + qty
    projected_sales = min(forecast, available_stock)
    unsold_stock = max(0, available_stock - projected_sales)
    
    revenue = projected_sales * selling_price
    cost = qty * cost_price # Cost of new order
    # Note: Logic for "Profit" usually includes Cost of Goods Sold (COGS). 
    # Here we simplify: Project Cost = Cost of New Order + Holding/Waste of Unsold.
    
    # Assuming unsold milk spoils (Waste Cost)
    waste_cost_total = unsold_stock * waste_cost
    
    net_profit = revenue - cost - waste_cost_total

    return {
        "optimized_qty": qty,
        "emissions": emissions,
        "feasible": feasible,
        "safety_stock": safety_stock,
        "financials": {
            "revenue": revenue,
            "order_cost": cost,
            "waste_cost": waste_cost_total,
            "net_profit": net_profit
        }
    }


# =====================================================
# Main Pipeline
# =====================================================
def main():
    device = get_device()
    log(f"Running on {device}")

    # Load LLM for Explanation only
    tokenizer, model = load_model()

    data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)

    # Federated LSTM Training
    fed = FedSim(data_manager)
    lstm_model, metrics = fed.run(tokenizer, model)

    # FINAL FORECAST (Using trained LSTM)
    client0_df = data_manager.get_client_data("0")
    data = client0_df["demand"].values.astype(np.float32)
    max_val = fed.max_val
    
    # Get last 5 weeks
    last_seq = data[-5:] / max_val
    inp = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    lstm_model.eval()
    with torch.no_grad():
        pred_norm = lstm_model(inp).item()
        
    forecast = max(0, int(pred_norm * max_val))
    
    # Get last known emission/risk factors
    last_week_data = client0_df.iloc[-1]
    
    log(f"Final LSTM Forecast: {forecast}")

    opt = optimize(
        forecast=forecast,
        inventory=50,
        emission_factor=float(last_week_data["emission_factor"]),
        risk=float(last_week_data["disruption_prob"])
    )

    print("\nAI Recommendation (Explanation):\n")
    print(llm_generate(
        f"Forecast: {forecast}, Order Qty: {opt['optimized_qty']}, Emissions: {opt['emissions']}. Provide recommendation.",
        tokenizer,
        model,
        max_tokens=120,
        temperature=0.7 
    ))

    print("\nSuggested Order:", opt["optimized_qty"])
    print(f"Final MAE: {metrics['mae'][-1]:.2f}")
    
    user = input("Press Enter to approve or type new quantity: ").strip()

    if user.isdigit():
        new_qty = int(user)
        log_entry = {"event": "override", "new": new_qty}
    else:
        log_entry = {"event": "approved", "qty": opt['optimized_qty']}

    with open(os.path.join(SCConfig.LOG_DIR, "decision_log.json"), "w") as f:
        json.dump(to_serializable(log_entry), f, indent=2)

    log("Decision saved.")


if __name__ == "__main__":
    main()
