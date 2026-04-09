# Federated Supply Chain Milk Phase 2

This project is a beginner-friendly demo of a **federated learning supply chain control tower** for milk distribution.

It has two main apps:

- **Control Tower**: runs the federated learning round, generates forecasts, creates market benchmarks, shows node status, and receives raw client dataset uploads.
- **Client Node App**: represents a client/brand node, shows its own performance, compares itself with the market, receives privacy-safe benchmark bundles, and can upload its raw `.csv`, `.xls`, or `.xlsx` file to the control tower.

The project is designed as a practical demo of:

- federated learning concepts
- privacy-preserving aggregation
- secure node communication
- local AI explanations using a GGUF model
- supply chain forecasting and optimization

---

## 1. What This Project Does

At a high level, this system simulates a milk supply chain network where multiple client nodes contribute data to a federated learning process.

Instead of showing every client's raw business data to everyone:

- each client keeps its own identity
- the control tower trains a shared forecasting model
- only privacy-safe market summaries are distributed back to clients
- the client can compare its own performance against the market without seeing peer-level raw data

This makes it useful as a demo for:

- supply chain analytics
- federated learning education
- privacy-preserving benchmarking
- local AI-assisted decision support

---

## 2. Main Features

### Control Tower

- Loads local GGUF LLM model for explanations and chat
- Runs federated learning across client datasets
- Forecasts demand using an LSTM model
- Optimizes order quantity, emissions feasibility, and financials
- Shows **node online/offline state**
- Shows **last seen**
- Shows **last secure sync**
- Accepts **secure raw file uploads** from client nodes
- Sends encrypted benchmark bundles to clients
- Stores human override decisions

### Client Node App

- Runs a secure node endpoint
- Shows local operational metrics
- Shows market comparison using privacy-safe benchmark data
- Can generate AI insight for that client
- Can upload raw `.csv`, `.xls`, or `.xlsx` files to the tower

---

## 3. Project Structure

### Core Files

- `app.py`
  Control tower Streamlit app

- `client_app.py`
  Client node Streamlit app

- `main.py`
  Core federated learning, forecasting, optimization, and model loading logic

- `privacy_network.py`
  Secure communication layer for:
  - node health ping
  - secure bundle sync
  - secure raw dataset upload

- `dashboard_data.py`
  Privacy-safe market benchmarking and client bundle creation

- `requirements.txt`
  Python dependencies

- `run.txt`
  Quick run examples

### Data Folders

- `DATASETS/`
  Raw client dataset files used by the simulation

- `models/`
  GGUF local language model

- `sc50_logs/`
  Logs, node registry, secure endpoint info, and decision logs

---

## 4. Core Architecture

### A. Control Tower

The control tower is the central coordinator.

It is responsible for:

- receiving client raw uploads
- loading datasets
- running the federated learning round
- generating aggregated market benchmarks
- pushing private client bundles back to nodes

### B. Client Nodes

Each client node represents one participant in the network.

A node can:

- expose a secure health endpoint
- receive secure benchmark bundles
- upload raw files to the tower
- view its own data and privacy-safe market comparison

### C. Local LLM

The project uses a local GGUF model for:

- strategic explanation
- chat assistant
- client-facing AI insight

Important:

- The **LLM does not train the forecasting model**
- The **forecasting model is an LSTM in PyTorch**

---

## 5. How the System Works End to End

Here is the normal flow:

1. A client uploads its raw `.csv`, `.xls`, or `.xlsx` file to the control tower.
2. The control tower saves that upload in `DATASETS/` using a client-specific file name.
3. The control tower runs the federated learning simulation.
4. Each client dataset is used as one federated participant.
5. The shared LSTM model is trained round by round.
6. Differential privacy noise is added to client updates before aggregation.
7. The tower generates privacy-safe market benchmarks.
8. The tower securely sends each client its own benchmark bundle.
9. The client app receives the bundle and shows:
   - its own metrics
   - market comparison
   - AI insight

---

## 6. Federated Learning Logic

The federated learning logic lives in `main.py`.

### What model is used?

A simple **LSTM** model is used to forecast demand.

### What does each client contribute?

Each client contributes its own demand time series.

The loader requires these columns:

- `demand`
- `disruption_prob`
- `emission_factor`

Other columns may exist, but these are the minimum required for the forecasting and optimization flow.

### How does training happen?

For each round:

1. The control tower broadcasts the current global model weights.
2. Each client trains a local copy of the model on its own data.
3. Differential privacy noise is added to local weight updates.
4. The control tower averages the client weights using **FedAvg**.
5. The global model is updated.

### What is predicted?

The model predicts the next demand value using the previous 5 points of demand history.

---

## 7. Optimization Logic

After forecasting demand, the tower computes a suggested order quantity.

The optimization considers:

- forecasted demand
- inventory
- disruption risk
- emission factor
- carbon cap
- selling price
- cost price
- waste cost

### Basic formula

- `safety_stock = forecast * (0.1 + risk)`
- `order_qty = max(0, forecast + safety_stock - inventory)`
- `emissions = order_qty * emission_factor`

It also calculates:

- projected revenue
- order cost
- waste cost
- net profit

---

## 8. Privacy Model

This project tries to preserve the core idea of privacy in federated learning.

### What is private?

- A client should not see another client's raw data
- A client should not get peer-by-peer rankings
- A client should only receive:
  - its own metrics
  - aggregated market benchmarks
  - AI insight built from those two things

### How privacy is preserved

- Client updates are noise-perturbed during federated training
- Market comparison is built from **aggregated values**
- Clients do not receive raw peer rows
- Minimum cohort logic is used before exposing market benchmark data
- Benchmark values are DP-sanitized

### Important practical note

This repo is a **demo** and can run both control tower and client apps on the same PC and same workspace.

That is fine for development and testing.

In a real deployment:

- client apps would run on separate machines
- raw client files would be uploaded to the tower over the secure upload channel
- local data visibility would be restricted by infrastructure, not just app logic

---

## 9. Secure Communication Model

The secure communication logic is in `privacy_network.py`.

### What is secured?

- node health checks
- client benchmark bundle delivery
- raw file uploads from node to tower

### How it works

- Payloads are encrypted using a Fernet key derived from `SC_SHARED_SECRET`
- Envelopes are signed with HMAC-SHA256
- Timestamps are used so stale payloads can be rejected

### Endpoints used

#### Client Node

- `/health`
  The control tower pings this to check if the node is online

- `/sync`
  The control tower pushes the client's private benchmark bundle here

#### Control Tower

- `/health`
  Can be checked like a secure service endpoint

- `/upload`
  Clients securely upload raw `.csv`, `.xls`, or `.xlsx` files here

### Very important port rule

Do **not** confuse:

- the **Streamlit UI port**
- the **secure node endpoint port**

Example:

- Streamlit UI might run on `8503`
- secure node endpoint might run on `8800`

The control tower must talk to the **secure endpoint port**, not the browser UI port.

---

## 10. File Upload Rules

Clients can upload only these file types:

- `.csv`
- `.xls`
- `.xlsx`

The upload is rejected for any other type.

Uploaded files are saved by the control tower as:

- `DATASETS/client_0__upload__current.csv`
- `DATASETS/client_1__upload__current.xlsx`
- and so on

The next simulation automatically prefers these uploaded files for that client.

---

## 11. Dataset Requirements

The minimum required columns are:

- `demand`
- `disruption_prob`
- `emission_factor`

Optional useful columns include:

- `week`
- `date`
- `brand`
- `region`
- `supply`
- `inventory_level`
- `profit_margin`

### Example CSV

```csv
week,demand,disruption_prob,emission_factor
0,100,0.20,1.50
1,120,0.18,1.55
2,115,0.25,1.60
3,130,0.22,1.48
4,140,0.19,1.52
5,150,0.17,1.50
```

---

## 12. Installation

Use Python 3.10+ if possible.

### Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `pandas`
- `numpy`
- `streamlit`
- `matplotlib`
- `cryptography`
- `openpyxl`
- `xlrd`

---

## 13. Model Requirement

The local AI features need a GGUF model file.

Expected default location:

```text
models/SupplyChain-Qwen.gguf
```

You also need the llama.cpp runtime already bundled in:

```text
Llama-cpp-cpu/
```

The app tries:

- `llama-cpp-python` first
- then bundled `llama-completion.exe` / `llama-cli.exe`

---

## 14. Commands to Run Everything

## A. Run the Control Tower

```bash
python -m streamlit run app.py
```

Default Streamlit UI:

```text
http://localhost:8501
```

This app also starts a **secure tower endpoint** by default on:

```text
http://127.0.0.1:9900
```

You can change that from the sidebar.

---

## B. Run One Client Node

Example for client `0`:

### Windows CMD

```cmd
set SC_CLIENT_ID=0
set SC_NODE_PORT=8800
python -m streamlit run client_app.py --server.port 8503
```

### PowerShell

```powershell
$env:SC_CLIENT_ID="0"
$env:SC_NODE_PORT="8800"
python -m streamlit run client_app.py --server.port 8503
```

In this example:

- Streamlit UI is on `8503`
- secure node endpoint is on `8800`

---

## C. Run a Second Client Node

Example for client `1`:

### Windows CMD

```cmd
set SC_CLIENT_ID=1
set SC_NODE_PORT=8801
python -m streamlit run client_app.py --server.port 8504
```

### PowerShell

```powershell
$env:SC_CLIENT_ID="1"
$env:SC_NODE_PORT="8801"
python -m streamlit run client_app.py --server.port 8504
```

---

## D. Run a Third Client Node

Example for client `2`:

### Windows CMD

```cmd
set SC_CLIENT_ID=2
set SC_NODE_PORT=8802
python -m streamlit run client_app.py --server.port 8505
```

### PowerShell

```powershell
$env:SC_CLIENT_ID="2"
$env:SC_NODE_PORT="8802"
python -m streamlit run client_app.py --server.port 8505
```

---

## 15. Recommended Demo Workflow

If you are new, follow this exact order.

### Step 1

Start the control tower:

```bash
python -m streamlit run app.py
```

### Step 2

Start one or more client apps.

Example:

```powershell
$env:SC_CLIENT_ID="0"
$env:SC_NODE_PORT="8800"
python -m streamlit run client_app.py --server.port 8503
```

### Step 3

In the client app:

- go to **Upload Data**
- upload a raw `.csv`, `.xls`, or `.xlsx`
- send it to the control tower

### Step 4

In the control tower:

- go to **Node Network**
- click **Refresh Node Status**
- verify node is online

### Step 5

In the control tower:

- load the local model
- run the federated simulation

### Step 6

In the control tower:

- click **Secure Sync Online Nodes**

### Step 7

In the client app:

- open **Market View**
- see private comparison versus market

### Step 8

Optionally:

- load local model in the client app
- generate **Client AI Insight**

---

## 16. Control Tower Tabs Explained

### Control Center

Used for:

- loading the local model
- running the federated learning simulation
- previewing client data

### Node Network

Used for:

- viewing online/offline node state
- seeing last seen
- seeing last secure sync
- seeing control tower upload status
- syncing bundles to online clients

### Results and Override

Used for:

- viewing forecast metrics
- viewing order recommendation
- viewing emissions and finance
- human approval or override

### AI Assistant

Used for:

- asking strategy questions
- getting local LLM responses

---

## 17. Client App Tabs Explained

### Operations

Shows:

- node status
- last seen
- last secure sync
- local performance metrics

### Upload Data

Used for:

- selecting raw `.csv`, `.xls`, or `.xlsx`
- securely uploading it to the control tower

### Market View

Shows:

- client forecast vs market
- client profit vs market
- client emissions vs market
- client risk vs market

Only privacy-safe aggregate market values are shown.

### AI Insight

Used for:

- loading the local model
- generating private client advice from:
  - that client's own metrics
  - market aggregates only

---

## 18. Environment Variables

You can run the project without setting everything manually, but these variables are useful.

### Shared Security

```text
SC_SHARED_SECRET
```

Used to derive encryption and signing keys for secure communication.

Example:

### PowerShell

```powershell
$env:SC_SHARED_SECRET="my-demo-shared-secret"
```

### Windows CMD

```cmd
set SC_SHARED_SECRET=my-demo-shared-secret
```

### Client Selection

```text
SC_CLIENT_ID
SC_NODE_PORT
SC_NODE_HOST
```

### Tower Endpoint

```text
SC_TOWER_HOST
SC_TOWER_PORT
```

### Model Paths

```text
SC_LLM_GGUF_PATH
SC_LLAMA_CPP_CLI_PATH
```

---

## 19. Logs and Generated Files

### Important files inside `sc50_logs/`

- `decision_log.json`
  Human approval and override log

- `node_registry.json`
  Known node endpoints and last known state

- `control_tower_endpoint.json`
  Control tower secure upload endpoint info

### Important generated files inside `DATASETS/`

- `client_0__upload__current.csv`
- `client_1__upload__current.xlsx`

These are the latest uploaded raw datasets per client.

---

## 20. Troubleshooting

## Problem: Client is online in browser but tower says node is offline

Most likely reason:

- you used the Streamlit UI port instead of the secure node port

Example of wrong setup:

- client browser UI at `8503`
- tower registry also pointed to `8503`

Correct setup:

- browser UI can stay on `8503`
- secure node endpoint should be `8800`

The app now tries to warn you if you point the tower to a Streamlit page.

---

## Problem: Uploaded file is ignored

Check:

- file extension is `.csv`, `.xls`, or `.xlsx`
- required columns exist
- upload actually succeeded
- you ran a new simulation after uploading

---

## Problem: Model does not load

Check:

- `models/SupplyChain-Qwen.gguf` exists
- `Llama-cpp-cpu/` exists
- dependencies installed

---

## Problem: Excel file fails to load

Check:

- `openpyxl` and `xlrd` are installed
- file is not corrupted
- column names are correct

---

## Problem: Market comparison is unavailable

Possible reason:

- not enough clients for privacy-safe market benchmark

The app can withhold market stats if the cohort is too small.

---

## 21. Beginner Summary

If you want the simplest mental model:

- Clients send raw Excel or CSV to the tower
- Tower trains a shared demand model
- Tower protects privacy using aggregation and differential privacy
- Tower sends back only safe market intelligence
- Clients see their own performance against the market
- AI explains what to do next

---

## 22. Quick Start in 30 Seconds

### Install

```bash
pip install -r requirements.txt
```

### Start tower

```bash
python -m streamlit run app.py
```

### Start one client

```powershell
$env:SC_CLIENT_ID="0"
$env:SC_NODE_PORT="8800"
python -m streamlit run client_app.py --server.port 8503
```

### Then

- upload a raw file from the client
- refresh node status in the tower
- run FL simulation
- sync online nodes
- open Market View in the client app

---

## 23. Final Notes

This project is a demo, but it already includes a strong structure:

- federated learning workflow
- privacy-safe market intelligence
- encrypted and signed communication
- human-in-the-loop override
- local AI insight

It is a good base if you want to extend toward:

- real multi-machine deployment
- stronger authentication
- database-backed storage
- real-time node telemetry
- larger supply chain optimization logic

