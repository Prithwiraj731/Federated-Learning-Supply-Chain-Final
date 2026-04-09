import json
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard_data import build_client_bundles, infer_client_catalog, load_node_registry, save_node_registry
from main import FedSim, SCConfig, SupplyChainDataManager, get_device, llm_generate, load_model, to_serializable
from privacy_network import (
    ensure_control_tower_server,
    get_control_tower_server_state,
    get_shared_secret,
    ping_node,
    sync_bundle,
)


NODE_ICON_SVG = """
<svg width="52" height="52" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="3" y="4" width="18" height="12" rx="2.2" stroke="#8bd2ff" stroke-width="1.5"/>
  <path d="M8 20H16" stroke="#8bd2ff" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M10 16L9 20" stroke="#8bd2ff" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M14 16L15 20" stroke="#8bd2ff" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="18" cy="7" r="1.3" fill="#35d07f"/>
</svg>
"""


st.set_page_config(
    page_title="Supply Chain 5.0 - Control Tower",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

:root {
    --bg-0: #0b0f19;
    --bg-1: #101624;
    --card: #151c2e;
    --card-hover: #1a2338;
    --line: #263354;
    --text: #e8ecf4;
    --muted: #94a3c0;
    --primary: #6c5ce7;
    --primary-2: #7c6ff7;
    --accent: #f0b429;
    --success: #00e676;
    --danger: #ff5252;
    --glow: rgba(108, 92, 231, 0.35);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text) !important;
    background-color: var(--bg-0) !important;
}

p, span, label, div {
    color: var(--text) !important;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.02em;
    color: #ffffff !important;
}

.stApp {
    background: linear-gradient(165deg, var(--bg-0) 0%, var(--bg-1) 50%, #0d1225 100%);
}

section[data-testid="stSidebar"] {
    background: #0e1322 !important;
    border-right: 1px solid var(--line) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: var(--text) !important;
}

section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label {
    color: var(--muted) !important;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.hero {
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 22px 24px;
    background: linear-gradient(135deg, #151c2e 0%, #1a2540 100%);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.05rem;
    background: linear-gradient(135deg, #c4b5fd, #6c5ce7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    margin: 0.45rem 0 0;
    color: var(--muted) !important;
}

.pill-row {
    margin-top: 0.8rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.pill {
    border: 1px solid var(--line);
    background: rgba(108, 92, 231, 0.12);
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 600;
    padding: 0.35rem 0.7rem;
    color: #c4b5fd !important;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 18px;
    background: var(--card);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    animation: riseIn .42s ease;
    margin-bottom: 1rem;
}

.node-card {
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 18px;
    background: linear-gradient(180deg, rgba(21,28,46,0.98), rgba(17,24,39,0.98));
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
    min-height: 230px;
}

.node-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.8rem;
}

.node-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.04rem;
    font-weight: 700;
}

.node-muted {
    color: var(--muted) !important;
    font-size: 0.9rem;
}

.node-status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 700;
}

.node-status.online {
    background: rgba(0, 230, 118, 0.14);
    color: var(--success) !important;
}

.node-status.offline {
    background: rgba(255, 82, 82, 0.14);
    color: var(--danger) !important;
}

.node-metric {
    margin-top: 0.85rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(148, 163, 192, 0.14);
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0px); }
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid transparent;
    background: linear-gradient(135deg, var(--primary), #8b5cf6);
    color: #ffffff !important;
    font-weight: 700;
    letter-spacing: 0.01em;
    transition: all 0.22s ease;
    box-shadow: 0 6px 20px var(--glow);
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, var(--primary-2), #9d7ff7);
    box-shadow: 0 10px 28px rgba(108, 92, 231, 0.5);
}

div[data-testid="stMetric"] {
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: 14px;
    padding: 0.35rem 0.55rem;
}

div[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
}

div[data-baseweb="tab-list"] {
    gap: 0.4rem;
}

div[data-baseweb="tab-list"] button {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: var(--text) !important;
    transition: all 0.2s ease;
}

div[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #c4b5fd !important;
    border-color: var(--primary);
}

div[data-baseweb="tab-highlight"] {
    background: var(--primary);
    height: 3px;
    border-radius: 999px;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #1a2338 !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

.stSelectbox > div > div,
div[data-baseweb="select"] > div {
    background-color: #1a2338 !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

.stExpander {
    border-color: var(--line) !important;
    background-color: var(--card) !important;
}

.stChatInput > div {
    background-color: #1a2338 !important;
    border-color: var(--line) !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), #a78bfa) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def plot_financial_pie(financials: Dict[str, float]):
    sizes = [
        max(0, financials["net_profit"]),
        financials["order_cost"],
        financials["waste_cost"],
    ]
    labels = ["Net Profit", "Order Cost", "Waste Risk"]
    colors = ["#16a34a", "#f59e0b", "#ef4444"]
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=100,
    )
    ax.axis("equal")
    return fig


def format_timestamp(raw_value: Optional[str]) -> str:
    return raw_value if raw_value else "Never"


def init_session_state() -> None:
    defaults = {
        "model": None,
        "tokenizer": None,
        "simulation_done": False,
        "opt_result": None,
        "forecast": None,
        "metrics": None,
        "messages": [],
        "run_simulation": False,
        "data_manager": None,
        "client_bundles": {},
        "market_snapshot": None,
        "primary_client_summary": None,
        "node_registry": [],
        "sync_results": [],
        "control_tower_server": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_or_create_data_manager():
    data_manager = st.session_state.get("data_manager")
    if data_manager is None or data_manager.num_clients != SCConfig.NUM_CLIENTS:
        data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)
        st.session_state.data_manager = data_manager
    return data_manager


def refresh_node_statuses() -> List[Dict[str, Any]]:
    registry = load_node_registry(dataset_dir=SCConfig.DATASET_DIR, num_clients=SCConfig.NUM_CLIENTS)
    refreshed: List[Dict[str, Any]] = []
    secret = get_shared_secret()
    for node in registry:
        status = ping_node(node["base_url"], secret=secret)
        merged = dict(node)
        if status["online"]:
            merged["status"] = "online"
            merged["last_seen"] = status.get("last_seen")
            merged["latency_ms"] = status.get("latency_ms")
            merged["last_sync_at"] = status.get("last_sync_at") or node.get("last_sync_at")
            merged["error"] = None
        else:
            merged["status"] = "offline"
            merged["latency_ms"] = None
            merged["error"] = status.get("error")
        refreshed.append(merged)
    save_node_registry(refreshed)
    st.session_state.node_registry = refreshed
    return refreshed


def sync_online_nodes() -> List[Dict[str, Any]]:
    registry = st.session_state.get("node_registry") or refresh_node_statuses()
    bundles = st.session_state.get("client_bundles", {})
    secret = get_shared_secret()
    results: List[Dict[str, Any]] = []
    updated_registry: List[Dict[str, Any]] = []

    for node in registry:
        updated_node = dict(node)
        client_id = str(node["client_id"])
        bundle = bundles.get(client_id)
        if node.get("status") != "online":
            results.append({"node_name": node["node_name"], "status": "skipped", "detail": "Node is offline."})
            updated_registry.append(updated_node)
            continue
        if not bundle:
            results.append({"node_name": node["node_name"], "status": "skipped", "detail": "No bundle available for this client."})
            updated_registry.append(updated_node)
            continue

        try:
            response = sync_bundle(node["base_url"], bundle=bundle, secret=secret)
            updated_node["last_sync_at"] = response.get("last_sync_at", updated_node.get("last_sync_at"))
            updated_node["status"] = "online"
            updated_node["error"] = None
            results.append({"node_name": node["node_name"], "status": "synced", "detail": updated_node["last_sync_at"]})
        except Exception as exc:
            updated_node["status"] = "offline"
            updated_node["error"] = str(exc)
            results.append({"node_name": node["node_name"], "status": "failed", "detail": str(exc)})
        updated_registry.append(updated_node)

    save_node_registry(updated_registry)
    st.session_state.node_registry = updated_registry
    st.session_state.sync_results = results
    return results


def render_node_card(node: Dict[str, Any]) -> str:
    status_class = "online" if node.get("status") == "online" else "offline"
    status_label = "Online" if node.get("status") == "online" else "Offline"
    latency = f"{node['latency_ms']} ms" if node.get("latency_ms") is not None else "--"
    error_text = node.get("error") or "Secure channel healthy."
    return f"""
<div class="node-card">
  <div class="node-card-header">
    <div>
      <div class="node-title">{node['node_name']}</div>
      <div class="node-muted">{node['label']}</div>
    </div>
    <div>{NODE_ICON_SVG}</div>
  </div>
  <div class="node-metric">
    <span class="node-status {status_class}">{status_label}</span>
  </div>
  <div class="node-metric">
    <div class="node-muted">Endpoint</div>
    <div>{node['base_url']}</div>
  </div>
  <div class="node-metric">
    <div class="node-muted">Last Seen</div>
    <div>{format_timestamp(node.get('last_seen'))}</div>
  </div>
  <div class="node-metric">
    <div class="node-muted">Latency</div>
    <div>{latency}</div>
  </div>
  <div class="node-metric">
    <div class="node-muted">Last Secure Sync</div>
    <div>{format_timestamp(node.get('last_sync_at'))}</div>
  </div>
  <div class="node-metric">
    <div class="node-muted">Channel Note</div>
    <div>{error_text}</div>
  </div>
</div>
"""


def run_simulation_pipeline() -> None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Loading client data from DATASETS...")
        data_manager = SupplyChainDataManager(SCConfig.NUM_CLIENTS)
        st.session_state.data_manager = data_manager
        progress_bar.progress(15)

        status_text.text(f"Running federated learning for {SCConfig.NUM_ROUNDS} rounds...")
        fed = FedSim(data_manager)
        lstm_model, metrics = fed.run(
            st.session_state.tokenizer,
            st.session_state.model,
            epsilon=SCConfig.DP_EPSILON,
        )
        progress_bar.progress(75)

        status_text.text("Building privacy-safe market bundles for each node...")
        client_bundles = build_client_bundles(
            data_manager=data_manager,
            lstm_model=lstm_model,
            scale=fed.max_val,
            inventory=50,
            epsilon=SCConfig.DP_EPSILON,
        )
        primary_bundle = client_bundles.get("0")
        if not primary_bundle:
            raise RuntimeError("Primary client bundle could not be generated.")

        primary_client = primary_bundle["client"]
        opt_result = {
            "optimized_qty": primary_client["projected_order_qty"],
            "emissions": primary_client["projected_emissions"],
            "feasible": primary_client["feasible"],
            "financials": primary_client["financials"],
        }

        st.session_state.forecast = primary_client["forecast_next"]
        st.session_state.opt_result = opt_result
        st.session_state.metrics = metrics
        st.session_state.simulation_done = True
        st.session_state.client_bundles = client_bundles
        st.session_state.market_snapshot = primary_bundle["market"]
        st.session_state.primary_client_summary = primary_client

        progress_bar.progress(100)
        status_text.text("Simulation complete.")
        st.success("Federated round finished. Results are ready and secure client bundles can now be synced.")
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
    finally:
        st.session_state.run_simulation = False


init_session_state()

st.sidebar.header("Control Settings")
st.sidebar.caption("Tune the scenario, refresh node health, and rerun federated rounds.")
st.sidebar.info(f"Product locked: {SCConfig.PRODUCT_NAME}")

SCConfig.NUM_CLIENTS = st.sidebar.slider("Number of Clients", 1, 10, SCConfig.NUM_CLIENTS)
SCConfig.NUM_ROUNDS = st.sidebar.slider("Federated Rounds", 1, 10, SCConfig.NUM_ROUNDS)
SCConfig.CARBON_CAP = st.sidebar.number_input("Carbon Cap", value=SCConfig.CARBON_CAP)
SCConfig.DP_EPSILON = st.sidebar.slider(
    "DP Epsilon",
    0.1,
    20.0,
    SCConfig.DP_EPSILON,
    help="Lower values improve privacy with more injected noise.",
)
SCConfig.LOG_DIR = st.sidebar.text_input("Log Directory", SCConfig.LOG_DIR)
tower_host = st.sidebar.text_input("Tower Secure Host", value=os.getenv("SC_TOWER_HOST", "127.0.0.1"))
tower_port = st.sidebar.number_input(
    "Tower Secure Port",
    min_value=1024,
    max_value=65535,
    value=int(os.getenv("SC_TOWER_PORT", "9900")),
    step=1,
)

tower_server_result = ensure_control_tower_server(
    port=int(tower_port),
    dataset_dir=SCConfig.DATASET_DIR,
    host=tower_host,
    secret=get_shared_secret(),
)
if not tower_server_result["ok"]:
    st.sidebar.error(f"Tower secure endpoint failed to start: {tower_server_result['error']}")
    tower_state = None
else:
    tower_state = get_control_tower_server_state(int(tower_port))
    st.session_state.control_tower_server = tower_state

catalog = infer_client_catalog(dataset_dir=SCConfig.DATASET_DIR, num_clients=SCConfig.NUM_CLIENTS)
st.sidebar.caption(f"Secure registry secret source: {'custom env' if os.getenv('SC_SHARED_SECRET') else 'default demo secret'}")
if tower_state:
    st.sidebar.code(f"Tower upload endpoint: http://{tower_host}:{int(tower_port)}")

if not st.session_state.node_registry:
    st.session_state.node_registry = load_node_registry(dataset_dir=SCConfig.DATASET_DIR, num_clients=SCConfig.NUM_CLIENTS)

model_state = "Active" if st.session_state.model else "Not loaded"
online_nodes = sum(1 for node in st.session_state.node_registry if node.get("status") == "online")
tower_status = "Online" if tower_state else "Unavailable"

st.markdown(
    f"""
<div class="hero">
  <h1>Supply Chain Control Tower</h1>
  <p>Federated planning for milk demand, secure node visibility, human override, and privacy-safe client intelligence.</p>
  <div class="pill-row">
    <span class="pill">Device: {get_device()}</span>
    <span class="pill">Model: {SCConfig.MODEL_NAME}</span>
    <span class="pill">Model Status: {model_state}</span>
    <span class="pill">Dataset Source: DATASETS</span>
    <span class="pill">Online Nodes: {online_nodes}/{len(st.session_state.node_registry)}</span>
    <span class="pill">Tower Upload Endpoint: {tower_status}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


tab_control, tab_nodes, tab_results, tab_chat = st.tabs(
    ["Control Center", "Node Network", "Results and Override", "AI Assistant"]
)


with tab_control:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Model Station")
        st.caption("Load your local GGUF model through llama.cpp CPU runtime.")

        if st.button("Load Local Model", use_container_width=True):
            with st.spinner("Loading local model..."):
                try:
                    tokenizer, model = load_model()
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.success("Model loaded and ready.")
                except Exception as exc:
                    st.error(f"Model loading failed: {exc}")

        if st.session_state.model:
            st.success("Model is active for AI Insight, chat, and bundle generation.")
        else:
            st.warning("Load model first to enable AI explanation and chat.")

    with right:
        st.subheader("Simulation Runner")
        st.caption("Run federated simulation and prepare privacy-safe benchmark bundles for each node.")

        if st.button(
            "Run Federated Simulation",
            use_container_width=True,
            disabled=st.session_state.model is None,
        ):
            st.session_state.run_simulation = True

        st.write(
            f"Current setup: {SCConfig.NUM_CLIENTS} clients, {SCConfig.NUM_ROUNDS} rounds, "
            f"epsilon={SCConfig.DP_EPSILON:.2f}, carbon cap={SCConfig.CARBON_CAP:.1f}."
        )

        data_manager = get_or_create_data_manager()
        preview_df = data_manager.get_client_data("0").head(5)
        with st.expander("Preview Client 0 Data", expanded=False):
            st.dataframe(preview_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("run_simulation", False):
        run_simulation_pipeline()

    if st.session_state.simulation_done and st.session_state.metrics:
        metrics = st.session_state.metrics
        k1, k2, k3 = st.columns(3)
        k1.metric("Final MAE", f"{metrics['mae'][-1]:.2f}")
        k2.metric("Final RMSE", f"{metrics['rmse'][-1]:.2f}")
        k3.metric("Rounds", f"{len(metrics['rounds'])}")


with tab_nodes:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Secure Node Visibility")
    st.caption("Client nodes expose signed `/health`, receive encrypted benchmark bundles over `/sync`, and send raw CSV/XLS/XLSX uploads to the tower over `/upload`.")

    action_left, action_right = st.columns([1, 1])
    if action_left.button("Refresh Node Status", use_container_width=True):
        refresh_node_statuses()
        st.success("Node status refreshed.")
    if action_right.button(
        "Secure Sync Online Nodes",
        use_container_width=True,
        disabled=not st.session_state.get("client_bundles"),
    ):
        sync_results = sync_online_nodes()
        success_count = sum(1 for item in sync_results if item["status"] == "synced")
        st.success(f"Secure bundle sync complete. Successful deliveries: {success_count}.")

    registry = st.session_state.get("node_registry") or refresh_node_statuses()
    online_count = sum(1 for node in registry if node.get("status") == "online")
    last_sync_count = sum(1 for node in registry if node.get("last_sync_at"))
    uploads_received = int((tower_state or {}).get("uploads_received", 0))
    m1, m2, m3 = st.columns(3)
    m1.metric("Registered Nodes", len(registry))
    m2.metric("Online Nodes", online_count)
    m3.metric("Nodes Synced", last_sync_count)
    st.metric("Uploads Received", uploads_received)
    st.markdown("</div>", unsafe_allow_html=True)

    if tower_state:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Control Tower Upload Endpoint")
        st.write(f"Secure endpoint: `http://{tower_host}:{int(tower_port)}`")
        latest_upload = tower_state.get("last_upload") or {}
        if latest_upload:
            st.success(
                f"Latest upload from client {latest_upload['client_id']}: {latest_upload['file_name']} "
                f"saved to {latest_upload['saved_path']} at {latest_upload['received_at']}."
            )
        else:
            st.info("No client raw-file uploads have been received yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    node_columns = st.columns(3)
    for index, node in enumerate(registry):
        with node_columns[index % 3]:
            st.markdown(render_node_card(node), unsafe_allow_html=True)

    with st.expander("Node Endpoint Settings", expanded=False):
        with st.form("node_registry_form"):
            updated_registry: List[Dict[str, Any]] = []
            for node in registry:
                a, b = st.columns([1.2, 2.2])
                node_name = a.text_input(f"Node Name ({node['client_id']})", value=node["node_name"], key=f"node_name_{node['client_id']}")
                base_url = b.text_input(f"Base URL ({node['client_id']})", value=node["base_url"], key=f"base_url_{node['client_id']}")
                updated_node = dict(node)
                updated_node["node_name"] = node_name
                updated_node["base_url"] = base_url
                updated_registry.append(updated_node)

            if st.form_submit_button("Save Node Registry", use_container_width=True):
                save_node_registry(updated_registry)
                st.session_state.node_registry = updated_registry
                st.success("Node registry saved.")

    if st.session_state.sync_results:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Latest Sync Results")
        st.dataframe(pd.DataFrame(st.session_state.sync_results), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


with tab_results:
    if not (st.session_state.simulation_done and st.session_state.opt_result):
        st.info("Run a simulation from Control Center to unlock charts, finance, and human override.")
    else:
        data_manager = get_or_create_data_manager()
        opt = st.session_state.opt_result
        forecast = st.session_state.forecast
        metrics = st.session_state.metrics
        primary_client = st.session_state.primary_client_summary or {}

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Training Performance")
        train_chart = pd.DataFrame(
            {
                "Round": metrics["rounds"],
                "Training Loss": metrics["loss"],
                "MAE": metrics["mae"],
            }
        )
        st.line_chart(train_chart, x="Round", y=["Training Loss", "MAE"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Demand Outlook")
        client0_df = data_manager.get_client_data("0")
        history_df = client0_df.tail(20).copy()
        chart_data = history_df[["week", "demand"]].set_index("week")
        st.line_chart(chart_data)
        st.caption(f"Last 20 weeks shown. Next week forecast: {forecast} units.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Forecast", f"{forecast} units")
        m2.metric("Recommended Order", f"{opt['optimized_qty']} units")
        m3.metric("Projected Emissions", f"{opt['emissions']:.2f}")
        m4.metric("Feasible", "Yes" if opt["feasible"] else "No")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Financial Projection")
        fin = opt["financials"]
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.metric("Projected Revenue", f"${fin['revenue']:.2f}")
            st.metric("Order Cost", f"${fin['order_cost']:.2f}")
            st.metric("Waste Cost", f"${fin['waste_cost']:.2f}")
            st.metric("Net Profit", f"${fin['net_profit']:.2f}")
        with c2:
            st.pyplot(plot_financial_pie(fin), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Primary Node Snapshot")
        if primary_client:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Client Label", primary_client["label"])
            p2.metric("Forecast Error", f"{primary_client['forecast_error']:.2f}")
            p3.metric("Risk Level", f"{primary_client['risk_level']:.3f}")
            p4.metric("Emission Factor", f"{primary_client['emission_factor']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("AI Insight")
        if st.button("Generate Strategic Explanation", use_container_width=True):
            with st.spinner("Generating insight from local GGUF model..."):
                try:
                    system_msg = {
                        "role": "system",
                        "content": (
                            f"You are a supply chain expert. Product: {SCConfig.PRODUCT_NAME}. "
                            f"Forecast: {forecast}. Emissions: {opt['emissions']:.2f}."
                        ),
                    }
                    user_msg = {
                        "role": "user",
                        "content": (
                            f"Recommended order quantity is {opt['optimized_qty']}. "
                            "Provide a concise strategic recommendation."
                        ),
                    }
                    insight = llm_generate(
                        [system_msg, user_msg],
                        st.session_state.tokenizer,
                        st.session_state.model,
                        max_tokens=140,
                    )
                    st.success(insight)
                except Exception as exc:
                    st.error(f"AI insight failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Human in the Loop")
        with st.form("override_form"):
            new_qty = st.number_input(
                "Adjust Order Quantity",
                value=int(opt["optimized_qty"]),
                step=1,
            )
            submitted = st.form_submit_button("Approve and Run Next Round")

            if submitted:
                if new_qty != opt["optimized_qty"]:
                    log_entry = {
                        "event": "override",
                        "new": int(new_qty),
                        "original": opt["optimized_qty"],
                        "product": SCConfig.PRODUCT_NAME,
                    }
                    st.warning(f"Order quantity overridden to {new_qty}.")
                else:
                    log_entry = {
                        "event": "approved",
                        "qty": opt["optimized_qty"],
                        "product": SCConfig.PRODUCT_NAME,
                    }
                    st.success("AI recommendation approved.")

                os.makedirs(SCConfig.LOG_DIR, exist_ok=True)
                with open(os.path.join(SCConfig.LOG_DIR, "decision_log.json"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(to_serializable(log_entry)) + "\n")

                st.toast("Decision saved. Running next round...")
                time.sleep(0.8)
                st.session_state.run_simulation = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


with tab_chat:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Supply Chain Assistant")
    st.caption("Ask questions about forecast, ordering strategy, emissions, risk, and client sync readiness.")

    context_note = ""
    if st.session_state.simulation_done and st.session_state.opt_result:
        opt = st.session_state.opt_result
        forecast = st.session_state.forecast
        context_note = (
            f"Context: Forecast={forecast}, Recommended Order={opt['optimized_qty']}, "
            f"Emissions={opt['emissions']:.2f}."
        )
        st.info(context_note)

    chat_container = st.container(height=520)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input(f"Ask about {SCConfig.PRODUCT_NAME} strategy...", key="control_tower_chat")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        if st.session_state.model:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Supply Chain Assistant. "
                        f"{context_note}"
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            try:
                with st.spinner("Thinking..."):
                    response = llm_generate(
                        messages,
                        st.session_state.tokenizer,
                        st.session_state.model,
                        max_tokens=170,
                    )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
        else:
            st.error("Load the model in Control Center first.")

    st.markdown("</div>", unsafe_allow_html=True)
