import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from dashboard_data import infer_client_catalog, load_node_registry, load_raw_client_frame, save_node_registry
from main import SCConfig, get_device, llm_generate, load_model
from privacy_network import (
    ensure_node_server,
    get_node_server_state,
    get_shared_secret,
    load_control_tower_endpoint,
    load_secure_bundle,
    send_dataset_upload,
)


def apply_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

:root {
    --bg-0: #07121d;
    --bg-1: #0d1c2d;
    --card: rgba(18, 34, 52, 0.92);
    --line: #27425f;
    --text: #eef5fb;
    --muted: #99b3cb;
    --primary: #35c2ff;
    --success: #1ed760;
    --danger: #ff5d6c;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text) !important;
    background: var(--bg-0) !important;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #ffffff !important;
}

.stApp {
    background: radial-gradient(circle at top right, rgba(53,194,255,0.15), transparent 28%), linear-gradient(180deg, var(--bg-0), var(--bg-1));
}

.hero, .panel {
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: 20px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
}

.hero {
    padding: 24px;
    margin-bottom: 1rem;
}

.hero p {
    color: var(--muted) !important;
}

.panel {
    padding: 18px;
    margin-bottom: 1rem;
}

.node-strip {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}

.chip {
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(53,194,255,0.08);
    color: var(--text);
    font-size: 0.86rem;
    font-weight: 700;
}

.status-online { color: var(--success) !important; }
.status-offline { color: var(--danger) !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "client_model": None,
        "client_tokenizer": None,
        "upload_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def bundle_dir_for_client(client_id: str) -> str:
    return os.path.join(SCConfig.LOG_DIR, "client_inbox", f"client_{client_id}")


def format_timestamp(raw_value: Optional[str]) -> str:
    return raw_value if raw_value else "Never"


def build_ai_prompt(bundle: Dict[str, Any]) -> str:
    client = bundle["client"]
    market = bundle["market"].get("kpis", {})
    comparison = bundle.get("comparison", {})
    return (
        f"Client: {client['label']}. "
        f"Forecast next week: {client['forecast_next']}. "
        f"Projected order: {client['projected_order_qty']}. "
        f"Projected emissions: {client['projected_emissions']:.2f}. "
        f"Projected net profit: {client['financials']['net_profit']:.2f}. "
        f"Market average forecast: {market.get('avg_forecast', 0)}. "
        f"Market average emissions: {market.get('avg_projected_emissions', 0)}. "
        f"Market average net profit: {market.get('avg_net_profit', 0)}. "
        f"Delta versus market: {comparison}. "
        "Give a concise client-facing recommendation without mentioning any peer-specific data."
    )


def register_secure_endpoint(client_id: str, node_name: str, host: str, port: int) -> str:
    secure_base_url = f"http://{host}:{port}"
    registry = load_node_registry(dataset_dir=SCConfig.DATASET_DIR, num_clients=SCConfig.NUM_CLIENTS)
    updated_registry = []
    changed = False

    for node in registry:
        updated_node = dict(node)
        if str(updated_node.get("client_id")) == str(client_id):
            if updated_node.get("base_url") != secure_base_url:
                updated_node["base_url"] = secure_base_url
                changed = True
            if updated_node.get("node_name") != node_name:
                updated_node["node_name"] = node_name
                changed = True
        updated_registry.append(updated_node)

    if changed:
        save_node_registry(updated_registry)

    return secure_base_url


st.set_page_config(
    page_title="Supply Chain 5.0 - Client Node",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
init_session_state()

catalog = infer_client_catalog()
client_ids = [item["client_id"] for item in catalog]
default_client_id = os.getenv("SC_CLIENT_ID", client_ids[0] if client_ids else "0")
default_port = int(os.getenv("SC_NODE_PORT", str(8800 + int(default_client_id))))
default_tower = load_control_tower_endpoint() or {}

st.sidebar.header("Client Node Settings")
selected_client_id = st.sidebar.selectbox(
    "Client Identity",
    options=client_ids,
    index=client_ids.index(default_client_id) if default_client_id in client_ids else 0,
)
selected_meta = next(item for item in catalog if item["client_id"] == selected_client_id)
node_port = st.sidebar.number_input("Secure Node Port", min_value=1024, max_value=65535, value=default_port, step=1)
node_host = st.sidebar.text_input("Node Host", value=os.getenv("SC_NODE_HOST", "127.0.0.1"))
tower_host = st.sidebar.text_input("Tower Secure Host", value=os.getenv("SC_TOWER_HOST", default_tower.get("host", "127.0.0.1")))
tower_port = st.sidebar.number_input(
    "Tower Secure Port",
    min_value=1024,
    max_value=65535,
    value=int(os.getenv("SC_TOWER_PORT", str(default_tower.get("port", 9900)))),
    step=1,
)
st.sidebar.caption("Use the same host and port in the control tower node registry.")
st.sidebar.caption("Secure sync uses signed and encrypted payloads keyed by `SC_SHARED_SECRET`.")

server_result = ensure_node_server(
    node_id=f"node-{selected_client_id}",
    node_name=selected_meta["node_name"],
    client_id=selected_client_id,
    port=int(node_port),
    bundle_dir=bundle_dir_for_client(selected_client_id),
    host=node_host,
    secret=get_shared_secret(),
)

if not server_result["ok"]:
    st.sidebar.error(f"Node server failed to start: {server_result['error']}")
node_state = get_node_server_state(int(node_port)) if server_result["ok"] else None
registered_endpoint = register_secure_endpoint(selected_client_id, selected_meta["node_name"], node_host, int(node_port))
st.sidebar.code(f"Secure node endpoint: {registered_endpoint}")
st.sidebar.code(f"Tower upload endpoint: http://{tower_host}:{int(tower_port)}")

bundle = load_secure_bundle(bundle_dir_for_client(selected_client_id), secret=get_shared_secret())
local_df = load_raw_client_frame(selected_client_id)
latest_local = local_df.iloc[-1]

model_state = "Loaded" if st.session_state.client_model else "Not loaded"
sync_state = "Secure bundle ready" if bundle else "Awaiting control tower sync"
status_class = "status-online" if node_state else "status-offline"
status_text = "Online" if node_state else "Offline"

st.markdown(
    f"""
<div class="hero">
  <h1>{selected_meta['label']} Client Dashboard</h1>
  <p>Private client operations view with market benchmarking sourced only from secure control-tower aggregates.</p>
  <div class="node-strip">
    <span class="chip">Node Status: <span class="{status_class}">{status_text}</span></span>
    <span class="chip">Endpoint: http://{node_host}:{int(node_port)}</span>
    <span class="chip">Streamlit UI: browser only</span>
    <span class="chip">Model: {model_state}</span>
    <span class="chip">Secure Sync: {sync_state}</span>
    <span class="chip">Device: {get_device()}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_ops, tab_upload, tab_market, tab_ai = st.tabs(["Operations", "Upload Data", "Market View", "AI Insight"])

with tab_ops:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Node Presence")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_text)
    col2.metric("Last Seen", format_timestamp(node_state.get("last_seen") if node_state else None))
    col3.metric("Last Secure Sync", format_timestamp(node_state.get("last_sync_at") if node_state else None))
    st.caption("The control tower uses the secure node endpoint for `/health` and `/sync`, not the Streamlit UI port in the browser.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Local Performance")
    ops_a, ops_b, ops_c, ops_d = st.columns(4)
    ops_a.metric("Latest Demand", f"{int(latest_local['demand'])}")
    ops_b.metric("Latest Supply", f"{int(latest_local.get('supply', latest_local['demand']))}")
    ops_c.metric("Inventory", f"{int(latest_local.get('inventory_level', 0))}")
    ops_d.metric("Profit Margin", f"{float(latest_local.get('profit_margin', 0)):.2%}")
    history_chart = local_df.tail(24)[["week", "demand"]].set_index("week")
    st.line_chart(history_chart)
    st.caption("This chart is built only from the client's own local dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_market:
    if not bundle:
        st.info("No secure benchmark bundle has been synced yet. Start the control tower, run a simulation, and push client bundles from the Node Network tab.")
    else:
        client = bundle["client"]
        market = bundle["market"]
        comparison = bundle["comparison"]

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Market Comparison")
        if not market.get("available"):
            st.warning(market.get("reason", "Market benchmark unavailable."))
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Your Forecast", f"{client['forecast_next']}", delta=f"{comparison['forecast_vs_market']:+.2f} vs market")
            m2.metric("Your Profit", f"${client['financials']['net_profit']:.2f}", delta=f"{comparison['profit_vs_market']:+.2f} vs market")
            m3.metric("Your Emissions", f"{client['projected_emissions']:.2f}", delta=f"{comparison['emissions_vs_market']:+.2f} vs market")
            m4.metric("Your Risk", f"{client['risk_level']:.3f}", delta=f"{comparison['risk_vs_market']:+.4f} vs market")

            demand_history = pd.DataFrame(client["history"])[["week", "demand"]].rename(columns={"demand": "Client Demand"})
            market_history = pd.DataFrame(market.get("history", []))
            if not market_history.empty:
                merged = demand_history.merge(market_history, on="week", how="left").set_index("week")
                st.line_chart(merged.rename(columns={"market_avg_demand": "Market Avg Demand"}))
            else:
                st.line_chart(demand_history.set_index("week"))

            st.caption(
                f"Market benchmark uses aggregated data only across {market['client_count']} clients with DP epsilon {market['privacy']['dp_epsilon']}."
            )

            band_a, band_b = st.columns(2)
            growth_band = market["kpis"]["demand_growth_band"]
            profit_band = market["kpis"]["profit_band"]
            band_a.metric("Market Growth Band", f"{growth_band['low']:.2f}% to {growth_band['high']:.2f}%")
            band_b.metric("Market Profit Band", f"${profit_band['low']:.2f} to ${profit_band['high']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_upload:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Secure Raw File Upload")
    st.caption("Only raw `.csv`, `.xls`, and `.xlsx` files are accepted. The file is encrypted and sent to the control tower secure endpoint.")

    uploaded_file = st.file_uploader(
        "Choose raw dataset file",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key=f"raw_upload_{selected_client_id}",
    )
    if uploaded_file is not None:
        st.write(f"Selected file: `{uploaded_file.name}`")
        st.write(f"Size: `{uploaded_file.size}` bytes")

    if st.button("Send Raw File to Control Tower", use_container_width=True, disabled=uploaded_file is None):
        with st.spinner("Encrypting and uploading file..."):
            try:
                response = send_dataset_upload(
                    base_url=f"http://{tower_host}:{int(tower_port)}",
                    client_id=selected_client_id,
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    sender=f"client-node-{selected_client_id}",
                    secret=get_shared_secret(),
                    timeout=10.0,
                )
                st.session_state.upload_result = response
                st.success(f"Upload complete. Saved at `{response['saved_path']}`")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.upload_result:
        result = st.session_state.upload_result
        st.info(
            f"Last upload received at {result.get('received_at', 'unknown time')} with size "
            f"{result.get('size_bytes', 'unknown')} bytes."
        )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_ai:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Private AI Insight")
    st.caption("The prompt includes only this client's data and privacy-safe market aggregates.")

    if st.button("Load Local Model for Client Insight", use_container_width=True):
        with st.spinner("Loading local model..."):
            try:
                tokenizer, model = load_model()
                st.session_state.client_tokenizer = tokenizer
                st.session_state.client_model = model
                st.success("Client model loaded.")
            except Exception as exc:
                st.error(f"Model loading failed: {exc}")

    if bundle and st.session_state.client_model and st.button("Generate Client AI Insight", use_container_width=True):
        prompt = build_ai_prompt(bundle)
        with st.spinner("Generating guidance..."):
            try:
                response = llm_generate(
                    [
                        {
                            "role": "system",
                            "content": "You are a supply chain advisor. Never mention peer-specific data. Use only the client's metrics and market aggregates supplied to you.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    st.session_state.client_tokenizer,
                    st.session_state.client_model,
                    max_tokens=180,
                )
                st.success(response)
            except Exception as exc:
                st.error(f"Insight generation failed: {exc}")
    elif not bundle:
        st.info("Wait for a secure bundle from the control tower before generating market-aware AI insight.")
    elif not st.session_state.client_model:
        st.info("Load the local model to generate client-facing AI insight.")

    st.markdown("</div>", unsafe_allow_html=True)
