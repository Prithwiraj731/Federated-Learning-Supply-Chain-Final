import glob
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from main import DifferentialPrivacy, SCConfig, optimize, to_serializable


DEFAULT_INVENTORY = 50
MIN_PRIVACY_COHORT = 3


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _discover_dataset_files(dataset_dir: str) -> List[str]:
    dataset_files = []
    for pattern in ("*.csv", "*.xls", "*.xlsx"):
        dataset_files.extend(glob.glob(os.path.join(dataset_dir, pattern)))
    return sorted(set(dataset_files))


def _map_dataset_files(dataset_dir: str, num_clients: int) -> Dict[str, str]:
    dataset_files = _discover_dataset_files(dataset_dir)
    mapped_files: Dict[str, str] = {}
    remaining_files: List[str] = []

    for file_path in dataset_files:
        file_name = os.path.basename(file_path).lower()
        upload_match = re.search(r"client_(\d+)__upload__current\.(csv|xls|xlsx)$", file_name)
        if upload_match:
            mapped_files[upload_match.group(1)] = file_path
        else:
            remaining_files.append(file_path)

    unassigned_clients = [str(cid) for cid in range(num_clients) if str(cid) not in mapped_files]
    for cid, file_path in zip(unassigned_clients, remaining_files):
        mapped_files[cid] = file_path

    return mapped_files


def node_registry_path() -> str:
    return os.path.join(SCConfig.LOG_DIR, "node_registry.json")


def infer_client_catalog(dataset_dir: str = SCConfig.DATASET_DIR, num_clients: int = SCConfig.NUM_CLIENTS) -> List[Dict[str, Any]]:
    dataset_map = _map_dataset_files(dataset_dir=dataset_dir, num_clients=num_clients)
    catalog: List[Dict[str, Any]] = []

    for cid in range(num_clients):
        entry = {
            "client_id": str(cid),
            "brand": f"Client {cid}",
            "region": "Synthetic",
            "label": f"Client {cid}",
            "dataset_path": "",
            "node_name": f"Node {cid}",
        }
        dataset_path = dataset_map.get(str(cid), "")
        if dataset_path:
            entry["dataset_path"] = dataset_path
            try:
                preview = pd.read_csv(dataset_path, nrows=1)
                if not preview.empty:
                    row = preview.iloc[0]
                    brand = str(row.get("brand", f"Client {cid}")).strip()
                    region = str(row.get("region", "Region")).strip()
                    entry["brand"] = brand
                    entry["region"] = region
                    entry["label"] = f"{brand} - {region}"
                    entry["node_name"] = f"{brand} Edge Node"
            except Exception:
                try:
                    preview = pd.read_excel(dataset_path, nrows=1)
                    if not preview.empty:
                        row = preview.iloc[0]
                        brand = str(row.get("brand", f"Client {cid}")).strip()
                        region = str(row.get("region", "Region")).strip()
                        entry["brand"] = brand
                        entry["region"] = region
                        entry["label"] = f"{brand} - {region}"
                        entry["node_name"] = f"{brand} Edge Node"
                    else:
                        entry["label"] = os.path.splitext(os.path.basename(dataset_path))[0].replace("_", " ").title()
                        entry["node_name"] = f"{entry['label']} Node"
                except Exception:
                    entry["label"] = os.path.splitext(os.path.basename(dataset_path))[0].replace("_", " ").title()
                    entry["node_name"] = f"{entry['label']} Node"
        catalog.append(entry)
    return catalog


def load_raw_client_frame(client_id: str, dataset_dir: str = SCConfig.DATASET_DIR, num_clients: int = SCConfig.NUM_CLIENTS) -> pd.DataFrame:
    catalog = infer_client_catalog(dataset_dir=dataset_dir, num_clients=num_clients)
    match = next((item for item in catalog if item["client_id"] == str(client_id)), None)
    if not match or not match.get("dataset_path"):
        raise FileNotFoundError(f"No dataset file mapped to client {client_id}.")
    suffix = os.path.splitext(match["dataset_path"])[1].lower()
    if suffix == ".csv":
        return pd.read_csv(match["dataset_path"])
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(match["dataset_path"])
    raise ValueError(f"Unsupported dataset file type: {match['dataset_path']}")


def _predict_from_sequence(lstm_model, sequence: np.ndarray, scale: float) -> int:
    if len(sequence) == 0:
        return 0
    normalized = sequence.astype(np.float32) / max(scale, 1.0)
    inp = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    lstm_model.eval()
    with torch.no_grad():
        pred_norm = float(lstm_model(inp).item())
    return max(0, int(pred_norm * max(scale, 1.0)))


def compute_client_metrics(
    data_manager,
    lstm_model,
    scale: float,
    inventory: int = DEFAULT_INVENTORY,
    sequence_length: int = 5,
) -> List[Dict[str, Any]]:
    catalog = infer_client_catalog(dataset_dir=data_manager.dataset_dir, num_clients=data_manager.num_clients)
    metrics: List[Dict[str, Any]] = []

    for cid in range(data_manager.num_clients):
        client_id = str(cid)
        df = data_manager.get_client_data(client_id).copy()
        demand = df["demand"].astype(float).to_numpy()
        last_known = int(demand[-1]) if len(demand) else 0

        if len(demand) >= sequence_length + 1:
            observed_pred = _predict_from_sequence(lstm_model, demand[-sequence_length - 1:-1], scale)
            forecast_next = _predict_from_sequence(lstm_model, demand[-sequence_length:], scale)
            forecast_error = abs(last_known - observed_pred)
        else:
            observed_pred = last_known
            forecast_next = last_known
            forecast_error = 0.0

        last_row = df.iloc[-1]
        optimization = optimize(
            forecast=forecast_next,
            inventory=inventory,
            emission_factor=float(last_row["emission_factor"]),
            risk=float(last_row["disruption_prob"]),
        )

        trailing_demand = df["demand"].tail(12)
        recent_avg = float(trailing_demand.mean()) if not trailing_demand.empty else 0.0
        recent_risk = float(df["disruption_prob"].tail(8).mean()) if len(df) else 0.0
        recent_emission = float(df["emission_factor"].tail(8).mean()) if len(df) else 0.0
        recent_window = df["demand"].tail(4)
        prior_window = df["demand"].tail(8).head(4)
        prior_avg = float(prior_window.mean()) if not prior_window.empty else 0.0
        demand_growth_pct = 0.0 if prior_avg == 0 else ((float(recent_window.mean()) - prior_avg) / prior_avg) * 100

        history = df.tail(24)[["week", "demand", "disruption_prob", "emission_factor"]].copy().fillna(0)
        meta = catalog[cid] if cid < len(catalog) else {
            "client_id": client_id,
            "label": f"Client {client_id}",
            "brand": f"Client {client_id}",
            "region": "Unknown",
            "node_name": f"Node {client_id}",
        }
        metrics.append(
            {
                "client_id": client_id,
                "label": meta["label"],
                "brand": meta["brand"],
                "region": meta["region"],
                "node_name": meta["node_name"],
                "latest_demand": last_known,
                "avg_demand_12w": recent_avg,
                "demand_growth_pct": demand_growth_pct,
                "forecast_next": forecast_next,
                "observed_prediction": observed_pred,
                "forecast_error": forecast_error,
                "risk_level": recent_risk,
                "emission_factor": recent_emission,
                "projected_order_qty": optimization["optimized_qty"],
                "projected_emissions": optimization["emissions"],
                "feasible": optimization["feasible"],
                "financials": optimization["financials"],
                "history": history.to_dict(orient="records"),
            }
        )

    return metrics


def _safe_market_value(values: List[float], epsilon: float, precision: int = 2) -> float:
    if not values:
        return 0.0
    base_value = float(np.mean(values))
    noisy = DifferentialPrivacy.add_noise(base_value, epsilon=max(epsilon, 0.1), sensitivity=1.0)
    return round(noisy, precision)


def _safe_market_band(values: List[float], epsilon: float, precision: int = 2) -> Dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    arr = np.asarray(values, dtype=float)
    low = float(np.percentile(arr, 25))
    high = float(np.percentile(arr, 75))
    low = DifferentialPrivacy.add_noise(low, epsilon=max(epsilon, 0.1), sensitivity=1.0)
    high = DifferentialPrivacy.add_noise(high, epsilon=max(epsilon, 0.1), sensitivity=1.0)
    low, high = sorted((low, high))
    return {"low": round(low, precision), "high": round(high, precision)}


def build_market_snapshot(client_metrics: List[Dict[str, Any]], epsilon: float, min_cohort: int = MIN_PRIVACY_COHORT) -> Dict[str, Any]:
    if len(client_metrics) < min_cohort:
        return {
            "available": False,
            "client_count": len(client_metrics),
            "reason": f"Need at least {min_cohort} clients before market benchmarks are exposed.",
        }

    market_history_df = pd.DataFrame()
    for metric in client_metrics:
        history = pd.DataFrame(metric["history"])
        if history.empty:
            continue
        temp = history[["week", "demand"]].rename(columns={"demand": metric["client_id"]})
        market_history_df = temp if market_history_df.empty else market_history_df.merge(temp, on="week", how="outer")

    market_history: List[Dict[str, Any]] = []
    if not market_history_df.empty:
        market_history_df = market_history_df.sort_values("week").tail(24)
        market_history_df["market_avg_demand"] = market_history_df.drop(columns=["week"]).mean(axis=1, skipna=True)
        market_history = market_history_df[["week", "market_avg_demand"]].fillna(0).to_dict(orient="records")

    net_profit_values = [item["financials"]["net_profit"] for item in client_metrics]
    return {
        "available": True,
        "client_count": len(client_metrics),
        "generated_at": utc_now_iso(),
        "kpis": {
            "avg_forecast": _safe_market_value([item["forecast_next"] for item in client_metrics], epsilon, precision=1),
            "avg_order_qty": _safe_market_value([item["projected_order_qty"] for item in client_metrics], epsilon, precision=1),
            "avg_emission_factor": _safe_market_value([item["emission_factor"] for item in client_metrics], epsilon, precision=3),
            "avg_projected_emissions": _safe_market_value([item["projected_emissions"] for item in client_metrics], epsilon, precision=2),
            "avg_risk_level": _safe_market_value([item["risk_level"] for item in client_metrics], epsilon, precision=3),
            "avg_forecast_error": _safe_market_value([item["forecast_error"] for item in client_metrics], epsilon, precision=2),
            "avg_net_profit": _safe_market_value(net_profit_values, epsilon, precision=2),
            "profit_band": _safe_market_band(net_profit_values, epsilon, precision=2),
            "demand_growth_band": _safe_market_band([item["demand_growth_pct"] for item in client_metrics], epsilon, precision=2),
        },
        "history": market_history,
        "privacy": {
            "aggregation_only": True,
            "peer_rows_shared": False,
            "minimum_cohort": min_cohort,
            "dp_epsilon": round(float(epsilon), 3),
        },
    }


def build_client_bundles(
    data_manager,
    lstm_model,
    scale: float,
    inventory: int = DEFAULT_INVENTORY,
    epsilon: float = SCConfig.DP_EPSILON,
) -> Dict[str, Dict[str, Any]]:
    client_metrics = compute_client_metrics(
        data_manager=data_manager,
        lstm_model=lstm_model,
        scale=scale,
        inventory=inventory,
    )
    market_snapshot = build_market_snapshot(client_metrics, epsilon=epsilon)
    bundles: Dict[str, Dict[str, Any]] = {}

    for metric in client_metrics:
        market_kpis = market_snapshot.get("kpis", {})
        comparison = {
            "forecast_vs_market": round(float(metric["forecast_next"]) - float(market_kpis.get("avg_forecast", 0.0)), 2),
            "order_vs_market": round(float(metric["projected_order_qty"]) - float(market_kpis.get("avg_order_qty", 0.0)), 2),
            "emissions_vs_market": round(float(metric["projected_emissions"]) - float(market_kpis.get("avg_projected_emissions", 0.0)), 2),
            "profit_vs_market": round(float(metric["financials"]["net_profit"]) - float(market_kpis.get("avg_net_profit", 0.0)), 2),
            "risk_vs_market": round(float(metric["risk_level"]) - float(market_kpis.get("avg_risk_level", 0.0)), 4),
        }
        bundles[metric["client_id"]] = to_serializable(
            {
                "bundle_type": "client_market_sync",
                "generated_at": utc_now_iso(),
                "client": metric,
                "market": market_snapshot,
                "comparison": comparison,
            }
        )

    return bundles


def default_node_registry(dataset_dir: str = SCConfig.DATASET_DIR, num_clients: int = SCConfig.NUM_CLIENTS) -> List[Dict[str, Any]]:
    catalog = infer_client_catalog(dataset_dir=dataset_dir, num_clients=num_clients)
    nodes: List[Dict[str, Any]] = []
    for item in catalog:
        cid = int(item["client_id"])
        nodes.append(
            {
                "node_id": f"node-{item['client_id']}",
                "client_id": item["client_id"],
                "node_name": item["node_name"],
                "label": item["label"],
                "brand": item["brand"],
                "region": item["region"],
                "base_url": f"http://127.0.0.1:{8800 + cid}",
                "last_seen": None,
                "status": "offline",
                "latency_ms": None,
                "last_sync_at": None,
                "error": None,
            }
        )
    return nodes


def load_node_registry(dataset_dir: str = SCConfig.DATASET_DIR, num_clients: int = SCConfig.NUM_CLIENTS) -> List[Dict[str, Any]]:
    registry_path = node_registry_path()
    if not os.path.exists(registry_path):
        registry = default_node_registry(dataset_dir=dataset_dir, num_clients=num_clients)
        save_node_registry(registry)
        return registry

    with open(registry_path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)

    default_registry = {item["client_id"]: item for item in default_node_registry(dataset_dir=dataset_dir, num_clients=num_clients)}
    normalized: List[Dict[str, Any]] = []
    for item in registry:
        client_id = str(item.get("client_id"))
        base = default_registry.get(client_id, {}).copy()
        base.update(item)
        normalized.append(base)

    known_ids = {item["client_id"] for item in normalized}
    for client_id, item in default_registry.items():
        if client_id not in known_ids:
            normalized.append(item)
    save_node_registry(normalized)
    return normalized


def save_node_registry(registry: List[Dict[str, Any]]) -> None:
    os.makedirs(SCConfig.LOG_DIR, exist_ok=True)
    with open(node_registry_path(), "w", encoding="utf-8") as handle:
        json.dump(to_serializable(registry), handle, indent=2)
