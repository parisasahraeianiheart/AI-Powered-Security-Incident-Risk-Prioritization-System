"""
SOC Triage Dashboard (Cloud-safe)

Reads ONLY triage_output.json produced by soc_triage_agent.py.
Does NOT depend on training/testing CSVs (so it works on Streamlit Cloud).

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_TRIAGE_JSON = ROOT / "triage_output.json"


# -----------------------------
# Helpers
# -----------------------------
def load_json_bytes(data: bytes) -> Dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def load_json_path(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_alerts_df(triage: Dict[str, Any]) -> pd.DataFrame:
    alerts = triage.get("top_alerts", []) or triage.get("alerts", []) or []
    if not alerts:
        return pd.DataFrame()

    df = pd.DataFrame(alerts)

    # normalize columns if some are missing
    for col in ["idx", "p", "decision", "explanation", "suggested_actions"]:
        if col not in df.columns:
            df[col] = None

    # make sure types are nice
    if "p" in df.columns:
        df["p"] = pd.to_numeric(df["p"], errors="coerce")

    if "idx" in df.columns:
        df["idx"] = pd.to_numeric(df["idx"], errors="coerce").astype("Int64")

    return df.sort_values("p", ascending=False, na_position="last")


def metric_get(triage: Dict[str, Any], path: list[str], default=None):
    cur: Any = triage
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="SOC Triage Dashboard",
    layout="wide",
    page_icon="🛡️",
)

st.title("🛡️ SOC Triage Dashboard")
st.caption("Human-in-the-loop triage using threshold policy + optional AI brief (from triage_output.json).")

with st.sidebar:
    st.header("Data Source")

    uploaded = st.file_uploader("Upload triage_output.json", type=["json"])
    use_local = st.checkbox("Use local triage_output.json (repo file)", value=(uploaded is None))

    triage: Optional[Dict[str, Any]] = None
    if uploaded is not None:
        try:
            triage = load_json_bytes(uploaded.read())
            st.success("Using: uploaded JSON")
        except Exception as e:
            st.error(f"Could not read uploaded JSON: {e}")
    elif use_local:
        triage = load_json_path(DEFAULT_TRIAGE_JSON)
        if triage:
            st.success("Using: local triage_output.json")
        else:
            st.warning("No local triage_output.json found. Upload one above.")

    st.divider()
    st.header("Filters")

    decision_filter = st.multiselect("Decision", ["ESCALATE", "DEPRIORITIZE"], default=["ESCALATE"])
    p_min, p_max = st.slider("Probability range (p)", 0.0, 1.0, (0.0, 1.0), 0.01)

    search_text = st.text_input("Search (idx / explanation text)", value="")

    st.divider()
    st.header("Report")



if not triage:
    st.stop()

# -----------------------------
# Top summary KPIs
# -----------------------------
model_name = triage.get("model", "—")
mode = triage.get("mode", "—")
c_fp = metric_get(triage, ["cost_model", "C_FP"], "—")
c_fn = metric_get(triage, ["cost_model", "C_FN"], "—")

threshold = triage.get("threshold", triage.get("best_threshold", None))
cost_reduction = triage.get("cost_reduction_vs_baseline", triage.get("cost_reduction", None))

metrics_at_t = triage.get("metrics_at_threshold", {}) or {}
precision = metrics_at_t.get("precision", None)
recall = metrics_at_t.get("recall", None)
fpr = metrics_at_t.get("fpr", None)

k1, k2, k3, k4, k5 = st.columns([1.2, 1.0, 1.0, 1.0, 1.0])
k4.metric("Precision", f"{precision:.3f}" if isinstance(precision, (int, float)) else "—")
k5.metric("Recall", f"{recall:.3f}" if isinstance(recall, (int, float)) else "—")

k6, k7, k8 = st.columns(3)
k6.metric("FPR", f"{fpr:.3f}" if isinstance(fpr, (int, float)) else "—")

if isinstance(cost_reduction, (int, float)):
    st.caption(f"Estimated cost reduction vs baseline threshold: **{100*cost_reduction:.2f}%**")

st.divider()

# -----------------------------
# Alert Queue + Risk Distribution
# -----------------------------
alerts_df = to_alerts_df(triage)

left, right = st.columns([1.25, 0.95], gap="large")

with left:
    st.subheader("🚨 Alert Queue")

    if alerts_df.empty:
        st.warning("No alerts found in JSON. Expected key: `top_alerts` (list).")
    else:
        df = alerts_df.copy()

        # filter by decision, probability range, search text
        df = df[df["decision"].isin(decision_filter)]
        df = df[(df["p"] >= p_min) & (df["p"] <= p_max)]

        if search_text.strip():
            s = search_text.strip().lower()
            df = df[
                df["idx"].astype(str).str.contains(s, na=False)
                | df["explanation"].astype(str).str.lower().str.contains(s, na=False)
            ]

        st.caption("Ranked by risk probability (p). Click a row in the table using the selector below.")

        show_cols = ["idx", "p", "decision"]
        if "attack_cat" in df.columns:
            show_cols.append("attack_cat")

        st.dataframe(
            df[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=340,
        )

        st.subheader("📊 Risk Distribution")
        st.bar_chart(df["p"].dropna(), use_container_width=True)

with right:
    st.subheader("🧾 Alert Details")

    if alerts_df.empty:
        st.info("Upload a triage_output.json with `top_alerts` to see alert details.")
    else:
        # pick an alert
        idx_list = alerts_df["idx"].dropna().astype(int).tolist()
        chosen_idx = st.selectbox("Select alert idx", idx_list, index=0)

        row = alerts_df[alerts_df["idx"] == chosen_idx].iloc[0].to_dict()

        st.markdown(f"**Alert idx:** `{row.get('idx')}`")
        st.markdown(f"**Probability (p):** `{row.get('p'):.3f}`" if isinstance(row.get("p"), (int, float)) else "**Probability (p):** —")
        st.markdown(f"**Decision:** `{row.get('decision')}`")

        st.divider()
        st.markdown("### Explanation")
        st.write(row.get("explanation", "—"))

        st.markdown("### Suggested Actions")
        actions = row.get("suggested_actions") or []
        if isinstance(actions, list) and actions:
            for a in actions:
                st.write(f"- {a}")
        else:
            st.write("—")

        # optional: show the LLM brief if present
        llm_brief = triage.get("llm_brief", None)
        if llm_brief:
            st.divider()
            st.markdown("### AI Brief (Queue-level)")
            st.write(llm_brief)
