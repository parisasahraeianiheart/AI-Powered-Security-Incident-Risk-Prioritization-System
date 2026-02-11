"""
Streamlit dashboard for the UNSW-NB15 SOC triage project.

Shows:
- Model evaluation outputs produced by `model.py` (JSON + saved plots)
- SHAP explainability plots produced by `XGBOOSTexpalin.py`

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List

import streamlit as st


# -----------------------------
# Paths / Files
# -----------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "results_models_deep_dive.json"

EDA_PLOTS = [
    ROOT / "eda_class_distribution.png",
    ROOT / "eda_attack_categories.png",
    ROOT / "eda_hist_dur.png",
    ROOT / "eda_train_vs_test_dur.png",
    ROOT / "eda_hist_sbytes.png",
    ROOT / "eda_train_vs_test_sbytes.png",
    ROOT / "eda_hist_dbytes.png",
    ROOT / "eda_train_vs_test_dbytes.png",
    ROOT / "eda_hist_sttl.png",
    ROOT / "eda_train_vs_test_sttl.png",
    ROOT / "eda_hist_dttl.png",
    ROOT / "eda_train_vs_test_dttl.png",
]

SHAP_PLOTS = [
    ROOT / "shap_bar_xgb.png",
    ROOT / "shap_summary_xgb.png",
    ROOT / "shap_waterfall_example0.png",
]


# -----------------------------
# Helpers
# -----------------------------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing: `{path.name}` (run the generating script).")


def _run_script(script_name: str, timeout_s: int = 600) -> str:
    """
    Run a script in the project root and return combined output.
    """
    cmd = ["python3", str(ROOT / script_name)]
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return out.strip()


def _find_pdfs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.pdf"))


# -----------------------------
# Page Config + Style
# -----------------------------
st.set_page_config(page_title="SOC Triage Dashboard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .kpi-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.02);
      }
      .small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
      .section-title { margin-top: 0.25rem; margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ SOC Triage Dashboard (UNSW-NB15)")
st.caption("Interactive view of EDA, model performance, thresholding artifacts, and SHAP explainability outputs.")


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    st.markdown("**Regenerate artifacts (optional)**")
    if st.button("▶ Run model.py", use_container_width=True):
        with st.spinner("Running model.py (this may take time)..."):
            try:
                logs = _run_script("model.py", timeout_s=3600)
                st.session_state["model_logs"] = logs
                st.success("model.py finished. Refreshing dashboard files.")
            except Exception as e:
                st.session_state["model_logs"] = f"Failed to run model.py: {e}"
                st.error("model.py failed. See logs below.")

    with st.expander("Latest logs", expanded=False):
        st.code(st.session_state.get("model_logs", "No logs yet."), language="text")

    st.divider()
    st.markdown("**Download report**")

    pdfs = _find_pdfs(ROOT)
    if pdfs:
        chosen_pdf = st.selectbox(
            "Select a PDF to download",
            options=pdfs,
            format_func=lambda p: p.name,
        )
        try:
            st.download_button(
                label=f"⬇️ Download {chosen_pdf.name}",
                data=chosen_pdf.read_bytes(),
                file_name=chosen_pdf.name,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.warning("Could not read the selected PDF file.")
    else:
        st.info("No PDF found in project folder. Put your report PDF next to this app (same directory).")

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        ["Overview", "Model Results", "EDA", "SHAP"],
        index=0,
        label_visibility="collapsed",
    )


# -----------------------------
# Load Results (shared)
# -----------------------------
results = _read_json(RESULTS_JSON)


# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.subheader("Overview")

    if not results:
        st.warning(f"Missing or unreadable `{RESULTS_JSON.name}`. Run `model.py` first.")
    else:
        data_info = results.get("data", {})
        split_info = results.get("split_strategy", "")
        cost_model = results.get("cost_model", {})

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Train rows", f"{data_info.get('train_rows', '-')}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Test rows", f"{data_info.get('test_rows', '-')}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Train + rate", f"{data_info.get('train_pos_rate', float('nan')):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Test + rate", f"{data_info.get('test_pos_rate', float('nan')):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Methodology")
        st.write(split_info)
        st.markdown("#### Cost Model")
        st.write(f"C_FP = {cost_model.get('C_FP')}  |  C_FN = {cost_model.get('C_FN')}")

        st.markdown("#### Quick takeaway")
        st.write(
            "This dashboard reads artifacts generated by the training scripts (JSON + plots). "
            "If anything is missing, run `model.py` and refresh."
        )


elif page == "Model Results":
    st.subheader("Model Results")

    if not results:
        st.warning(f"Missing or unreadable `{RESULTS_JSON.name}`. Run `model.py` to generate it.")
    else:
        models = results.get("models") or {}
        model_keys = list(models.keys())

        if not model_keys:
            st.warning("No models found in results JSON.")
        else:
            picked = st.selectbox("Select model", model_keys, index=0)
            m = models[picked]

            thr = m.get("threshold_selected_on_train_oof", None)
            st.markdown(f"**Selected threshold (train OOF):** `{thr}`")

            test_metrics = m.get("test_metrics") or {}
            k1, k2, k3 = st.columns(3, gap="medium")
            k1.metric("PR-AUC", f"{test_metrics.get('pr_auc', float('nan')):.4f}")
            k2.metric("ROC-AUC", f"{test_metrics.get('roc_auc', float('nan')):.4f}")
            k3.metric("Brier", f"{test_metrics.get('brier', float('nan')):.4f}")

            st.markdown("#### Confusion matrix + rates (test @ selected threshold)")
            st.json(m.get("test_at_threshold") or {})

            artifacts = (m.get("artifacts") or {})
            pr_plot = ROOT / str(artifacts.get("test_pr_curve", ""))
            train_cost_plot = ROOT / str(artifacts.get("train_cost_curve", ""))
            test_cost_plot = ROOT / str(artifacts.get("test_cost_curve", ""))

            st.markdown("#### Plots")
            tabs = st.tabs(["PR Curve", "Train Cost Curve", "Test Cost Curve"])
            with tabs[0]:
                _show_image_if_exists(pr_plot, f"PR curve (test) — {picked}")
            with tabs[1]:
                _show_image_if_exists(train_cost_plot, f"Cost curve (train OOF) — {picked}")
            with tabs[2]:
                _show_image_if_exists(test_cost_plot, f"Cost curve (test) — {picked}")

            st.caption("Tip: thresholds far from 0.5 are expected because FN cost is much higher than FP cost.")


elif page == "EDA":
    st.subheader("EDA (Exploratory Data Analysis)")
    st.caption("Charts saved by `model.py` during EDA. Missing plots mean the script hasn’t been run (or files were moved).")

    # show in two columns for a nicer gallery layout
    cols = st.columns(2, gap="large")
    for i, p in enumerate(EDA_PLOTS):
        with cols[i % 2]:
            _show_image_if_exists(p, p.name)


elif page == "SHAP":
    st.subheader("SHAP Explainability")
    st.caption("Explainability plots produced by `XGBOOSTexpalin.py` (global importance, beeswarm, local waterfall).")

    # show in a neat order
    for p in SHAP_PLOTS:
        _show_image_if_exists(p, p.name)

    st.info(
        "If these are missing, run your SHAP script (e.g., `python3 XGBOOSTexpalin.py`) "
        "and refresh the dashboard."
    )