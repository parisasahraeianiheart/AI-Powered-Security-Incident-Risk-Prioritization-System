"""
SOC Triage Agent: threshold selection, explanation, and action suggestions.

Usage:
  python soc_triage_agent.py --model xgboost --mode min_cost --top_alerts 10
  python soc_triage_agent.py --model xgboost --mode policy_precision --min_precision 0.70 --top_alerts 10
  python soc_triage_agent.py --model xgboost --mode min_cost --top_alerts 10 --use_llm
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from preprocessing import get_feature_types
from utils import (
    TRAIN_CSV,
    TEST_CSV,
    C_FP,
    C_FN,
    split_xy,
    expected_cost,
    find_best_threshold,
    eval_at_threshold,
)
from model import get_pipeline, best_recall_under_precision

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

_SYSTEM_PROMPT = (
    "You are a SOC triage copilot. Summarize alert queues and propose safe, practical next steps. "
    "Be concise, avoid speculation, and clearly separate observed facts from recommendations."
)


# -----------------------------
# Helpers: safe JSON
# -----------------------------
def _to_jsonable(x: Any) -> Any:
    """Convert numpy/pandas objects into JSON-serializable python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Series,)):
        return x.to_dict()
    if isinstance(x, (pd.DataFrame,)):
        return x.to_dict(orient="records")
    if isinstance(x, (Path,)):
        return str(x)
    return x


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_to_jsonable))


# -----------------------------
# LLM
# -----------------------------
def llm_soc_brief(
    context: Dict[str, Any],
    api_key: str,
    model: str,
    timeout_s: int = 30,
) -> str:
    """
    Simple LLM call for SOC summarization (stdlib-only).

    Enable by setting OPENAI_API_KEY and running with --use_llm.
    """
    url = "https://api.openai.com/v1/chat/completions"

    user_prompt = (
        "Given the JSON context below, produce:\n"
        "1) Executive summary (3-6 sentences)\n"
        "2) Prioritized recommended analyst actions (5-10 bullets)\n"
        "3) Escalation criteria (bullets)\n"
        "4) Key questions / data to pull next (bullets)\n\n"
        "Keep it SOC-friendly and concrete. Avoid vendor-specific claims.\n\n"
        f"CONTEXT_JSON:\n{json.dumps(context, indent=2, default=_to_jsonable)}"
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"].strip()


# -----------------------------
# Threshold + cost helpers
# -----------------------------
def cost_reduction_vs_baseline(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_opt: float,
    t_base: float = 0.5,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> Tuple[float, float, float]:
    base_cost = expected_cost(y_true, y_prob, t_base, c_fp, c_fn)
    opt_cost = expected_cost(y_true, y_prob, t_opt, c_fp, c_fn)
    reduction = (base_cost - opt_cost) / (base_cost + 1e-12)
    return float(base_cost), float(opt_cost), float(reduction)


# -----------------------------
# Explanation (heuristic, SOC-friendly)
# -----------------------------
def explain_alert_heuristic(row: pd.Series, prob: float) -> Tuple[str, List[str]]:
    key_cols = [
        "sttl",
        "ct_state_ttl",
        "sbytes",
        "ct_dst_sport_ltm",
        "ct_srv_src",
        "dload",
        "ct_dst_src_ltm",
        "smean",
        "ct_srv_dst",
    ]
    present = [(c, row[c]) for c in key_cols if c in row.index]
    present = present[:6]

    msg = [f"Risk score p={prob:.3f} driven by network behavior/volume/TTL signals."]
    if present:
        msg.append(
            "Top observed signals (raw feature values): "
            + ", ".join([f"{k}={_to_jsonable(v)}" for k, v in present])
        )

    actions: List[str] = []
    if "sttl" in row.index:
        actions.append("Validate TTL consistency (possible spoofing / scanning artifacts).")
    if "ct_srv_src" in row.index or "ct_dst_sport_ltm" in row.index:
        actions.append("Check repeated connections to same service/port (scan/bruteforce patterns).")
    if "sbytes" in row.index or "dload" in row.index:
        actions.append("Inspect unusual traffic volume/throughput (possible exfil/C2).")
    if not actions:
        actions.append("Review source/destination context and validate service/port behavior.")

    return " ".join(msg), actions


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SOC Triage Agent (threshold + explanation + action suggestions)")
    parser.add_argument("--model", default="xgboost", choices=["logreg", "svm_linear", "xgboost", "lightgbm"])
    parser.add_argument("--mode", default="min_cost", choices=["min_cost", "policy_precision"])
    parser.add_argument("--min_precision", type=float, default=0.70)
    parser.add_argument("--top_alerts", type=int, default=10)
    parser.add_argument("--baseline_threshold", type=float, default=0.5)

    parser.add_argument("--use_llm", action="store_true", help="Use LLM to summarize and suggest final next steps.")
    parser.add_argument("--llm_model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--llm_timeout_s", type=int, default=30)
    parser.add_argument("--llm_max_alerts", type=int, default=10)

    # NEW: always produce a JSON file for dashboards
    parser.add_argument("--save_json", action="store_true", help="Write triage_output.json (recommended).")
    parser.add_argument("--json_path", default=str(ROOT / "triage_output.json"))

    args = parser.parse_args()

    # --- Make train/test paths robust (TRAIN_CSV may be str or Path)
    train_csv = Path(TRAIN_CSV) if not isinstance(TRAIN_CSV, Path) else TRAIN_CSV
    test_csv = Path(TEST_CSV) if not isinstance(TEST_CSV, Path) else TEST_CSV

    if not train_csv.is_absolute():
        train_csv = ROOT / train_csv
    if not test_csv.is_absolute():
        test_csv = ROOT / test_csv

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Expected {train_csv.name} and {test_csv.name} in the project root.")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    num_cols, cat_cols = get_feature_types(X_train)

    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = float(neg / max(pos, 1))

    print("\n=== SOC TRIAGE AGENT ===")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Cost model: C_FP={C_FP}, C_FN={C_FN}")
    if args.mode == "policy_precision":
        print(f"Policy constraint: precision >= {args.min_precision:.2f}")

    pipe = get_pipeline(args.model, num_cols, cat_cols, spw)

    print("\nTraining on official training split...")
    pipe.fit(X_train, y_train)

    print("Scoring holdout test split...")
    prob_test = pipe.predict_proba(X_test)[:, 1]

    best_t: float
    stats: Dict[str, Any]

    if args.mode == "min_cost":
        best_t, _, _ = find_best_threshold(y_test, prob_test, C_FP, C_FN)
        stats = eval_at_threshold(y_test, prob_test, best_t)

        base_cost, opt_cost, reduction = cost_reduction_vs_baseline(
            y_test, prob_test, best_t, t_base=args.baseline_threshold, c_fp=C_FP, c_fn=C_FN
        )

        print("\n--- Agent Decision (Min Cost) ---")
        print(f"Chosen threshold t* = {best_t:.3f}")
        print(f"Baseline cost @ t={args.baseline_threshold:.2f}: {base_cost:.0f}")
        print(f"Optimized cost @ t*={best_t:.3f}: {opt_cost:.0f}")
        print(f"Cost reduction vs baseline: {100*reduction:.2f}%")
        print(f"Precision={stats['precision']:.3f} | Recall={stats['recall']:.3f} | FPR={stats['fpr']:.3f}")
        print(f"Confusion: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']}")

    else:
        policy = best_recall_under_precision(y_test, prob_test, min_precision=args.min_precision)
        print("\n--- Agent Decision (Policy: max recall s.t. precision constraint) ---")
        if policy is None:
            print(f"No threshold achieves precision >= {args.min_precision:.2f}.")
            print("Suggestion: lower constraint, calibrate probabilities, or apply analyst-capacity constraints.")
            return
        _r, _p, best_t = policy
        stats = eval_at_threshold(y_test, prob_test, best_t)
        print(f"Chosen threshold t = {best_t:.3f}")
        print(f"Precision={stats['precision']:.3f} | Recall={stats['recall']:.3f} | FPR={stats['fpr']:.3f}")
        print(f"Confusion: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']}")

    # --- Top alerts
    idx_sorted = np.argsort(-prob_test)
    topn = min(args.top_alerts, len(idx_sorted))
    top_idx = idx_sorted[:topn]

    print("\n--- Top Alerts (ranked by risk) ---")
    for rank, i in enumerate(top_idx, start=1):
        decision = "ESCALATE" if prob_test[i] >= best_t else "DEPRIORITIZE"
        print(f"{rank:02d}. idx={int(i)}  p={prob_test[i]:.3f}  decision={decision}")

    print("\n--- Example Explanations + Action Suggestions ---")

    triage_alerts: List[Dict[str, Any]] = []
    for i in top_idx:
        row = X_test.iloc[int(i)]
        p = float(prob_test[i])
        decision = "ESCALATE" if p >= float(best_t) else "DEPRIORITIZE"
        explanation, actions = explain_alert_heuristic(row, p)

        triage_alerts.append(
            {
                "idx": int(i),
                "p": p,
                "decision": decision,
                "explanation": explanation,
                "suggested_actions": actions,
            }
        )

    # Print only first 5 like before
    for a in triage_alerts[: min(5, len(triage_alerts))]:
        print(f"\nAlert idx={a['idx']} | p={a['p']:.3f} | decision={a['decision']}")
        print("Explanation:", a["explanation"])
        print("Suggested actions:")
        for act in a["suggested_actions"]:
            print(" -", act)

    # --- Optional LLM brief
    llm_brief: str | None = None
    llm_error: str | None = None

    if args.use_llm:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("\n--- Agentic AI Brief (skipped) ---")
            print("Set env var OPENAI_API_KEY to enable LLM summarization.")
            llm_error = "OPENAI_API_KEY not set"
        else:
            llm_top_n = min(int(args.llm_max_alerts), len(triage_alerts))
            llm_context: Dict[str, Any] = {
                "model": args.model,
                "mode": args.mode,
                "cost_model": {"C_FP": float(C_FP), "C_FN": float(C_FN)},
                "threshold": float(best_t),
                "metrics_at_threshold": stats,
                "top_alerts": triage_alerts[:llm_top_n],
                "notes": [
                    "LLM output is advisory; validate against logs and environment context.",
                    "Decisions above are computed from deterministic metrics; LLM is used only for summarization/suggestions.",
                ],
            }

            print("\n--- Agentic AI Brief (LLM) ---")
            try:
                llm_brief = llm_soc_brief(
                    llm_context,
                    api_key=api_key,
                    model=str(args.llm_model),
                    timeout_s=int(args.llm_timeout_s),
                )
                print(llm_brief)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
                llm_error = str(e)
                print(f"LLM call failed: {e}")
                print("Falling back to heuristic explanations above.")

    # --- NEW: write triage_output.json for Streamlit/dashboard use
    if args.save_json or True:
        out_path = Path(args.json_path)
        if not out_path.is_absolute():
            out_path = ROOT / out_path

        output: Dict[str, Any] = {
            "run": {
                "model": args.model,
                "mode": args.mode,
                "cost_model": {"C_FP": float(C_FP), "C_FN": float(C_FN)},
                "baseline_threshold": float(args.baseline_threshold),
                "threshold_selected": float(best_t),
            },
            "metrics_at_threshold": stats,
            "top_alerts": triage_alerts,
            "llm": {
                "enabled": bool(args.use_llm),
                "model": str(args.llm_model),
                "brief": llm_brief,
                "error": llm_error,
            },
        }

        write_json(out_path, output)
        print(f"\n[OK] Wrote dashboard output JSON: {out_path}")

    print("\n--- Agent Guardrails / Limitations ---")
    print("1) Agent decisions are computed from deterministic metrics (no free-form hallucinated actions).")
    print("2) If data distribution shifts, thresholds should be re-optimized and probabilities recalibrated.")
    print("3) This agent uses heuristic explanations; you can optionally replace/augment with SHAP local explanations.")
    if args.use_llm:
        print("4) LLM brief (if enabled) is advisory; treat it as a drafting aid, not ground truth.")


if __name__ == "__main__":
    main()
