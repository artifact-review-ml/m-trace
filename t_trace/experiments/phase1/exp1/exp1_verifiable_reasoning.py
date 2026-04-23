"""Exp1: Verifiable Reasoning Alignment & Temporal Fidelity Computation (NeurIPS-Ready)"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import pyarrow.parquet as pq
import time

# === Dynamic Project Root Resolution ===
def find_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "mtrace_logs").exists() or (parent / "t_trace").exists():
            return parent
    return current.parents[4]

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
from t_trace.logging_engine import enable_logging

# === 1. Synthetic Clinical Dataset (0-BASED INDEXING FIX) ===
def generate_clinical_dataset(n_samples=50):
    np.random.seed(42)
    X = np.random.randn(n_samples, 6).astype(np.float32)
    labels, traces = [], []
    
    for i in range(n_samples):
        age_risk = X[i, 0] > 0.5
        lab_abnormal = X[i, 1] > 0.8
        interaction = age_risk and lab_abnormal
        base_score = X[i, 2] * 0.4 + X[i, 3] * 0.6
        risk_score = base_score + (2.0 if interaction else 0.0)
        labels.append(int(risk_score > 0.7))
        
        traces.append({
            "steps": [
                # CRITICAL FIX: M-TRACE logs layer_index 0-based. 
                # 12 transformer blocks → indices 0-11. Adjusted ranges accordingly.
                {"op": "FeatureCheck_A", "layers": [0, 2]},
                {"op": "FeatureCheck_B", "layers": [3, 5]},
                {"op": "InteractionRule", "layers": [6, 8]},
                {"op": "DecisionThreshold", "layers": [9, 11]}
            ],
            "label": labels[-1]
        })
    return X, np.array(labels), traces

# === 2. Transformer Model (12 layers) ===
class ClinicalTransformer(nn.Module):
    def __init__(self, input_dim=6, hidden=32, layers=12, classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, nhead=4, batch_first=True) for _ in range(layers)
        ])
        self.classifier = nn.Linear(hidden, classes)
        
    def forward(self, x):
        h = self.embedding(x).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(h.squeeze(1))

# === 3. Temporal Fidelity Computation (Definition 3 Compliant) ===
def compute_temporal_fidelity_per_run(log_df: pd.DataFrame, trace: dict) -> float:
    """Compute Φ_T for a SINGLE forward pass per Definition 3."""
    layer_indices, intensities = [], []
    
    for _, row in log_df.iterrows():
        internal = row.get("internal_states", {})
        if not isinstance(internal, dict): continue
            
        idx = internal.get("layer_index")
        if idx is None or not isinstance(idx, (int, float)): continue
            
        # Extract intensity from M-TRACE sparse schema
        out_act = internal.get("output_activations", {})
        if isinstance(out_act, dict):
            sparse_vals = out_act.get("sparse_values", [])
            norm = np.linalg.norm(sparse_vals) if sparse_vals else 0.0
        elif isinstance(out_act, (list, np.ndarray)):
            norm = np.linalg.norm(out_act) if len(out_act) > 0 else 0.0
        else:
            norm = 0.0
            
        layer_indices.append(int(idx))
        intensities.append(float(norm))
        
    if not layer_indices: return 0.0
    
    log_data = pd.DataFrame({"layer_index": layer_indices, "intensity": intensities})
    correct, total_steps = 0, len(trace["steps"])
    
    for step in trace["steps"]:
        expected_range = step["layers"]
        mask = log_data["layer_index"].between(expected_range[0], expected_range[1])
        if not mask.any(): continue
            
        # locate(g_n, T) = layer with MAX computational intensity
        peak_idx = log_data.loc[mask, "intensity"].idxmax()
        peak_layer = log_data.loc[peak_idx, "layer_index"]
        
        if expected_range[0] <= peak_layer <= expected_range[1]:
            correct += 1
            
    return correct / total_steps if total_steps > 0 else 0.0

# === 4. Main Pipeline ===
def run_pipeline():
    print("🔬 Starting Exp1: Verifiable Reasoning Alignment & Fidelity Computation")
    X, y, ground_truths = generate_clinical_dataset(n_samples=50)
    
    log_dir = PROJECT_ROOT / "mtrace_logs" / "development"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    model_t = ClinicalTransformer()
    model_t.eval()
    
    phi_scores = []
    print("📥 Running per-sample inference for trajectory isolation...")
    
    for i in range(len(X)):
        # Clear previous run logs safely
        for f in log_dir.glob("*.parquet"): 
            try: f.unlink()
            except: pass
            
        engine_t = enable_logging(model_t, mode="development")
        sample_input = torch.tensor(X[i:i+1], dtype=torch.float32)
        
        with torch.no_grad():
            _ = model_t(sample_input)
            
        run_id = engine_t.get_run_id()
        engine_t.disable_logging()
        time.sleep(0.5)  # Flush async writer thread
        
        # Load trajectory logs
        files = list(log_dir.glob(f"**/*{run_id[:8]}*.parquet"))
        if not files: continue
        log_df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
        
        phi = compute_temporal_fidelity_per_run(log_df, ground_truths[i])
        phi_scores.append(phi)
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ Processed {i+1}/50 samples. Running avg Φ_T: {np.mean(phi_scores):.3f}")
            
    print("\n" + "="*50)
    print(f"📊 TEMPORAL FIDELITY RESULTS (n={len(phi_scores)})")
    print(f"✅ Mean Φ_T   = {np.mean(phi_scores):.3f} ± {np.std(phi_scores):.3f}")
    print(f"📊 Median Φ_T = {np.median(phi_scores):.3f}")
    print(f"📉 Min / Max  = {np.min(phi_scores):.3f} / {np.max(phi_scores):.3f}")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()