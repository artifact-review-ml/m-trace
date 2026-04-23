"""Generate NeurIPS Figure 1: Trajectory Alignment Heatmap"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import pyarrow.parquet as pq
import re
import sys

# === Dynamic Project Root Resolution ===
def find_project_root():
    """Robustly finds project root by searching for mtrace_logs or t_trace."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "mtrace_logs").exists() or (parent / "t_trace").exists():
            return parent
    return current.parents[4]  # Fallback

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# Automatically find the most recent log file
LOG_DIR = PROJECT_ROOT / "mtrace_logs" / "development"
def get_latest_log(log_dir):
    files = list(log_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {log_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)

LOG_FILE = get_latest_log(LOG_DIR)
OUTPUT_PNG = PROJECT_ROOT / "t_trace" / "experiments" / "phase1" / "exp1" / "results" / "fig1_trajectory_alignment.png"

print(f"📂 Using log file: {LOG_FILE.name}")
print(f"🖼️ Saving figure to: {OUTPUT_PNG}")

# Load logs
df = pq.read_table(LOG_FILE).to_pandas()

# Extract block indices and intensities
blocks, intensities, timestamps = [], [], []
for _, row in df.iterrows():
    internal = row.get("internal_states", {})
    if isinstance(internal, dict):
        layer_name = internal.get("layer_name", "")
        match = re.search(r"layers\.(\d+)", layer_name)
        block_idx = int(match.group(1)) + 1 if match else internal.get("layer_index", 0)
        
        out_act = internal.get("output_activations", {})
        sparse_vals = out_act.get("sparse_values", []) if isinstance(out_act, dict) else []
        norm = np.linalg.norm(sparse_vals) if sparse_vals else 1.0
        
        blocks.append(block_idx)
        intensities.append(norm)
        timestamps.append(row["model_metadata"]["timestamp"])

# Aggregate intensity per block
unique_blocks = sorted(set(blocks))
block_intensity = {b: [] for b in unique_blocks}
for b, i in zip(blocks, intensities):
    block_intensity[b].append(i)
mean_intensity = [np.mean(block_intensity[b]) for b in unique_blocks]

# Create heatmap-style bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=[f"Block {b}" for b in unique_blocks],
    y=mean_intensity,
    marker=dict(color=mean_intensity, colorscale="Viridis", showscale=True),
    name="Computational Intensity"
))

# Add ground-truth step ranges
steps = [
    ("FeatureCheck_A", 1, 3),
    ("FeatureCheck_B", 4, 6),
    ("InteractionRule", 7, 9),
    ("DecisionThreshold", 10, 12)
]
for name, start, end in steps:
    fig.add_vrect(
        x0=f"Block {start}", x1=f"Block {end}",
        fillcolor="rgba(255, 100, 100, 0.15)", layer="below", line_width=0,
        annotation_text=name, annotation_position="top left"
    )

fig.update_layout(
    title="M-TRACE Temporal Fidelity: Trajectory vs. Ground-Truth Alignment (Φ_T = 1.0)",
    xaxis_title="Transformer Block Index",
    yaxis_title="Mean Activation Intensity (L2 Norm)",
    template="plotly_white",
    height=450,
    margin=dict(l=60, r=40, t=80, b=60)
)

# Save high-res PNG for paper
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
pio.write_image(fig, str(OUTPUT_PNG), scale=3, width=900, height=450)
print(f"✅ Figure saved to {OUTPUT_PNG}")