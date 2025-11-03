import plotly.graph_objects as go
from pathlib import Path
import json
from collections import defaultdict

params = [
    (0.001, 0.01),
    (0.005, 0.02),
    (0.01, 0.05),
    (0.02, 0.1),
]

results_dir = Path("results")

# Store data by family: (use_base_model, use_abs) -> list of (capabilities_score, asr_score, p)
data_by_family = defaultdict(list)

for p, q in params:
    # Find all folders in results/ whose name starts with p{p}_q{q}
    pattern_prefix = f"p{p}_q{q}_"
    
    for folder in results_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith(pattern_prefix):
            continue
        
        # Load config.json
        config_path = folder / "config.json"
        if not config_path.exists():
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Filter: ignore folders where safety_dataset != "adv_llama_short"
        if config.get("safety_dataset") != "adv_llama_short":
            continue
        
        # Determine family
        use_base_model = config.get("base_model") is not None
        use_abs = config.get("take_abs", False)
        family = (use_base_model, use_abs)
        
        # Load zero_shot_results.json
        zero_shot_path = folder / "zero_shot_results.json"
        if not zero_shot_path.exists():
            continue
        
        with open(zero_shot_path, 'r') as f:
            zero_shot = json.load(f)
        
        # Calculate capabilities score: average of acc_norms (or acc if not available) across 3 eval datasets
        eval_datasets = ["rte", "arc_challenge", "openbookqa"]
        acc_scores = []
        for dataset in eval_datasets:
            if dataset in zero_shot.get("results", {}):
                dataset_results = zero_shot["results"][dataset]
                # Prefer acc_norm, fallback to acc
                score = dataset_results.get("acc_norm") or dataset_results.get("acc")
                if score is not None:
                    acc_scores.append(score)
        
        if not acc_scores:
            continue
        
        capabilities_score = sum(acc_scores) / len(acc_scores)
        
        # Load ASR_overview.json
        asr_overview_path = folder / "ASR_overview.json"
        if not asr_overview_path.exists():
            continue
        
        with open(asr_overview_path, 'r') as f:
            asr_overview = json.load(f)
        
        # Calculate ASR score: maximum of scores for keys "0", "1", "2"
        asr_scores = []
        for key in ["0", "1", "2"]:
            if key in asr_overview:
                asr_scores.append(asr_overview[key])
        
        if not asr_scores:
            continue
        
        asr_score = max(asr_scores)
        
        # Store data point with p value
        data_by_family[family].append((capabilities_score, asr_score, p))

# Create scatter plot
fig = go.Figure()

# Marker shape based on abs: True = circle, False = star
def get_marker_shape(use_abs):
    return 'circle' if use_abs else 'star'

family_labels = {
    (True, True): "Base Model=True, Abs=True",
    (True, False): "Base Model=True, Abs=False",
    (False, True): "Base Model=False, Abs=True",
    (False, False): "Base Model=False, Abs=False"
}

# Create discrete color mapping for p values (ordered by p value size)
p_values_all = sorted(set([p for p, q in params]))
# Create mapping from p to q
p_to_q = {p: q for p, q in params}
# Assign distinct colors in order (lighter to darker for increasing p)
discrete_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # blue, green, orange, red
p_to_color = {p: discrete_colors[i] for i, p in enumerate(p_values_all)}

# Plot family traces
for family, data_points in sorted(data_by_family.items()):
    if not data_points:
        continue
    
    use_base_model, use_abs = family
    capabilities_scores, asr_scores, p_values = zip(*data_points)
    
    # Get marker shape based on abs
    marker_symbol = get_marker_shape(use_abs)
    
    # Map p values to discrete colors
    colors = [p_to_color[p] for p in p_values]
    
    # Marker configuration: solid for base_model=True, hollow for base_model=False
    if use_base_model:
        # Solid markers - make slightly larger to match visual size of hollow ones
        marker_config = dict(
            symbol=marker_symbol,
            color=colors,
            size=8,
            line=dict(
                color='black',
                width=0.5
            )
        )
    else:
        # Hollow markers: white fill with colored outline (p value) and thin black outline
        # Use colored outline to show p values, with black as a subtle border
        marker_config = dict(
            symbol=marker_symbol,
            color=['white'] * len(colors),
            size=7.5,
            line=dict(
                color=colors,
                width=2
            )
        )
    
    fig.add_trace(go.Scatter(
        x=list(capabilities_scores),
        y=list(asr_scores),
        mode='markers',
        name="",  # Empty name - we'll add separate legend entries
        marker=marker_config,
        showlegend=False  # Don't show individual family traces in legend
    ))

# Add invisible traces for Base Model legend entries
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name="Base Model=True",
    marker=dict(
        color='black',
        size=10,
        symbol='circle'
    ),
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name="Base Model=False",
    marker=dict(
        color='white',
        size=10,
        symbol='circle',
        line=dict(color='black', width=2)
    ),
    showlegend=True
))

# Add invisible traces for Abs legend entries
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name="Abs=True",
    marker=dict(
        color='black',
        size=10,
        symbol='circle'
    ),
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name="Abs=False",
    marker=dict(
        color='black',
        size=10,
        symbol='star'
    ),
    showlegend=True
))

# Add invisible traces for (p, q) value color legend
for p in p_values_all:
    q = p_to_q[p]
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name=f"p={p}, q={q}",
        marker=dict(
            color=p_to_color[p],
            size=10,
            symbol='circle'
        ),
        showlegend=True
    ))

fig.update_layout(
    title='Capabilities vs ASR Score by Configuration',
    xaxis_title='Capabilities Score (Average Accuracy)',
    yaxis_title='ASR Score (Maximum)',
    hovermode='closest',
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top'
    )
)

save_path = Path("plotting/base_abs_small.pdf")
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(str(save_path))
print(f"Plot saved to {save_path}")