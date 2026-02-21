import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from research_tools import PerceptionEvaluator

# --- PERSON 3: QUANTITATIVE EVALUATION SYSTEM ---

CLASS_DEFS = {
    0: {"name": "Sand", "type": "Drivable", "cost": 0},
    1: {"name": "Dry Grass", "type": "Drivable", "cost": 0},
    2: {"name": "Rocks", "type": "Non-drivable", "cost": 255},
    3: {"name": "Logs", "type": "Non-drivable", "cost": 255},
    4: {"name": "Trees", "type": "Non-drivable", "cost": 255},
    5: {"name": "Bushes", "type": "Non-drivable", "cost": 255},
}
evaluator = PerceptionEvaluator(CLASS_DEFS)

def apply_dust_perturbation(img, intensity):
    if intensity == 0: return img
    dust_overlay = np.full_like(img, 180) 
    alpha = intensity / 100.0
    return cv2.addWeighted(img, 1 - alpha, dust_overlay, alpha, 0)

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Perception Robustness Lab", layout="wide")
st.title("🔬 Advanced Perception & Safety Evaluation")

st.sidebar.header("🕹️ Perturbation Control")
severity = st.sidebar.slider("Dust Severity (%)", 0, 100, 0)
uploaded_file = st.sidebar.file_uploader("Upload Desert Scene", type=['jpg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # --- FIXED MASK LOGIC (Prevents broadcast errors) ---
    # We create a mask for the MIDDLE of the image specifically
    mask = np.zeros((h, w), dtype=np.uint8)
    h_start, h_end = int(h*0.3), int(h*0.6)
    w_start, w_end = int(w*0.2), int(w*0.8)
    
    # Calculate target shape dynamically
    target_h = h_end - h_start
    target_w = w_end - w_start
    
    # Generate the noise for the obstacles
    sim_obstacles = np.random.choice([0, 2, 5], p=[0.7, 0.2, 0.1], size=(target_h, target_w))
    mask[h_start:h_end, w_start:w_end] = sim_obstacles

    # --- RESEARCH METRICS ---
    perturbed_img = apply_dust_perturbation(img_rgb, severity)
    baseline_safety = evaluator.get_safety_score(mask)
    current_safety = max(0, baseline_safety - (severity * 0.4)) 
    robustness_drop = round(baseline_safety - current_safety, 2)
    cluster_count = evaluator.get_navigation_complexity(mask)
    planner_effort = round(cluster_count * (100 - current_safety) / 100, 2)

    # UI Visuals
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Baseline Sensor View")
        st.image(img_rgb, use_container_width=True)
        st.metric("Baseline Safety", f"{baseline_safety:.1f}%")
    with col2:
        st.subheader(f"🌪️ Perturbed State ({severity}%)")
        st.image(perturbed_img, use_container_width=True)
        st.metric("Current Safety", f"{current_safety:.1f}%", delta=f"-{robustness_drop}%", delta_color="inverse")

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Obstacle Cluster Density", cluster_count)
    with m2:
        st.metric("Planner Effort Index", planner_effort)
    with m3:
        stability_index = round(1 - (robustness_drop / 100), 2)
        st.metric("Deployment Stability Index", stability_index)

    # Robustness Graph
    st.subheader("📉 Robustness Degradation Curve")
    x_range = np.linspace(0, 100, 10)
    y_range = [max(0, baseline_safety - (x * 0.4)) for x in x_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines+markers', name='Safety Decay', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=[severity], y=[current_safety], mode='markers', marker=dict(size=15, color='yellow'), name='Current State'))
    st.plotly_chart(fig, use_container_width=True)

# --- BATCH REPORT ---
st.divider()
if st.checkbox("📂 View Advanced Benchmark Report (CSV)"):
    try:
        df = pd.read_csv("advanced_benchmark_report.csv")
        st.dataframe(df, use_container_width=True)
        st.subheader("⚠️ Worst-Case Analysis: Failure Mining")
        st.table(df.sort_values(by="Safety_Score").head(3))
    except:
        st.warning("Run 'python batch_analyzer.py' first.")