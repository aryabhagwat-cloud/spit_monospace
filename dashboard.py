import streamlit as st
import cv2
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Perception Mission Control", layout="wide", page_icon="🛰️")

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { 
        font-size: 36px; 
        color: #00FFCC !important; 
        font-family: 'Courier New', Courier, monospace; 
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] { color: #FF4B4B !important; }
    .stAlert { border-radius: 8px; border: 1px solid #00FFCC; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Perception Research & Analytics Dashboard")
st.markdown("Real-time terrain safety and obstacle detection system.")
st.divider()

# --- CORE ACCURACY LOGIC (The Grain Filter) ---
def calculate_mission_metrics(mask):
    """
    Calculates robotics-grade metrics by filtering out sensor noise.
    Forces strict 0 (Safe) and 255 (Lethal) thresholding.
    """
    # Ensure binary mask (catch any remaining grayscale noise)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Group connected pixels into distinct objects
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    
    # NOISE FILTER: Only keep clusters larger than 100 pixels (ignores dust/artifacts)
    real_clusters = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > 100]
    
    cluster_count = len(real_clusters)
    total_pixels = binary_mask.size
    obstacle_pixels = np.sum(binary_mask == 255) 
    
    # Calculate Safety as the percentage of sand free from obstacles
    safety_score = max(0.0, 100.0 - (obstacle_pixels / total_pixels * 100.0))
    
    # Planner Effort: Measures navigation complexity based on cluster count and size
    planner_effort = (cluster_count * 1.5) + (obstacle_pixels / 10000)
    
    return round(safety_score, 1), cluster_count, round(planner_effort, 2)

# --- SIDEBAR: MISSION CONTROL SETTINGS ---
st.sidebar.header("🕹️ Mission Parameters")

# Look strictly in the refined folder for high accuracy
data_dir = "refined_masks" 

if not os.path.exists(data_dir):
    st.sidebar.error("⚠️ 'refined_masks' folder missing! Please run your refiner script first.")
    available_images = []
else:
    # Get all PNG files
    available_images = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])

selected_file = st.sidebar.selectbox("Select Terrain Scan", available_images if available_images else ["No images found"])
dust_slider = st.sidebar.slider("Dust Interference (Stability Test)", 0, 100, 0, help="Simulate lens dust or sensor noise.")

# --- MAIN DASHBOARD LAYOUT ---
if selected_file != "No images found" and available_images:
    file_path = os.path.join(data_dir, selected_file)
    original_mask = cv2.imread(file_path, 0)

    if original_mask is not None:
        # Apply Stability Testing (Environmental Noise)
        working_mask = original_mask.copy()
        if dust_slider > 0:
            # Add random noise to simulate harsh desert conditions
            noise = np.random.randint(0, dust_slider, working_mask.shape, dtype='uint8')
            working_mask = cv2.add(working_mask, noise)

        # Process metrics using high-accuracy logic
        safety, clusters, effort = calculate_mission_metrics(working_mask)
        stability = round(1.0 - (dust_slider / 200), 2) 

        # Setup columns for UI
        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.subheader("🖼️ Binary Perception Feed")
            # Show the raw mask
            st.image(working_mask, caption=f"Processing Scan: {selected_file}", use_container_width=True)
            
            st.divider()
            
            # Heatmap for the robotics navigation team
            st.subheader("🗺️ Navigation Cost Map")
            heatmap = cv2.applyColorMap(working_mask, cv2.COLORMAP_JET)
            st.image(heatmap, caption="Red = Lethal Obstacle | Blue = Safe Traversable Sand", use_container_width=True)

        with col2:
            st.subheader("📊 Robotics Analytics")
            
            # Metrics Row 1
            m1, m2 = st.columns(2)
            m1.metric("Safety Score", f"{safety}%", help="Percentage of traversable terrain.")
            m2.metric("Stability Index", f"{stability}", help="System robustness against environmental noise.")
            
            st.divider()

            # Metrics Row 2
            m3, m4 = st.columns(2)
            m3.metric("Filtered Clusters", clusters, delta="Noise Removed", delta_color="normal", help="Distinct physical obstacles detected.")
            m4.metric("Planner Effort", effort, help="Computed difficulty for navigation algorithms.")
            
            st.divider()

            # Mission Status Report 
            st.write("### Mission Assessment")
            if safety >= 80:
                st.success("🟢 **OPTIMAL**: Terrain is highly traversable. Proceed at standard speed.")
            elif safety >= 40:
                st.warning("🟡 **CAUTION**: Moderate obstacle density. Pathing adjustments required.")
            else:
                st.error("🔴 **LETHAL**: High obstacle density. Stop and reroute immediately.")

            st.write("")
            st.caption(f"File Source: `{file_path}`")

    else:
        st.error(f"Failed to load image at {file_path}. Please check the file.")
else:
    st.info("Upload or generate refined masks to see the dashboard in action.")