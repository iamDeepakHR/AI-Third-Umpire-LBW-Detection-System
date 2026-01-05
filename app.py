import os
from pathlib import Path
from typing import List, Tuple, Optional
import time

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from lbw_ai.config import AppConfig
from lbw_ai.detector import BallDetector
from lbw_ai.tracker import SimpleBallTracker
from lbw_ai.bounce import detect_bounce_point
from lbw_ai.trajectory import fit_polynomial_trajectory, fit_trajectory
from lbw_ai.impact import estimate_stumps_region, predict_stump_intersection
from lbw_ai.classifier import LBWClassifier, LBWFeatures
from lbw_ai.visualize import draw_overlay, render_overlay_video, create_complete_trajectory_image
from lbw_ai.visualize_3d import create_3d_trajectory_plot, create_trajectory_comparison_plot
from lbw_ai.explainer import generate_explanation, ExplanationInputs
from lbw_ai.analytics import AnalyticsDashboard, create_review_record
from lbw_ai.report_generator import generate_pdf_report

# Streamlit page config
st.set_page_config(
    page_title="AI Third Umpire - LBW",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌙 --- Custom CSS: Better text contrast + visibility ---
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #1c1c28 0%, #2e2e45 100%);
        color: #e6e6e6;
    }

    /* Global text enhancements */
    p, div, label, span {
        color: #e0e0e0 !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #00ffaa !important;
        text-shadow: 0 0 15px rgba(0, 255, 170, 0.6);
    }

    /* Title */
    h1 {
        font-weight: 800;
        text-align: center;
        font-size: 2.5rem;
    }

    /* Info and helper text */
    .stAlert p, .stCaption, .stInfo {
        color: #cfcfcf !important;
    }

    /* Input fields - Text Input */
    .stTextInput > div > div > input {
        background-color: #2a2a3e !important;
        color: #ffffff !important;
        border: 1px solid #00ffaa !important;
        border-radius: 5px !important;
    }
    
    .stTextInput label {
        color: #b0ffdd !important;
        font-weight: 500 !important;
    }

    /* File Uploader */
    .stFileUploader {
        background-color: #2a2a3e !important;
        border: 2px dashed #00ffaa !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    .stFileUploader > div > div {
        color: #e0e0e0 !important;
    }
    
    .stFileUploader > div > div > div {
        color: #ffffff !important;
    }

    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: #2a2a3e !important;
        color: #ffffff !important;
        border: 1px solid #00ffaa !important;
        border-radius: 5px !important;
    }
    
    .stSelectbox label {
        color: #b0ffdd !important;
        font-weight: 500 !important;
    }

    /* Sliders and input fields */
    .stSlider label, .stNumberInput label {
        color: #b0ffdd !important;
        font-weight: 500 !important;
    }
    
    .stSlider > div > div {
        color: #ffffff !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #2a2a3e !important;
        color: #ffffff !important;
        border: 1px solid #00ffaa !important;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #181824 0%, #1f1f32 100%);
        color: #d9d9d9;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #00e0b5 !important;
        text-shadow: 0 0 10px rgba(0, 224, 181, 0.6);
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #b0ffdd !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #00ffaa !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #2a2a3e !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2a2a3e !important;
        color: #ffffff !important;
    }
    
    .streamlit-expanderContent {
        background-color: #252538 !important;
        color: #e0e0e0 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1e3a1e !important;
        color: #90ee90 !important;
    }
    
    .stError {
        background-color: #3a1e1e !important;
        color: #ff6b6b !important;
    }
    
    .stInfo {
        background-color: #1e2a3a !important;
        color: #87ceeb !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0099ff 0%, #00ff99 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
        transition: all 0.3s;
    }

    .stButton>button:hover {
        box-shadow: 0 6px 25px rgba(0, 255, 180, 0.6);
        transform: scale(1.04);
    }

    /* Decision boxes */
    .decision-out {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        color: #fff;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 0 25px rgba(255, 65, 108, 0.6);
        animation: pulse 2s infinite;
    }

    .decision-not-out {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        color: #fff;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 0 25px rgba(0, 176, 155, 0.6);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    /* Spinner animation */
    .loading-spinner {
        border: 4px solid #3a3a3a;
        border-top: 4px solid #00ffaa;
        border-radius: 50%;
        width: 55px;
        height: 55px;
        animation: spin 1.2s linear infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        color: #e5e5e5;
    }

    /* Tabs styling */
    div[data-baseweb="tab-list"] button {
        color: #00ffcc !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = AnalyticsDashboard()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'video_history' not in st.session_state:
    st.session_state.video_history = []
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None


def load_video_frames(video_path: str, max_frames: int | None = None) -> List[np.ndarray]:
    """Reads all frames from the uploaded video."""
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    return frames


def show_loading_animation():
    """Display loading animation"""
    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #00ff88;'>Analyzing... Decision Pending...</h3>", unsafe_allow_html=True)
    return placeholder


def show_decision_reveal(decision: str, confidence: float):
    """Animated decision reveal"""
    decision_class = "decision-out" if decision == "OUT" else "decision-not-out"
    st.markdown(f'<div class="{decision_class}">DECISION: {decision}<br><small>Confidence: {confidence:.1%}</small></div>', unsafe_allow_html=True)


def main():
    # Main title
    st.markdown("<h1>🏏 AI Third Umpire - LBW Decision System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #aaa;'>Advanced AI-powered LBW analysis with 3D visualization and comprehensive analytics</p>", unsafe_allow_html=True)
    
    cfg = AppConfig()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")
        
        weights_path = st.text_input("YOLO Weights Path", value=cfg.yolo_weights_path)
        conf_thr = st.slider("Confidence Threshold", 0.05, 0.9, cfg.confidence_threshold, 0.05)
        iou_thr = st.slider("IoU Threshold", 0.1, 0.9, cfg.iou_threshold, 0.05)
        st.markdown("""
        <style>
        /* Style the number input box */
        div[data-baseweb="input"] > div {
            background-color: #1E1E1E;  /* Dark background */
            color: white;               /* Text color */
            border-radius: 8px;         /* Rounded corners */
            border: 1px solid #555;     /* Border color */
        }

        /* Style the number itself */
        div[data-baseweb="input"] input {
            color: white !important;
        }

        /* Style the label text */
        label[data-testid="stNumberInputLabel"] {
            color: #00FFFF !important; /* Cyan label text */
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
        max_frames = st.number_input("Max Frames (0=all)", 0, 2000, value=cfg.max_frames or 0)
        
        st.markdown("""
            <style>
            div[data-baseweb="select"] > div {
                background-color: #222; /* Background color */
                color: white;            /* Text color */
                border-radius: 10px;     /* Rounded corners */
                border: 1px solid #555;  /* Border color */
            }

            div[data-baseweb="select"] svg {
                color: white; /* Dropdown arrow color */
            }

            div[data-baseweb="select"] span {
                color: white !important; /* Selected text color */
            }
            </style>
        """, unsafe_allow_html=True)
        st.header("🤖 AI Settings")
        gemini_api_key = "AIzaSyBJZzHfUbvPJvy8janmlTXhnMybmCeXDg4"
        commentary_tone = st.selectbox("Commentary Tone", ["Analyst", "Commentator"], index=0)
        
        st.markdown("---")
        st.header("📊 Analytics")
        stats = st.session_state.analytics.get_statistics()
        st.metric("Total Reviews", stats["total_reviews"])
        st.metric("OUT Decisions", f"{stats['out_count']} ({stats['out_percentage']:.1f}%)")
        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
        
        if st.button("Clear History"):
            st.session_state.analytics.clear_history()
            st.success("History cleared!")
            st.rerun()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎥 Video Analysis", "📊 Analytics Dashboard", "📚 History"])

    with tab1:
        uploaded = st.file_uploader("Upload a cricket video", type=["mp4", "mov", "avi", "mkv"])
        samples_dir = Path("samples")
        demo_videos = []
        if samples_dir.exists():
            demo_videos = [f for f in os.listdir(samples_dir) if f.endswith('.mp4')]
        demo_choice = st.selectbox("Or pick a demo video from samples folder", options=["(none)"] + demo_videos)

        run_btn = st.button("🚀 Run LBW Analysis", width='stretch')

        # Check if we should use existing results
        use_existing = st.session_state.analysis_results is not None and not run_btn
        
        if not run_btn and not use_existing:
            st.info("👆 Upload/select a video and click 'Run LBW Analysis' to begin.")
            return

        # Handle video selection
        video_path: str | None = None
        
        # Use existing results if available and button not clicked
        if use_existing and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            video_path = results['video_path']
            frames = results['frames']
            track_points = results['track_points']
            future_points = results['future_points']
            decision = results['decision']
            prob = results['confidence']
            bounce_index = results['bounce_index']
            stumps = results['stumps']
            would_hit_stumps = results['would_hit_stumps']
            distance_to_stumps_px = results['distance_to_stumps_px']
            pitched_zone = results['pitched_zone']
            impact_in_line = results['impact_in_line']
            h = results['frame_height']
            w = results['frame_width']
            
            # Show decision
            show_decision_reveal(decision, prob)
        else:
            # Run new analysis
            if uploaded is not None:
                tmp_path = Path("uploads")
                tmp_path.mkdir(exist_ok=True)
                video_path = str(tmp_path / uploaded.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded.getbuffer())
            elif demo_choice != "(none)":
                video_path = str(samples_dir / demo_choice)
            else:
                st.error("Please upload or select a video.")
                return

            # Show loading animation
            loading_placeholder = show_loading_animation()
            time.sleep(0.5)  # Brief pause for effect

            # Load frames
            frames = load_video_frames(video_path, max_frames=None if max_frames == 0 else max_frames)
            if len(frames) == 0:
                loading_placeholder.empty()
                st.error("Failed to read video frames.")
                return

            # Initialize components
            detector = BallDetector(weights_path=weights_path, confidence_threshold=conf_thr, iou_threshold=iou_thr)
            tracker = SimpleBallTracker(max_history=100)

            # Ball tracking
            track_points: List[Tuple[int, int]] = []
            progress = st.progress(0, text="Detecting ball and building track...")
            for idx, frame in enumerate(frames):
                dets = detector.detect(frame)
                centers = [d.center for d in dets]
                pt = tracker.update(centers)
                if pt is not None:
                    track_points.append(pt)
                if idx % 5 == 0:
                    progress.progress(min((idx + 1) / len(frames), 1.0))
            progress.progress(1.0)

            if len(track_points) < 6:
                loading_placeholder.empty()
                st.error("Could not build a reliable ball track. Try a clearer clip or adjusted thresholds.")
                return

            # Bounce detection
            bounce_index = detect_bounce_point(track_points)
            if bounce_index is None:
                bounce_index = max(2, len(track_points) // 3)

            # Trajectory fit and prediction (LSTM if available, else polynomial)
            hist_points = track_points[: bounce_index + 1]
            traj = fit_trajectory(
                hist_points, 
                model_path=cfg.lstm_model_path if cfg.use_lstm_trajectory else None,
                use_lstm=cfg.use_lstm_trajectory
            )
            future_points = traj.predict(num_future=max(15, len(hist_points)))

            # Impact and stumps prediction
            h, w = frames[0].shape[:2]
            stumps = estimate_stumps_region(w, cfg.stumps_x_center_ratio, cfg.stumps_width_ratio)
            hit_point = predict_stump_intersection(track_points, future_points, stumps)

            would_hit_stumps = hit_point is not None
            distance_to_stumps_px = 9999.0
            if not would_hit_stumps:
                cx = (stumps.x_left + stumps.x_right) / 2.0
                distance_to_stumps_px = min(abs(p[0] - cx) for p in (track_points + future_points))

            # Simplified heuristics
            pitched_zone = "in-line"
            impact_in_line = True

            # Classifier
            clf = LBWClassifier(cfg.xgb_model_path)
            feats = LBWFeatures(
                pitched_in_line=1 if pitched_zone == "in-line" else 0,
                impact_in_line=1 if impact_in_line else 0,
                would_hit_stumps=1 if would_hit_stumps else 0,
                distance_to_stumps_px=float(distance_to_stumps_px),
            )
            pred = clf.predict(feats)
            decision = pred["label"]
            prob = float(pred["probability"])

            # Clear loading and show decision
            loading_placeholder.empty()
            show_decision_reveal(decision, prob)

            # Store in analytics
            review_record = create_review_record(
                video_name=Path(video_path).name,
                decision=decision,
                confidence=prob,
                would_hit_stumps=would_hit_stumps,
                distance_to_stumps_px=distance_to_stumps_px,
                num_track_points=len(track_points),
                bounce_index=bounce_index
            )
            st.session_state.analytics.add_review(review_record)
            st.session_state.analysis_results = {
                'video_path': video_path,
                'frames': frames,
                'track_points': track_points,
                'future_points': future_points,
                'decision': decision,
                'confidence': prob,
                'bounce_index': bounce_index,
                'stumps': stumps,
                'would_hit_stumps': would_hit_stumps,
                'distance_to_stumps_px': distance_to_stumps_px,
                'pitched_zone': pitched_zone,
                'impact_in_line': impact_in_line,
                'frame_height': h,
                'frame_width': w
            }
            st.session_state.video_path = video_path
            st.session_state.pdf_path = None  # Reset PDF path

        # Split view layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📹 Video Analysis")
            
            # Video timeline scrubber
            if len(frames) > 1:
                frame_idx = st.slider(
                    "Frame", 
                    0, 
                    len(frames) - 1, 
                    len(frames) - 1,
                    key="frame_slider",
                    help="Scrub through video frames"
                )
                
                # Mark key events
                key_events = []
                if bounce_index is not None:
                    key_events.append(("Bounce", bounce_index))
                if len(track_points) > 0:
                    key_events.append(("Impact", len(track_points) - 1))
                
                if key_events:
                    st.caption("Key Events: " + ", ".join([f"{name} at frame {idx}" for name, idx in key_events]))
                
                # Show selected frame with overlay
                tN = min(frame_idx + 1, len(track_points))
                pN = min(max(0, frame_idx - len(track_points) // 3), len(future_points))
                vis_frame = draw_overlay(
                    frames[frame_idx],
                    track_points[:tN],
                    future_points[:pN],
                    stumps.x_left,
                    stumps.x_right,
                    decision if frame_idx == len(frames) - 1 else None,
                    smooth_curve=True
                )
                st.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            else:
                vis_frame = draw_overlay(
                    frames[-1],
                    track_points,
                    future_points,
                    stumps.x_left,
                    stumps.x_right,
                    decision,
                    smooth_curve=True
                )
                st.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), width='stretch')

        with col2:
            st.subheader("📈 Trajectory Analysis")
            
            # 3D trajectory plot
            fig_3d = create_3d_trajectory_plot(
                track_points,
                future_points,
                bounce_index=bounce_index,
                frame_height=h,
                frame_width=w,
                decision=decision,
                confidence=prob
            )
            st.plotly_chart(fig_3d, width='stretch')

        # Additional visualizations
        st.subheader("🔍 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trajectory_img = create_complete_trajectory_image(
                track=track_points,
                predicted=future_points,
                stumps_x_left=stumps.x_left,
                stumps_x_right=stumps.x_right,
                decision=decision,
                bounce_index=bounce_index,
                frame_height=h,
                frame_width=w,
            )
            st.image(trajectory_img, caption="Complete Ball Trajectory Analysis", width='stretch')
        
        with col2:
            # Comparison plot
            fig_comp = create_trajectory_comparison_plot(
                track_points,
                future_points,
                frame_height=h,
                frame_width=w,
                decision=decision
            )
            st.plotly_chart(fig_comp, width='stretch')

        # Export & Report section
        st.subheader("💾 Export & Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎬 Render Replay Video"):
                out_dir = Path("outputs")
                out_dir.mkdir(exist_ok=True)
                out_path = str(out_dir / (Path(video_path).stem + "_overlay.mp4"))
                fps = cfg.output_video_fps
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps_cap = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if fps_cap and fps_cap > 1:
                        fps = int(fps_cap)
                except Exception:
                    pass
                path = render_overlay_video(frames, track_points, future_points, stumps.x_left, stumps.x_right, decision, fps, out_path, smooth_curve=True)
                st.success(f"✅ Saved replay to {path}")
                with open(path, "rb") as f:
                    st.download_button("📥 Download Replay MP4", f, file_name=Path(path).name)

        with col2:
            # Check if PDF already exists or generate it
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                current_video = Path(video_path).name
                
                # Generate PDF if not already generated or if video changed
                if st.session_state.pdf_path is None or not Path(st.session_state.pdf_path).exists():
                    if st.button("📄 Generate PDF Report"):
                        with st.spinner("Generating PDF report..."):
                            out_dir = Path("outputs")
                            out_dir.mkdir(exist_ok=True)
                            pdf_path = str(out_dir / (Path(video_path).stem + "_report.pdf"))
                            
                            explanation_text = generate_explanation(
                                ExplanationInputs(
                                    pitched_zone=results['pitched_zone'],
                                    impact_in_line=results['impact_in_line'],
                                    would_hit_stumps=results['would_hit_stumps'],
                                    decision=results['decision'],
                                    model_confidence=results['confidence'],
                                    track_points=results['track_points'],
                                    future_points=results['future_points'],
                                    bounce_index=results['bounce_index'],
                                    distance_to_stumps_px=results['distance_to_stumps_px'],
                                ),
                                use_ai=False,
                                simple=True,
                            )
                            
                            generate_pdf_report(
                                pdf_path,
                                Path(video_path).name,
                                results['decision'],
                                results['confidence'],
                                results['track_points'],
                                results['future_points'],
                                results['bounce_index'],
                                results['would_hit_stumps'],
                                results['distance_to_stumps_px'],
                                explanation_text,
                                frame_width=results['frame_width'],
                                frame_height=results['frame_height'],
                                stumps_x_left=results['stumps'].x_left,
                                stumps_x_right=results['stumps'].x_right
                            )
                            st.session_state.pdf_path = pdf_path
                            st.success(f"✅ PDF report generated!")
                            st.rerun()
                
                # Show download button if PDF exists
                if st.session_state.pdf_path and Path(st.session_state.pdf_path).exists():
                    with open(st.session_state.pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF Report", f, file_name=Path(st.session_state.pdf_path).name, key="pdf_download")
                elif st.session_state.pdf_path is None:
                    st.info("Click 'Generate PDF Report' to create a report")
            else:
                st.info("Run analysis first to generate PDF")

        with col3:
            import json
            report = {
                "video": str(video_path),
                "decision": decision,
                "confidence": prob,
                "would_hit_stumps": bool(would_hit_stumps),
                "distance_to_stumps_px": float(distance_to_stumps_px),
                "num_track_points": len(track_points),
            }
            st.download_button(
                "📥 Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name=f"{Path(video_path).stem}_lbw_report.json",
            )

        # Explanations section - Prominent Display
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #00ffaa; margin-top: 2rem;'>📋 Third Umpire Technical Report</h2>", unsafe_allow_html=True)
        
        explanation_tone = commentary_tone.lower()
        
        # AI-based detailed explanation
        explanation_ai = generate_explanation(
            ExplanationInputs(
                pitched_zone=pitched_zone,
                impact_in_line=impact_in_line,
                would_hit_stumps=would_hit_stumps,
                decision=decision,
                model_confidence=prob,
                track_points=track_points,
                future_points=future_points,
                bounce_index=bounce_index,
                distance_to_stumps_px=distance_to_stumps_px,
            ),
            use_ai=True,
            api_key=gemini_api_key if gemini_api_key else None,
            simple=False,
            tone=explanation_tone,
        )

        # Prominent Technical Report Display
        st.markdown(f"""
        <div style='
             border-radius: 15px;
             padding: 2rem;
             margin: 1.5rem 0;
             box-shadow: 0 8px 32px rgba(0, 255, 170, 0.3);
             backdrop-filter: blur(10px);
        '>
             <div style='
                 display: flex;
                 align-items: center;
                 justify-content: space-between;
                 margin-bottom: 1.5rem;
                 padding-bottom: 1rem;
                 border-bottom: 2px solid rgba(0, 255, 170, 0.3);
             '>
                 <h3 style='color: #00ffaa; margin: 0; font-size: 1.8rem;'>
                     🧠 Technical Analysis Report
                 </h3>
                 <span style='
                     background: rgba(0, 255, 170, 0.2);
                     color: #00ffaa;
                     padding: 0.5rem 1rem;
                     border-radius: 20px;
                     font-weight: bold;
                     font-size: 0.9rem;
                 '>{commentary_tone} Mode</span>
             </div>
             <div style='
                 color: #e0e0e0;
                 font-size: 1.1rem;
                 line-height: 1.8;
                 text-align: justify;
                 white-space: pre-wrap;
             '>
        """, unsafe_allow_html=True)

        # Display explanation with proper formatting
        st.markdown(explanation_ai)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Simple explanation in a separate section
        st.markdown("<h3 style='color: #00e0b5; margin-top: 2rem;'>📢 Simple Explanation for Audience</h3>", unsafe_allow_html=True)
        
        explanation_simple = generate_explanation(
            ExplanationInputs(
                pitched_zone=pitched_zone,
                impact_in_line=impact_in_line,
                would_hit_stumps=would_hit_stumps,
                decision=decision,
                model_confidence=prob,
            ),
            use_ai=False,
            simple=True,
        )
        st.markdown(explanation_simple)
        
        st.markdown("""
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.header("📊 Analytics Dashboard")
        
        stats = st.session_state.analytics.get_statistics()
        
        if stats["total_reviews"] == 0:
            st.info("No reviews yet. Run some analyses to see statistics here!")
        else:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reviews", stats["total_reviews"])
            with col2:
                st.metric("OUT Decisions", f"{stats['out_count']} ({stats['out_percentage']:.1f}%)")
            with col3:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
            with col4:
                st.metric("Hitting Stumps", f"{stats['hitting_stumps_percentage']:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Decision distribution
                fig_decision = go.Figure(data=[
                    go.Bar(
                        x=['OUT', 'NOT OUT'],
                        y=[stats['out_count'], stats['not_out_count']],
                        marker_color=['#ff416c', '#38ef7d'],
                        text=[stats['out_count'], stats['not_out_count']],
                        textposition='auto',
                    )
                ])
                fig_decision.update_layout(
                    title="Decision Distribution",
                    xaxis_title="Decision",
                    yaxis_title="Count",
                    template='plotly_dark'
                )
                st.plotly_chart(fig_decision, width='stretch')
            
            with col2:
                # Confidence distribution
                reviews = st.session_state.analytics.reviews
                if reviews:
                    confidences = [r.confidence for r in reviews]
                    fig_conf = go.Figure(data=[
                        go.Histogram(
                            x=confidences,
                            nbinsx=20,
                            marker_color='#667eea',
                        )
                    ])
                    fig_conf.update_layout(
                        title="Confidence Distribution",
                        xaxis_title="Confidence",
                        yaxis_title="Frequency",
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig_conf, width='stretch')
            
            # Recent reviews table
            st.subheader("Recent Reviews")
            recent = st.session_state.analytics.get_recent_reviews(10)
            if recent:
                df = pd.DataFrame([{
                    'Video': r.video_name,
                    'Decision': r.decision,
                    'Confidence': f"{r.confidence:.1%}",
                    'Would Hit Stumps': 'Yes' if r.would_hit_stumps else 'No',
                    'Timestamp': r.timestamp
                } for r in reversed(recent)])
                st.dataframe(df, width='stretch', hide_index=True)

    with tab3:
        st.header("📚 Review History")
        
        reviews = st.session_state.analytics.reviews
        if not reviews:
            st.info("No review history yet.")
        else:
            for i, review in enumerate(reversed(reviews)):
                with st.expander(f"Review #{len(reviews) - i}: {review.video_name} - {review.decision} ({review.confidence:.1%})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Decision:** {review.decision}")
                        st.write(f"**Confidence:** {review.confidence:.1%}")
                        st.write(f"**Would Hit Stumps:** {'Yes' if review.would_hit_stumps else 'No'}")
                    with col2:
                        st.write(f"**Distance to Stumps:** {review.distance_to_stumps_px:.1f} px")
                        st.write(f"**Track Points:** {review.num_track_points}")
                        st.write(f"**Timestamp:** {review.timestamp}")


if __name__ == "__main__":
    main()

