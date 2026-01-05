from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import cv2

Point = Tuple[int, int]


def draw_overlay(frame, track: List[Point], predicted: List[Point], stumps_x_left: int, stumps_x_right: int, decision: Optional[str] = None, smooth_curve: bool = True):
    """
    Draw enhanced overlay on frame with smooth trajectory curves.
    
    Args:
        frame: Input frame
        track: Tracked ball positions
        predicted: Predicted future positions
        stumps_x_left: Left boundary of stumps
        stumps_x_right: Right boundary of stumps
        decision: OUT or NOT OUT
        smooth_curve: Whether to draw smooth curves using interpolation
    """
    vis = frame.copy()
    
    # Draw stumps region with semi-transparent overlay
    overlay = vis.copy()
    cv2.rectangle(overlay, (stumps_x_left, 0), (stumps_x_right, vis.shape[0]), (255, 255, 0), -1)
    cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
    cv2.rectangle(vis, (stumps_x_left, 0), (stumps_x_right, vis.shape[0]), (255, 255, 0), 2)

    # Draw smooth trajectory curves
    if smooth_curve and len(track) > 2:
        # Interpolate track for smooth curve
        from scipy.interpolate import interp1d
        track_x = [p[0] for p in track]
        track_y = [p[1] for p in track]
        
        if len(track) >= 3:
            # Create interpolation function
            t = np.linspace(0, 1, len(track))
            t_smooth = np.linspace(0, 1, len(track) * 3)
            
            try:
                fx = interp1d(t, track_x, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                fy = interp1d(t, track_y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                
                x_smooth = fx(t_smooth)
                y_smooth = fy(t_smooth)
                
                # Draw smooth curve
                pts = np.array([[int(x), int(y)] for x, y in zip(x_smooth, y_smooth)], np.int32)
                cv2.polylines(vis, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
            except:
                # Fallback to simple line
                pts = np.array(track, np.int32)
                cv2.polylines(vis, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            pts = np.array(track, np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        # Draw simple line
        if len(track) > 1:
            pts = np.array(track, np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Draw track points
    for p in track:
        cv2.circle(vis, p, 4, (0, 255, 0), -1)
        cv2.circle(vis, p, 5, (255, 255, 255), 1)

    # Draw predicted path with smooth curve
    if predicted and len(predicted) > 1:
        if smooth_curve and len(predicted) >= 3:
            pred_x = [p[0] for p in predicted]
            pred_y = [p[1] for p in predicted]
            t = np.linspace(0, 1, len(predicted))
            t_smooth = np.linspace(0, 1, len(predicted) * 3)
            
            try:
                from scipy.interpolate import interp1d
                fx = interp1d(t, pred_x, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                fy = interp1d(t, pred_y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                
                x_smooth = fx(t_smooth)
                y_smooth = fy(t_smooth)
                
                pts = np.array([[int(x), int(y)] for x, y in zip(x_smooth, y_smooth)], np.int32)
                cv2.polylines(vis, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)
            except:
                pts = np.array(predicted, np.int32)
                cv2.polylines(vis, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            pts = np.array(predicted, np.int32)
            cv2.polylines(vis, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)
        
        # Draw predicted points
        for p in predicted:
            cv2.circle(vis, p, 3, (0, 165, 255), -1)

    # Draw decision with enhanced styling
    if decision:
        color = (0, 0, 255) if decision == "OUT" else (0, 200, 0)
        bg_color = (0, 0, 100) if decision == "OUT" else (0, 100, 0)
        
        # Background rectangle for text
        text = f"Decision: {decision}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(vis, (15, 15), (25 + text_width, 50 + text_height), bg_color, -1)
        cv2.rectangle(vis, (15, 15), (25 + text_width, 50 + text_height), color, 2)
        
        cv2.putText(vis, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    return vis


def create_complete_trajectory_image(track: List[Point], predicted: List[Point], stumps_x_left: int, stumps_x_right: int, 
                                   decision: Optional[str] = None, bounce_index: Optional[int] = None, 
                                   frame_height: int = 600, frame_width: int = 800) -> np.ndarray:
    """
    Create a complete ball trajectory visualization with enhanced graphics
    """
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_facecolor('white')
    
    # Extract coordinates
    track_x = [p[0] for p in track] if track else []
    track_y = [p[1] for p in track] if track else []
    pred_x = [p[0] for p in predicted] if predicted else []
    pred_y = [p[1] for p in predicted] if predicted else []
    
    # Draw cricket pitch outline
    pitch_width = frame_width * 0.15
    pitch_height = frame_height * 0.6
    pitch_x = (frame_width - pitch_width) / 2
    pitch_y = frame_height * 0.2
    
    # Main pitch rectangle
    pitch_rect = FancyBboxPatch((pitch_x, pitch_y), pitch_width, pitch_height,
                               boxstyle="round,pad=2", linewidth=3, 
                               edgecolor='green', facecolor='lightgreen', alpha=0.3)
    ax.add_patch(pitch_rect)
    
    # Stumps area
    stumps_width = stumps_x_right - stumps_x_left
    stumps_rect = FancyBboxPatch((stumps_x_left, pitch_y), stumps_width, pitch_height,
                                boxstyle="round,pad=1", linewidth=2,
                                edgecolor='yellow', facecolor='yellow', alpha=0.5)
    ax.add_patch(stumps_rect)
    
    # Draw ball trajectory
    if track:
        # Actual ball path
        ax.plot(track_x, track_y, 'o-', color='blue', linewidth=3, markersize=6, 
                label='Actual Ball Path', alpha=0.8)
        
        # Mark bounce point if available
        if bounce_index is not None and 0 <= bounce_index < len(track):
            ax.plot(track_x[bounce_index], track_y[bounce_index], 's', 
                   color='red', markersize=10, label='Bounce Point', markeredgecolor='darkred')
    
    # Draw predicted path
    if predicted:
        ax.plot(pred_x, pred_y, '--', color='orange', linewidth=2, 
                label='Predicted Path', alpha=0.7)
        
        # Mark predicted impact point
        if pred_x and pred_y:
            ax.plot(pred_x[-1], pred_y[-1], '^', 
                   color='red', markersize=8, label='Predicted Impact')
    
    # Draw stumps
    stump_height = 20
    for i in range(3):
        stump_x = stumps_x_left + (i + 1) * stumps_width / 4
        ax.plot([stump_x, stump_x], [pitch_y, pitch_y + stump_height], 
               'k-', linewidth=4, label='Stumps' if i == 0 else "")
    
    # Add decision text
    if decision:
        decision_color = 'red' if decision == "OUT" else 'green'
        ax.text(0.5, 0.95, f"Decision: {decision}", transform=ax.transAxes, 
               fontsize=16, fontweight='bold', color=decision_color,
               ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=decision_color))
    
    # Add trajectory statistics
    if track:
        total_distance = sum(np.sqrt((track_x[i+1] - track_x[i])**2 + (track_y[i+1] - track_y[i])**2) 
                           for i in range(len(track_x)-1))
        ax.text(0.02, 0.98, f"Track Points: {len(track)}\nTotal Distance: {total_distance:.1f}px", 
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Set axis properties
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    ax.invert_yaxis()  # Invert Y axis to match image coordinates
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title('Complete Ball Trajectory Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Remove alpha channel to get RGB
    
    plt.close(fig)
    return buf


def render_overlay_video(frames: List, track: List[Point], predicted: List[Point], stumps_x_left: int, stumps_x_right: int, decision: Optional[str], fps: int, out_path: str, smooth_curve: bool = True) -> str:
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    if len(frames) == 0:
        raise ValueError("No frames to render")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, f in enumerate(frames):
        # progressively reveal track and prediction
        tN = min(i + 1, len(track))
        pN = min(max(0, i - len(track) // 3), len(predicted))
        vis = draw_overlay(f, track[:tN], predicted[:pN], stumps_x_left, stumps_x_right, decision if i == len(frames) - 1 else None, smooth_curve=smooth_curve)
        writer.write(vis)
    writer.release()
    return out_path


