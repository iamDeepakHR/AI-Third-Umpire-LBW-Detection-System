from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import requests
import os

# Optional: Only needed if running inside Streamlit
try:
    import streamlit as st
    API_KEY = st.secrets.get("apikey", None)
except:
    API_KEY = os.getenv("OPENROUTER_API_KEY")


# =========================
# 📊 INPUT STRUCTURE
# =========================
@dataclass
class ExplanationInputs:
    pitched_zone: str
    impact_in_line: bool
    would_hit_stumps: bool
    decision: str
    model_confidence: float
    track_points: Optional[List] = None
    future_points: Optional[List] = None
    bounce_index: Optional[int] = None
    distance_to_stumps_px: Optional[float] = None


# =========================
# 🔑 OPENROUTER CONFIG
# =========================
URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://ai-third-umpire-lbw-detection-system-5bbb7uthfyjfhbkknfncop.streamlit.app/",
    "X-Title": "AI Third Umpire LBW"
}


# =========================
# 🧠 MAIN FUNCTION
# =========================
def generate_explanation(
    inputs: ExplanationInputs,
    use_ai: bool = False,
    simple: bool = True,
    tone: str = "analyst",
):
    """
    Generate LBW explanation:
    - Rule-based (simple)
    - OpenRouter AI (DeepSeek)
    """

    # -------------------------
    # 📌 BASE SUMMARY
    # -------------------------
    base = (
        f"Ball pitched: {inputs.pitched_zone}. "
        f"Impact in line: {'Yes' if inputs.impact_in_line else 'No'}. "
        f"Would hit stumps: {'Yes' if inputs.would_hit_stumps else 'No'}. "
        f"Decision: {inputs.decision} "
        f"(Confidence: {inputs.model_confidence:.2f})"
    )

    # -------------------------
    # 📘 SIMPLE RULE-BASED
    # -------------------------
    if simple or not use_ai:
        if inputs.impact_in_line and inputs.would_hit_stumps:
            msg = "The ball hit in line and would go on to hit the stumps. So it's OUT."
        elif not inputs.impact_in_line:
            msg = "The ball hit outside the line of the stumps. So it's NOT OUT."
        else:
            msg = "The ball would miss the stumps. So it's NOT OUT."

        return f"{base}\n\n📘 Simple Explanation:\n{msg} ({inputs.model_confidence:.0%})"

    # -------------------------
    # ❗ CHECK API KEY
    # -------------------------
    if not API_KEY:
        return base + "\n\n[Error: OPENROUTER_API_KEY not set]"

    try:
        # Tone selection
        if tone.lower() == "commentator":
            system_role = "You are a cricket commentator explaining LBW like live TV."
        else:
            system_role = "You are a professional cricket analyst explaining LBW decisions."

        prompt = f"""
Explain the LBW decision clearly.

Details:
- Pitch zone: {inputs.pitched_zone}
- Impact in line: {inputs.impact_in_line}
- Would hit stumps: {inputs.would_hit_stumps}
- Confidence: {inputs.model_confidence}
- Distance to stumps: {inputs.distance_to_stumps_px}

Trajectory:
- Track points: {len(inputs.track_points) if inputs.track_points else 'N/A'}
- Bounce index: {inputs.bounce_index}

Final Decision: {inputs.decision}

Give:
1. Clear explanation
2. Key reasoning
3. Final conclusion
"""

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(
            URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return base + f"\n\n[API Error {response.status_code}: {response.text}]"

    except Exception as e:
        return base + f"\n\n[Error: {str(e)}]"
