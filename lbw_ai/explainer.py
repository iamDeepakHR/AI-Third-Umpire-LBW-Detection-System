from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import google.generativeai as genai
import os


@dataclass
class ExplanationInputs:
    pitched_zone: str
    impact_in_line: bool
    would_hit_stumps: bool
    decision: str
    model_confidence: float
    track_points: Optional[list] = None
    future_points: Optional[list] = None
    bounce_index: Optional[int] = None
    distance_to_stumps_px: Optional[float] = None


def generate_explanation(
    inputs: ExplanationInputs,
    use_ai: bool = False,
    provider: str | None = None,
    api_key: str | None = None,
    simple: bool = True,  # 👈 New feature: generate easy explanation for audience
    tone: str = "analyst",  # 👈 New: "analyst" or "commentator"
) -> str:
    # --- Base Technical Summary ---
    base = (
        f"Ball pitched: {inputs.pitched_zone}. "
        f"Impact in line: {'Yes' if inputs.impact_in_line else 'No'}. "
        f"Projected to hit stumps: {'Yes' if inputs.would_hit_stumps else 'No'}. "
        f"Decision: {inputs.decision} (confidence {inputs.model_confidence:.2f})."
    )

    # --- Simple explanation for normal users ---
    if simple:
        if inputs.impact_in_line and inputs.would_hit_stumps:
            user_expl = (
                "The ball landed in a fair area and hit the batter’s leg in front of the stumps. "
                "The system predicts the ball would have gone on to hit the stumps. "
                f"So, the decision is **{inputs.decision.upper()}**."
            )
        elif not inputs.impact_in_line:
            user_expl = (
                "The ball hit the batter outside the line of the stumps. "
                "That means it’s less likely to hit the stumps, so it’s **NOT OUT**."
            )
        else:
            user_expl = (
                "The ball didn’t seem likely to hit the stumps after bouncing. "
                "Hence, it’s **NOT OUT**."
            )

        user_expl += f" (Confidence: {inputs.model_confidence:.0%})"
        simple_summary = f"{base}\n\n📘 Simple Explanation:\n{user_expl}"
    else:
        simple_summary = base

    # --- AI-based detailed explanation (optional) ---
    if use_ai and api_key:
        try:
            genai.configure(api_key="AIzaSyBvZi3U5THV3B89znexdK5NAo1EItyfCIU")
            model = genai.GenerativeModel("gemini-2.5-flash")

            # Tone-specific prompts
            if tone.lower() == "commentator":
                role_prompt = """You are a charismatic cricket commentator with an engaging, natural speaking style. 
                Make your explanation exciting and accessible, like you're narrating live on TV. 
                Use phrases like "What a delivery!", "The ball has done enough", "That's hitting the stumps!", etc."""
            else:  # analyst mode (default)
                role_prompt = """You are a professional cricket third umpire and technical analyst. 
                Provide a detailed, technical explanation with precise terminology and data-driven insights."""

            prompt = f"""
            {role_prompt}
            
            Give a clear, structured explanation of the following LBW decision.

            Ball Details:
            - Pitched in: {inputs.pitched_zone}
            - Impact in line: {'Yes' if inputs.impact_in_line else 'No'}
            - Would hit stumps: {'Yes' if inputs.would_hit_stumps else 'No'}
            - Confidence: {inputs.model_confidence:.2f}
            - Distance to stumps: {inputs.distance_to_stumps_px or 'N/A'}

            Technical Info:
            - Track points: {len(inputs.track_points) if inputs.track_points else 'N/A'}
            - Bounce frame: {inputs.bounce_index or 'N/A'}
            - Predicted future positions: {len(inputs.future_points) if inputs.future_points else 'N/A'}

            Decision: {inputs.decision}

            Please provide:
            1. A {tone}-style explanation of the decision
            2. Key technical points that led to this decision
            3. A clear conclusion about why it's OUT or NOT OUT
            """

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"{simple_summary}\n\n[AI Explanation Error: {str(e)}]"

    return simple_summary
