import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.preprocessing import join_ingredients_for_model, split_ingredients
from src.safety_score import calculate_safety_score

# Path to the 10-class model
MODEL_PATH = "models/tfidf_multiclass_model.joblib"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    # Safety check: make sure we actually loaded the 10-class model
    if not hasattr(model, "classes_") or len(model.classes_) != 10:
        raise ValueError(
            f"Incorrect model loaded from {MODEL_PATH}. "
            f"Expected 10-class model but got {len(getattr(model, 'classes_', []))} classes.\n"
            "Make sure ONLY tfidf_multiclass_model.joblib exists in /models."
        )

    return model


# -------------------------------------------------------------
# 🔴 KEYWORD LISTS
# -------------------------------------------------------------
UNSAFE_KEYWORDS = [
    "lauric acid",
    "myristic acid",
    "stearic acid",
    "oleic acid",
    "isopropyl myristate",
    "cetyl alcohol",
    "cetearyl alcohol",   # important FA trigger
    "glyceryl stearate",
    "polysorbate",
    "sorbitan",
]

NEUTRAL_RISK = [
    "dimethicone",
    "caprylic/capric triglyceride",
    "fragrance",
]

# -------------------------------------------------------------
# 🏷️ LABEL DESCRIPTIONS
# -------------------------------------------------------------
LABEL_INFO = {
    "safe": {
        "name": "Safe",
        "desc": "Low-risk, non-comedogenic, suitable for most skin types."
    },
    "neutral": {
        "name": "Neutral",
        "desc": "Does not strongly help or harm skin — minimal overall impact."
    },
    "malassezia_trigger": {
        "name": "Fungal Acne Trigger",
        "desc": "Contains ingredients that may feed Malassezia and worsen fungal acne."
    },
    "comedogenic": {
        "name": "Comedogenic",
        "desc": "Higher chance of clogging pores; acne-prone skin may react."
    },
    "irritant": {
        "name": "Irritant",
        "desc": "May cause redness, stinging or sensitivity in reactive skin."
    },
    "fragrance_heavy": {
        "name": "Fragrance Heavy",
        "desc": "Contains strong fragrance; can be irritating for sensitive skin."
    },
    "fatty_acid": {
        "name": "Fatty-Acid Rich",
        "desc": "Nourishing but can be risky for fungal acne or clogged pores."
    },
    "emollient_heavy": {
        "name": "Emollient Heavy",
        "desc": "Very moisturising and occlusive — good for dry skin, heavy for oily skin."
    },
    "surfactant": {
        "name": "Surfactant Based",
        "desc": "Contains cleansing agents typically used in washes/shampoos."
    },
    "preservative": {
        "name": "Preservative Focused",
        "desc": "Formula where preservatives are the main functional ingredients."
    },
}


# -------------------------------------------------------------
# 📘 EXPLANATION GENERATOR
# -------------------------------------------------------------
def generate_explanation(ingredients, score):
    ingredients_lower = [i.lower() for i in ingredients]

    detected_strong = [
        bad for bad in UNSAFE_KEYWORDS if any(bad in ing for ing in ingredients_lower)
    ]
    detected_mild = [
        mid for mid in NEUTRAL_RISK if any(mid in ing for ing in ingredients_lower)
    ]

    if score >= 8:
        explanation = (
            "This product appears **low risk** for fungal acne. "
            "No major fungal acne triggers were detected."
        )
    elif score >= 5:
        explanation = (
            "This product has a **moderate risk** rating. "
            "Some ingredients may cause issues for sensitive or acne-prone skin."
        )
        if detected_mild:
            explanation += f" Mild-risk ingredients: {', '.join(detected_mild)}."
    else:
        explanation = (
            "This product is rated **high risk** for fungal acne. "
            "It contains fatty acids, esters or other compounds known to feed Malassezia."
        )
        if detected_strong:
            explanation += f" High-risk ingredients: {', '.join(detected_strong)}."

    return explanation


# -------------------------------------------------------------
# 🎨 INGREDIENT HIGHLIGHTING (chips)
# -------------------------------------------------------------
def highlight_ingredients_html(ingredients):
    blocks = []
    for ing in ingredients:
        lower = ing.lower()

        if any(bad in lower for bad in UNSAFE_KEYWORDS):
            colour = "#fee2e2"  # light red
            border = "#ef4444"
            icon = "❌"
        elif any(mid in lower for mid in NEUTRAL_RISK):
            colour = "#fef9c3"  # light yellow
            border = "#eab308"
            icon = "⚠️"
        else:
            colour = "#dcfce7"  # light green
            border = "#22c55e"
            icon = "✅"

        blocks.append(
            f"""
            <div style="
                background:{colour};
                border:1px solid {border};
                padding:6px 10px;
                border-radius:999px;
                display:inline-flex;
                align-items:center;
                margin:4px 6px 4px 0;
                font-size:0.9rem;
            ">
                <span style="margin-right:6px;">{icon}</span> {ing}
            </div>
            """
        )
    return blocks


# -------------------------------------------------------------
# 🧠 MODEL CONFIDENCE BADGE
# -------------------------------------------------------------
def get_confidence_badge(max_prob: float):
    if max_prob >= 0.80:
        colour = "#22c55e"
        label = "High model confidence"
    elif max_prob >= 0.60:
        colour = "#eab308"
        label = "Moderate model confidence"
    else:
        colour = "#ef4444"
        label = "Low model confidence"

    return f"""
    <div style="
        background-color:{colour}20;
        color:{colour.replace('#', '#')};
        padding:6px 12px;
        border-radius:999px;
        display:inline-block;
        font-weight:600;
        font-size:0.85rem;
        margin-top:6px;
    ">
        {label} ({max_prob:.2f})
    </div>
    """


# -------------------------------------------------------------
# 🌈 RISK BADGE
# -------------------------------------------------------------
def get_risk_badge(score: int):
    if score >= 8:
        badge_color = "#22c55e"
        label = "Low Risk"
    elif score >= 5:
        badge_color = "#eab308"
        label = "Moderate Risk"
    else:
        badge_color = "#ef4444"
        label = "High Risk"

    html = f"""
    <div style="
        background-color:{badge_color}20;
        color:{badge_color};
        padding:8px 14px;
        border-radius:999px;
        display:inline-block;
        font-weight:600;
        font-size:0.9rem;
        margin-top:4px;
    ">
        {label}
    </div>
    """
    return html


# -------------------------------------------------------------
# 🚀 MAIN APP (Apple-clean UI + toggle)
# -------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Fungal Acne Ingredient Classifier",
        page_icon="🧴",
        layout="centered",
    )

    # ---------- Global CSS ----------
    st.markdown(
        """
        <style>
        /* Make the app feel more "Apple-clean" */
        .main {
            background-color: #f5f5f7;
        }
        section[data-testid="stSidebar"] {
            background-color: #f5f5f7;
        }
        .fa-card {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .fa-header {
            font-size: 2.0rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        .fa-subtle {
            color: #6b7280;
            font-size: 0.95rem;
        }
        .fa-section-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Top hero / header ----------
    st.markdown(
        """
        <div style="padding: 10px 4px 4px 4px;">
            <div class="fa-header">🧴 Fungal Acne Ingredient Classifier</div>
            <p class="fa-subtle">
                Paste any skincare ingredients list and get a fungal-acne safety rating,
                ingredient breakdown, and (optionally) an ML expert view with probabilities & LIME.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Input card ----------
    with st.container():
        st.markdown('<div class="fa-card">', unsafe_allow_html=True)

        cols_top = st.columns([3, 1])
        with cols_top[0]:
            text = st.text_area(
                "Ingredients list",
                height=140,
                placeholder="e.g. Aqua, Glycerin, Niacinamide, Panthenol, ...",
            )
        with cols_top[1]:
            st.write(" ")
            st.write(" ")
            expert_mode = st.toggle("Expert Mode", value=False, help="Show probabilities, charts & LIME explanation")

        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if not analyze_clicked:
        return

    if text.strip() == "":
        st.warning("Please enter an ingredients list first.")
        return

    # ---------- Prediction pipeline ----------
    model = load_model()
    clean_text = join_ingredients_for_model(text)

    pred_label = model.predict([clean_text])[0]
    pred_probs = model.predict_proba([clean_text])[0]
    classes = model.classes_

    ingredients = split_ingredients(text)
    score = calculate_safety_score(text)
    highlighted_html = highlight_ingredients_html(ingredients)

    info = LABEL_INFO.get(pred_label, {"name": pred_label, "desc": ""})
    display_name = info["name"]
    desc = info["desc"]

    max_prob = float(np.max(pred_probs))
    sorted_idx = pred_probs.argsort()[::-1]

    # ---------- Two-column result cards ----------
    col1, col2 = st.columns(2)

    # LEFT CARD: Prediction + confidence
    with col1:
        st.markdown('<div class="fa-card">', unsafe_allow_html=True)
        st.markdown('<div class="fa-section-title">🔍 Prediction</div>', unsafe_allow_html=True)
        st.markdown(f"**Result:** `{display_name}`")
        if desc:
            st.markdown(f"<p class='fa-subtle'>{desc}</p>", unsafe_allow_html=True)

        st.markdown(get_confidence_badge(max_prob), unsafe_allow_html=True)

        if expert_mode:
            st.markdown('<div class="fa-section-title" style="margin-top:12px;">📊 Class probabilities</div>', unsafe_allow_html=True)
            for idx in sorted_idx:
                st.write(f"- **{classes[idx]}**: {pred_probs[idx]:.3f}")

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT CARD: Score + risk + explanation
    with col2:
        st.markdown('<div class="fa-card">', unsafe_allow_html=True)
        st.markdown('<div class="fa-section-title">🧪 Fungal Acne Score</div>', unsafe_allow_html=True)
        st.markdown(f"### {score} / 10")
        st.markdown(get_risk_badge(score), unsafe_allow_html=True)

        st.markdown('<div class="fa-section-title" style="margin-top:14px;">📘 Explanation</div>', unsafe_allow_html=True)
        st.write(generate_explanation(ingredients, score))

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Ingredient breakdown card ----------
    st.markdown('<div class="fa-card">', unsafe_allow_html=True)
    st.markdown('<div class="fa-section-title">🧬 Ingredient Breakdown</div>', unsafe_allow_html=True)
    st.caption("Green = generally safe, Yellow = mild/questionable, Red = commonly linked to fungal acne issues.")

    chips_html = "".join(highlighted_html)
    st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Expert Mode: chart + LIME ----------
    if expert_mode:
        # Probabilities chart card
        st.markdown('<div class="fa-card">', unsafe_allow_html=True)
        st.markdown('<div class="fa-section-title">📈 Probability distribution</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame(
            {
                "label": classes[sorted_idx],
                "probability": pred_probs[sorted_idx],
            }
        ).set_index("label")
        st.bar_chart(prob_df)

        st.markdown("</div>", unsafe_allow_html=True)

        # LIME explanation card
        with st.expander("🔎 LIME explanation — why did the model predict this?"):
            st.caption("This shows which words/phrases pushed the prediction towards the top label.")
            explainer = LimeTextExplainer(class_names=list(classes))

            def predict_proba_lime(text_list):
                processed = [join_ingredients_for_model(t) for t in text_list]
                return model.predict_proba(processed)

            exp = explainer.explain_instance(
                text,
                predict_proba_lime,
                num_features=8,
                top_labels=1,
            )

            top_label_idx = exp.top_labels[0]
            top_label_name = classes[top_label_idx]
            st.write(f"Top label explained: **{top_label_name}**")

            lime_df = pd.DataFrame(exp.as_list(label=top_label_idx), columns=["feature", "weight"])
            st.dataframe(lime_df)

    # ---------- Tiny glossary card ----------
    st.markdown('<div class="fa-card">', unsafe_allow_html=True)
    st.markdown('<div class="fa-section-title">📗 Quick glossary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Comedogenic** — tends to clog pores and cause breakouts.  
        - **Fungal acne trigger** — can feed Malassezia yeast and worsen fungal acne.  
        - **Emollient** — moisturising and softening; can feel heavy on oily skin.  
        - **Surfactant** — cleansing ingredient, common in washes/shampoos.  
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
