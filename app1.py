import streamlit as st
import json
import base64
import time
import os
from openai import OpenAI

# ================= CONFIG =================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("❌ API key not configured. Please set OPENROUTER_API_KEY in secrets or environment variables.")
    st.stop()

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key="OPENROUTER_API_KEY",
# )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://ai-outfit-fitcheck.streamlit.app", 
        "X-Title": "AI Outfit Fitcheck"
    }
)


# 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
VISION_MODEL = "allenai/molmo-2-8b:free"
TEXT_MODEL = "allenai/molmo-2-8b:free"

# ================= UI =================
st.set_page_config(page_title="AI Outfit Fitcheck", layout="centered")

st.markdown("""
<style>
.block-container {
    max-width: 420px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧥 AI Outfit Fitcheck")
st.caption("Upload outfit image → Get Outfit Analysis")

uploaded_file = st.file_uploader(
    "Upload a full or near-full body image",
    type=["jpg", "jpeg", "png"]
)

# ================= PROMPTS =================

VISION_PROMPT = """
Describe ONLY what is visible in the image.

Rules:
- Clothing items only
- No opinions or styling advice
- No guessing
- Use short factual sentences
- Mention color, garment type, and fit if clearly visible
- Mention loose or fitted only if obvious
- If an item is not visible, say "not_detected"

You may respond in free text or JSON.
"""

TEXT_PROMPT = """
You are a STRICT JSON formatting engine.

Rules:
- Output ONLY valid JSON
- No explanations or extra text
- No speculation
- Do NOT evaluate items marked "not_detected"

FINAL SCHEMA (MUST MATCH):

{
  "overall_vibe": {
    "summary": "",
    "category": ""
  },
  "what_works": [],
  "what_needs_work": [],
  "suggestions": [],
  "item_flags": {
    "dress": "",
    "top": "",
    "bottom": "",
    "shoes": "",
    "bag": "",
    "accessories": ""
  }
}
"""

# ================= HELPERS =================

def extract_json_loose(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None

def is_sentence(s: str) -> bool:
    return len(s.split()) >= 4

def sanitize_final(result: dict) -> dict:

    for k in result["item_flags"]:
        result["item_flags"][k] = "visible" if result["item_flags"][k] == "visible" else "not_detected"

    for field, status in result["item_flags"].items():
        if status == "not_detected":
            for section in ["what_works", "what_needs_work", "suggestions"]:
                result[section] = [s for s in result[section] if field not in s.lower()]

    for section in ["what_works", "what_needs_work", "suggestions"]:
        result[section] = [s for s in result[section] if is_sentence(s)]

    result["what_works"] = result["what_works"][:3]
    result["what_needs_work"] = result["what_needs_work"][:2]
    result["suggestions"] = result["suggestions"][:2]

    while len(result["what_works"]) < 3:
        result["what_works"].append("Visible clothing items form a consistent appearance.")

    while len(result["what_needs_work"]) < 2:
        result["what_needs_work"].append("No clearly visible fit issues are present.")

    while len(result["suggestions"]) < 2:
        result["suggestions"].append("No changes are required based on visible elements.")

    return result

# ================= RESULT PAGE =================

# def show_outfit_page(data):
    st.divider()
    st.markdown("## 👗 Outfit Analysis")

    st.markdown("### ✨ Overall Vibe")
    st.info(f"**{data['overall_vibe']['summary']}**  \nCategory: {data['overall_vibe']['category']}")

    st.markdown("### ✅ What Works")
    for item in data["what_works"]:
        st.success(item)

    st.markdown("### ❌ What Needs Work")
    for item in data["what_needs_work"]:
        st.error(item)

    st.markdown("### 💡 How to Improve")
    for item in data["suggestions"]:
        st.warning(item)

def show_outfit_page(data):
    st.divider()
    st.markdown("## 👗 Outfit Analysis")

    # Overall vibe
    st.markdown("### ✨ Overall Vibe")
    st.info(f"**{data['overall_vibe']['summary']}**  \nCategory: {data['overall_vibe']['category']}")

    # What works
    st.markdown("### ✅ What Works")
    for item in data["what_works"]:
        st.success(item)

    # What needs work
    st.markdown("### ❌ What Needs Work")
    for item in data["what_needs_work"]:
        st.error(item)

    # Suggestions
    st.markdown("### 💡 How to Improve")
    for item in data["suggestions"]:
        st.warning(item)


# ================= RUN =================

if uploaded_file and st.button("Analyze My Outfit"):
    with st.spinner("Analyzing outfit…"):

        start = time.time()
        image_b64 = base64.b64encode(uploaded_file.read()).decode()

        vision_resp = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[
                {"role": "system", "content": VISION_PROMPT},
                {"role": "user","content": [
                        {"type": "text", "text": "Describe the visible outfit."},
                        {"type": "image_url","image_url": {"url": f"data:{uploaded_file.type};base64,{image_b64}"}}
                ]}
            ],
        )

        vision_raw = vision_resp.choices[0].message.content or ""
        vision_json = extract_json_loose(vision_raw)

        text_input = json.dumps(vision_json) if vision_json else vision_raw

        text_resp = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0,
            max_tokens=600,
            messages=[
                {"role": "system", "content": TEXT_PROMPT},
                {"role": "user", "content": text_input}
            ],
        )

        final_raw = text_resp.choices[0].message.content or ""
        final_json = extract_json_loose(final_raw)

        if not final_json:
            st.error("❌ Failed to generate valid result.")
            st.code(final_raw)
            st.stop()

        final_result = sanitize_final(final_json)
        latency = time.time() - start

        # 👉 SHOW OUTFIT ANALYSIS PAGE
        show_outfit_page(final_result)

        st.caption(f"⏱️ {latency:.2f}s • 🔐 2-step AI analysis")
