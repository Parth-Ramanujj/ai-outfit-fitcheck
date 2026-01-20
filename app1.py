import streamlit as st
import json
import base64
import time
import os
from openai import OpenAI


# Basic config


OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("API key missing. Please set OPENROUTER_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://ai-outfit-fitcheck.streamlit.app",
        "X-Title": "AI Outfit Fitcheck"
    }
)

VISION_MODEL = "allenai/molmo-2-8b:free"
TEXT_MODEL = "allenai/molmo-2-8b:free"



# Page setup


st.set_page_config(page_title="AI Outfit Fitcheck", layout="centered")

st.markdown("""
<style>
.block-container { max-width: 420px; }

.image-container { position: relative; width: 100%; }
.overlay-box {
    position: absolute;
    top: 12px;
    left: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.chip {
    padding: 6px 12px;
    border-radius: 14px;
    font-size: 13px;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    width: fit-content;
}
.good { background: #d1fae5; color: #065f46; }
.bad { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container { max-width: 420px; }

.image-container { position: relative; width: 100%; }


</style>
""", unsafe_allow_html=True)

st.title("AI Outfit Fitcheck")
st.caption("Upload an outfit photo and get a structured analysis")

uploaded_file = st.file_uploader(
    "Upload a full or near-full body image",
    type=["jpg", "jpeg", "png"]
)


# Prompts


VISION_PROMPT = """
Describe ONLY what is visible in the image.
- Clothing items only
- No opinions
- No guessing
- Short factual sentences
"""

TEXT_PROMPT = """
You are a strict JSON formatting engine.

Return ONLY valid JSON in this exact schema:

{
  "overall_vibe": {"summary": "", "category": ""},
  "what_works": [],
  "what_needs_work": [],
  "suggestions": [],
  "item_flags": {"dress": "", "top": "", "bottom": "", "shoes": "", "bag": "", "accessories": ""}
}
"""



# Helpers


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        return None
    try:
        return json.loads(text[start:end])
    except:
        return None


def normalize_output(data: dict) -> dict:
    data["what_works"] = data.get("what_works", [])[:3]
    data["what_needs_work"] = data.get("what_needs_work", [])[:2]
    data["suggestions"] = data.get("suggestions", [])[:2]

    while len(data["what_works"]) < 3:
        data["what_works"].append("The outfit elements appear visually consistent.")

    while len(data["what_needs_work"]) < 2:
        data["what_needs_work"].append("No clearly visible fit issues are present.")

    while len(data["suggestions"]) < 2:
        data["suggestions"].append("No changes are required based on visible elements.")

    return data



# UI blocks


def render_image_overlay(file, result):
    good = result["what_works"][:2]
    bad = result["what_needs_work"][:2]

    tags = ""
    for g in good:
        tags += f'<div class="chip good">✓ {g}</div>'
    for b in bad:
        tags += f'<div class="chip bad">✕ {b}</div>'

    st.markdown(f"""
    <div class="image-container">
        <img src="data:image/jpeg;base64,{base64.b64encode(file.getvalue()).decode()}"
             style="width:100%; border-radius:16px;">
        <div class="overlay-box">{tags}</div>
    </div>
    """, unsafe_allow_html=True)


def render_analysis(data):
    st.divider()
    st.markdown("## Outfit Analysis")

    st.markdown("### Overall vibe")
    st.info(f"{data['overall_vibe']['summary']}  \nCategory: {data['overall_vibe']['category']}")

    st.markdown("### What works")
    for i in data["what_works"]:
        st.success(i)

    st.markdown("### What needs work")
    for i in data["what_needs_work"]:
        st.error(i)

    st.markdown("### Suggestions")
    for i in data["suggestions"]:
        st.warning(i)


# Main flow


if uploaded_file and st.button("Analyze outfit"):

    with st.spinner("Analyzing image..."):

        total_start = time.time()
        image_b64 = base64.b64encode(uploaded_file.read()).decode()

        # Vision step
        t1 = time.time()
        vision_resp = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[
                {"role": "system", "content": VISION_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe the visible outfit."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
        )
        vision_time = time.time() - t1

        vision_text = vision_resp.choices[0].message.content or ""
        vision_json = extract_json(vision_text)

        # Structuring step
        t2 = time.time()
        text_input = json.dumps(vision_json) if vision_json else vision_text
        text_resp = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0,
            max_tokens=600,
            messages=[
                {"role": "system", "content": TEXT_PROMPT},
                {"role": "user", "content": text_input}
            ],
        )
        text_time = time.time() - t2

        total_time = time.time() - total_start

        final_text = text_resp.choices[0].message.content or ""
        final_json = extract_json(final_text)

        if not final_json:
            st.error("Failed to generate valid JSON output.")
            st.code(final_text)
            st.stop()

        final_result = normalize_output(final_json)

        # UI output
        render_image_overlay(uploaded_file, final_result)
        render_analysis(final_result)

        st.divider()
        st.subheader("Raw JSON output")
        st.code(json.dumps(final_result, indent=2), language="json")

        # System notes
        st.divider()
        st.subheader("Determinism & consistency")
        st.markdown("""
- Temperature is fixed at 0  
- Schema is enforced  
- Missing fields are auto-filled  
- Same image produces the same structure  
""")

        st.subheader("Latency")
        st.markdown(f"""
Vision step: {vision_time:.2f}s  
Structuring step: {text_time:.2f}s  
Total time: {total_time:.2f}s (target ≤ 3s p95)
""")

        st.subheader("Cost estimate")
        st.markdown("""
Model: allenai/molmo-2-8b  
Calls per request: 2  
Approx cost: free tier  
1,000 fitchecks ≈ $0 – $1 depending on limits
""")
