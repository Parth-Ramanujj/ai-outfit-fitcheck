import streamlit as st
import json
import base64
import time
import os
from openai import OpenAI

# ================= CONFIG =================

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

# ================= UI =================

st.set_page_config(page_title="AI Outfit Fitcheck", layout="centered")

st.markdown("""
<style>
.block-container { max-width: 420px; }

.image-wrap { position: relative; width:100%; }
.overlay {
    position:absolute;
    top:12px;
    left:12px;
    display:flex;
    flex-wrap:wrap;
    gap:8px;
}
.tag {
    padding:6px 12px;
    border-radius:999px;
    font-size:12px;
    font-weight:600;
    box-shadow:0 4px 10px rgba(0,0,0,.2);
}
.good { background:#d1fae5; color:#065f46; }
.bad { background:#fee2e2; color:#991b1b; }
img { border-radius:16px; }
</style>
""", unsafe_allow_html=True)

st.title("🧥 AI Outfit Fitcheck")
st.caption("Upload an outfit photo and get a structured analysis")

uploaded_file = st.file_uploader("Upload outfit image", type=["jpg","jpeg","png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded preview", use_container_width=True)

# ================= PROMPTS =================

VISION_PROMPT = """
Describe ONLY what is visible in the image.
- Clothing items only
- No opinions
- No guessing
- Short factual sentences
"""

TEXT_PROMPT = """
Return ONLY valid JSON in this exact schema:

{
  "overall_vibe": {"summary": "", "category": ""},
  "what_works": [],
  "what_needs_work": [],
  "suggestions": [],
  "item_flags": {"dress": "", "top": "", "bottom": "", "shoes": "", "bag": "", "accessories": ""}
}
"""

# ================= HELPERS =================

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        return None
    try:
        return json.loads(text[start:end])
    except:
        return None

def normalize(data):
    data["what_works"] = data.get("what_works", [])[:3]
    data["what_needs_work"] = data.get("what_needs_work", [])[:2]
    data["suggestions"] = data.get("suggestions", [])[:2]

    while len(data["what_works"]) < 3:
        data["what_works"].append("Outfit elements appear visually consistent.")

    while len(data["what_needs_work"]) < 2:
        data["what_needs_work"].append("No clearly visible fit issues are present.")

    while len(data["suggestions"]) < 2:
        data["suggestions"].append("No visible changes required.")

    return data

# ================= UI BLOCKS =================

def render_overlay(image, result):
    good = result["what_works"][:2]
    bad = result["what_needs_work"][:2]

    chips = ""
    for g in good:
        chips += f'<div class="tag good">✓ {g}</div>'
    for b in bad:
        chips += f'<div class="tag bad">✕ {b}</div>'

    st.markdown(f"""
    <div class="image-wrap">
        <img src="data:image/jpeg;base64,{base64.b64encode(image.getvalue()).decode()}" style="width:100%">
        <div class="overlay">{chips}</div>
    </div>
    """, unsafe_allow_html=True)

def render_analysis(data):
    st.divider()
    st.markdown("## Outfit Analysis")

    st.info(f"**{data['overall_vibe']['summary']}**  \nCategory: {data['overall_vibe']['category']}")

    st.markdown("### ✅ What works")
    for i in data["what_works"]:
        st.success(i)

    st.markdown("### ❌ What needs work")
    for i in data["what_needs_work"]:
        st.error(i)

    st.markdown("### 💡 Suggestions")
    for i in data["suggestions"]:
        st.warning(i)

# ================= RUN =================

if uploaded_file and st.button("Analyze outfit"):

    with st.spinner("Analyzing image..."):

        total_start = time.time()
        image_b64 = base64.b64encode(uploaded_file.read()).decode()

        # Vision
        t1 = time.time()
        vision = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[
                {"role":"system","content":VISION_PROMPT},
                {"role":"user","content":[
                    {"type":"text","text":"Describe the visible outfit."},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
        )
        vision_time = time.time() - t1

        vtext = vision.choices[0].message.content or ""
        vjson = extract_json(vtext)

        # Structuring
        t2 = time.time()
        tinput = json.dumps(vjson) if vjson else vtext
        text = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0,
            max_tokens=600,
            messages=[
                {"role":"system","content":TEXT_PROMPT},
                {"role":"user","content":tinput}
            ],
        )
        text_time = time.time() - t2

        total_time = time.time() - total_start

        ftext = text.choices[0].message.content or ""
        fjson = extract_json(ftext)

        if not fjson:
            st.error("Failed to generate valid JSON")
            st.code(ftext)
            st.stop()

        result = normalize(fjson)

        # UI
        render_overlay(uploaded_file, result)
        render_analysis(result)

        st.divider()
        st.subheader("Raw JSON output")
        st.code(json.dumps(result, indent=2), language="json")

        st.divider()
        st.subheader("System notes")
        st.markdown(f"""
Deterministic: temperature=0  
Vision time: {vision_time:.2f}s  
Structuring time: {text_time:.2f}s  
Total time: {total_time:.2f}s (target ≤ 3s p95)  

Model: allenai/molmo-2-8b  
Calls per request: 2  
Approx cost: free tier  
""")
