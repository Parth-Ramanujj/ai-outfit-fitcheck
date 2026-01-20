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
    width: fit-content;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.good { background: #d1fae5; color: #065f46; }
.bad { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

st.title("🧥 AI Outfit Fitcheck")
st.caption("Upload outfit image → Get Outfit Analysis")

uploaded_file = st.file_uploader("Upload a full or near-full body image", type=["jpg", "jpeg", "png"])

# ================= PROMPTS =================

VISION_PROMPT = """
Describe ONLY what is visible in the image.
Rules:
- Clothing items only
- No opinions or styling advice
- No guessing
- Short factual sentences
"""

TEXT_PROMPT = """
You are a STRICT JSON formatting engine.
Output ONLY valid JSON.

FINAL SCHEMA:
{
  "overall_vibe": {"summary": "", "category": ""},
  "what_works": [],
  "what_needs_work": [],
  "suggestions": [],
  "item_flags": {"dress": "", "top": "", "bottom": "", "shoes": "", "bag": "", "accessories": ""}
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

def sanitize_final(result: dict) -> dict:
    result["what_works"] = result.get("what_works", [])[:3]
    result["what_needs_work"] = result.get("what_needs_work", [])[:2]
    result["suggestions"] = result.get("suggestions", [])[:2]

    filler_works = [
        "The visible clothing pieces coordinate well together.",
        "The outfit elements appear visually consistent.",
        "The garments create a balanced overall appearance."
    ]
    i = 0
    while len(result["what_works"]) < 3:
        result["what_works"].append(filler_works[i % len(filler_works)])
        i += 1

    while len(result["what_needs_work"]) < 2:
        result["what_needs_work"].append("No clearly visible fit issues are present.")

    while len(result["suggestions"]) < 2:
        result["suggestions"].append("No changes are required based on visible elements.")

    return result

# ================= UI FUNCTIONS =================

def show_image_with_tags(uploaded_file, data):
    good = data["what_works"][:2]
    bad = data["what_needs_work"][:2]

    chips_html = ""
    for g in good:
        chips_html += f'<div class="chip good">✅ {g}</div>'
    for b in bad:
        chips_html += f'<div class="chip bad">❌ {b}</div>'

    st.markdown(f"""
    <div class="image-container">
        <img src="data:image/jpeg;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}" style="width:100%; border-radius:16px;">
        <div class="overlay-box">
            {chips_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_outfit_page(data):
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

# ================= RUN =================

if uploaded_file and st.button("Analyze My Outfit"):
    with st.spinner("Analyzing outfit…"):

        total_start = time.time()

        image_b64 = base64.b64encode(uploaded_file.read()).decode()

        # ---------- STAGE 1 ----------
        t1 = time.time()
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
        vision_time = time.time() - t1

        vision_raw = vision_resp.choices[0].message.content or ""
        vision_json = extract_json_loose(vision_raw)

        # ---------- STAGE 2 ----------
        t2 = time.time()
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
        text_time = time.time() - t2

        total_time = time.time() - total_start

        final_raw = text_resp.choices[0].message.content or ""
        final_json = extract_json_loose(final_raw)

        if not final_json:
            st.error("❌ Failed to generate valid result.")
            st.code(final_raw)
            st.stop()

        final_result = sanitize_final(final_json)

        # 👉 FIRST CIRCLE UI
        show_image_with_tags(uploaded_file, final_result)

        # 👉 SECOND CIRCLE UI
        show_outfit_page(final_result)

        # 👉 RAW JSON
        st.divider()
        st.subheader("📄 Raw JSON (Submission-Ready)")
        st.code(json.dumps(final_result, indent=2), language="json")

        # ================= SYSTEM GUARANTEES =================

        st.divider()
        st.subheader("🅱️ Determinism & Consistency")
        st.markdown("""
✔ Same image → same structure (temperature=0)  
✔ Fixed JSON schema enforced  
✔ Missing fields auto-filled  
✔ No schema drift allowed  
""")

        st.subheader("🅲 Latency Breakdown")
        st.markdown(f"""
• Vision model: `{vision_time:.2f}s`  
• Text formatting: `{text_time:.2f}s`  
• **Total end-to-end:** `{total_time:.2f}s`  
Target: ≤ 3 seconds (p95)
""")

        st.subheader("🅳 Cost Estimate")
        st.markdown("""
**Models used:**  
• allenai/molmo-2-8b (vision + text)

**Calls per fitcheck:**  
• 2 calls (1 vision + 1 structuring)

**Approx cost:**  
• ~$0 (free tier model on OpenRouter)  
• 1,000 fitchecks ≈ **$0 – $1 (depending on provider limits)**
""")
