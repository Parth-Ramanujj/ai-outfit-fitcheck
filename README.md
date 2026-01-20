# 🧥 AI Outfit Fitcheck

A Streamlit-powered outfit analysis app using OpenRouter's vision-language models.

## 🚀 Quick Start

### Local Development

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Sujaltalreja04/Fashion-Project.git
   cd Fashion-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
   - Add your OpenRouter API key:
     ```toml
     OPENROUTER_API_KEY = "your-api-key-here"
     ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 🔒 Security

- **API keys are never committed** - see `.gitignore`
- Uses `st.secrets` for local development
- Environment variables for production deployments

## 📦 Deployment

### Streamlit Cloud
1. Push to GitHub (secrets excluded via `.gitignore`)
2. Connect repo to Streamlit Cloud
3. Add `OPENROUTER_API_KEY` in Secrets settings
4. Deploy!

### Docker / Other Hosting
Set environment variable before running:
```bash
export OPENROUTER_API_KEY="your-key-here"
streamlit run app.py
```

## 📋 Features

- 🖼️ Upload outfit images
- 👁️ Two-stage vision analysis (Vision → Text refinement)
- 🎯 Structured JSON feedback
- 📊 Professional outfit fitcheck recommendations

## 🛠️ Tech Stack

- **Streamlit** - UI framework
- **OpenRouter API** - LLM & vision models
- **Python 3.8+**

## 📝 License

MIT
