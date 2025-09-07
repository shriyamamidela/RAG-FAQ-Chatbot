# FAQ Chatbot (Myntra) 🤖

A simple Streamlit chatbot that answers Myntra FAQs using your own `faqs.txt`. Optional Gemini (Google Generative AI) provides cleaner, grounded answers.

## Features

- Keyword search over your FAQs
- Optional Gemini answers grounded in retrieved FAQs
- Sidebar: Clear chat, Reload FAQs, Rebuild index
- Strict answers toggle (only answer from FAQs)
- Follow-up suggestion chips and quick topics
- Light UI polish

## Setup

1) Install
```bash
pip install -r requirements.txt
```

2) Add API key (optional)
```bash
cp env_template.txt .env
# edit .env and set GEMINI_API_KEY=...
```

3) Run
```bash
streamlit run app.py
```

Open http://localhost:8501 and start asking questions.

## Update FAQs
- Edit `faqs.txt`
- In the app sidebar, click Reload FAQs (and Rebuild index if using advanced search)

## Notes
- Without an API key, answers come from the closest FAQ.
- With an API key, Gemini generates concise answers grounded in your FAQs.

## License
MIT
