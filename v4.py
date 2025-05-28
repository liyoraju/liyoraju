import streamlit as st
import requests
import re
import os

# --- CONFIG ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLAMA_MODEL = "meta-llama/llama-3-8b-instruct"

st.set_page_config(page_title="OpenAlex Library Chatbot", layout="centered")
st.title("OpenAlex Library Chatbot (Powered by LLaMA 3.3)")
st.markdown("Ask a research question, and I’ll summarize relevant academic papers using OpenAlex and LLaMA.")

query = st.text_input("🔍 Enter your research query:")

# --- CALL LLaMA 3.3 via OpenRouter ---
def llama_completion_via_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ OpenRouter LLaMA error: {e}"

# --- EXTRACT CLEAN KEYWORDS ---
def extract_keywords(prompt):
    instruction = "Extract 2–5 academic search keywords from this research question. Respond ONLY with a comma-separated list:"
    raw_output = llama_completion_via_openrouter(f"{instruction}\n\n{prompt}")
    keywords_line = raw_output.split("\n")[0]
    return re.sub(r"[^a-zA-Z0-9,\- ]+", "", keywords_line).strip()

# --- SEARCH OPENALEX ---
def search_openalex(keywords, per_page=5):
    url = "https://api.openalex.org/works"
    params = {
        "search": keywords,
        "per-page": per_page,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error(f"Error fetching from OpenAlex: {response.status_code}")
        return []

# --- PARSE ABSTRACT ---
def convert_abstract(abstract_index, word_limit=60):
    if not abstract_index:
        return ""
    words = sorted(abstract_index.items(), key=lambda x: min(x[1]))
    flat = " ".join([word for word, _ in words])
    truncated = " ".join(flat.split()[:word_limit])
    return truncated + "..."

def get_work_type(type_str):
    if type_str:
        return type_str.replace("https://openalex.org/", "").replace("-", " ").capitalize()
    return "Unknown"

# --- SUMMARIZE RESULTS WITH LLaMA ---
def summarize_with_llm(user_query, docs):
    context = ""
    for idx, doc in enumerate(docs, 1):
        title = doc.get("display_name", "No Title")
        authors = ", ".join([a["author"]["display_name"] for a in doc.get("authorships", [])])
        abstract = convert_abstract(doc.get("abstract_inverted_index"), word_limit=80)
        context += f"{idx}. Title: {title}\nAuthors: {authors}\nAbstract: {abstract}\n\n"

    prompt = f"""You are an expert academic assistant. Summarize the impact of the topic in a professional, structured way using the information provided in academic paper abstracts and metadata.

Instructions:
1. Start with a brief overview paragraph summarizing the answer.
2. Follow with 2–4 clearly titled sections that explain specific mechanisms or effects related to the topic.
3. For each section, provide concrete evidence from the abstracts, including author names and publicat  ion years in parentheses (e.g., Alshehab, 2024).
4. End with a short conclusion about the significance or implication of the findings.
5. Keep your tone academic, concise, and easy to read for a research audience.

User Research Question:
"{user_query}"

Relevant Papers:
{context}

Answer:"""

    return llama_completion_via_openrouter(prompt)

# --- MAIN FLOW ---
if query:
    with st.spinner("🔍 Extracting keywords..."):
        keywords = extract_keywords(query)
        #st.success(f"✅ Keywords used: `{keywords}`")

    with st.spinner("🔎 Searching OpenAlex..."):
        results = search_openalex(keywords)

    if results:
        with st.spinner("🤖 Generating summary using LLaMA 3.3..."):
            summary = summarize_with_llm(query, results)

        # --- Display LLM Answer First ---
        st.subheader("🧠 LLaMA Summary")
        st.markdown(summary)
        st.markdown("---")

        # --- Display Search Results ---
        st.subheader("📄 Related Works from OpenAlex")
        for item in results:
            title = item.get("display_name", "No Title")
            authors = [auth["author"]["display_name"] for auth in item.get("authorships", [])]
            abstract_text = convert_abstract(item.get("abstract_inverted_index"))
            openalex_url = item.get("id", "#")
            work_type = get_work_type(item.get("type"))
            published_date = item.get("publication_date", "Unknown")

            st.markdown(f"### [{title}]({openalex_url})")
            st.markdown(f"**Type**: `{work_type}` | **Published**: {published_date}")
            st.markdown(f"**Authors**: {', '.join(authors) if authors else 'Unknown'}")
            st.markdown(f"{abstract_text}")
            st.markdown("---")
    else:
        st.warning("No results found for the extracted keywords.")
