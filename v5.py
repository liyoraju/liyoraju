import streamlit as st
import requests
import re
import os

# ======================
# CONFIGURATION
# ======================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLAMA_MODEL = "meta-llama/llama-3-8b-instruct"

# ======================
# API COMMUNICATION
# ======================
def llama_completion_via_openrouter(prompt: str) -> str:
    """Send prompt to LLaMA via OpenRouter API and return response."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ OpenRouter LLaMA error: {e}"

def search_openalex(keywords: str, per_page: int = 5) -> list:
    """Search OpenAlex API using extracted keywords."""
    url = "https://api.openalex.org/works"
    params = {"search": keywords, "per-page": per_page}
    response = requests.get(url, params=params)
    return response.json().get("results", []) if response.status_code == 200 else []

# ======================
# DATA PROCESSING
# ======================
def extract_keywords(prompt: str) -> str:
    """Extract clean search keywords using LLaMA."""
    instruction = "Extract 2–5 academic search keywords from this research question. Respond ONLY with a comma-separated list:"
    raw_output = llama_completion_via_openrouter(f"{instruction}\n\n{prompt}")
    keywords_line = raw_output.split("\n")[0]
    return re.sub(r"[^a-zA-Z0-9,\- ]+", "", keywords_line).strip()

def convert_abstract(abstract_index: dict, word_limit: int = 60) -> str:
    """Convert inverted index abstract to readable text."""
    if not abstract_index:
        return ""
    words = sorted(abstract_index.items(), key=lambda x: min(x[1]))
    flat = " ".join([word for word, _ in words])
    return " ".join(flat.split()[:word_limit]) + "..."

def get_work_type(type_str: str) -> str:
    """Convert OpenAlex type URI to readable format."""
    if not type_str:
        return "Unknown"
    return (
        type_str.replace("https://openalex.org/", "")
               .replace("-", " ")
               .capitalize()
    )

def summarize_with_llm(user_query: str, docs: list) -> str:
    """Generate structured summary using LLaMA."""
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
3. For each section, provide concrete evidence from the abstracts, including author names and publication years in parentheses (e.g., Alshehab, 2024).
4. End with a short conclusion about the significance or implication of the findings.
5. Keep your tone academic, concise, and easy to read for a research audience.

User Research Question:
"{user_query}"

Relevant Papers:
{context}

Answer:"""
    return llama_completion_via_openrouter(prompt)

# ======================
# UI COMPONENTS
# ======================
def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(page_title="OpenAlex Library Chatbot By Wexor ai", layout="centered")

def load_custom_css():
    """Inject custom CSS styles."""
    st.markdown("""
        <style>
            /* Hide Streamlit footer (Made with Streamlit) */
            footer, .st-emotion-cache-zj8gb8 {
                visibility: hidden;
                height: 0px;
            }

            /* Hide top right user avatar */
            header [data-testid="avatar"] {
                display: none !important;
            }

            /* Hide GitHub corner icon */
            [data-testid="stDecoration"] {
                display: none !important;
            }

            /* Hide bottom avatars / status widgets */
            [data-testid="stStatusWidget"] {
                display: none !important;
            }

            /* Optional: Hide hamburger sidebar nav */
            [data-testid="stSidebarNav"] {
                display: none !important;
            }

            .header-container {
                background-color: #5b3cc4;
                padding: 2rem;
                border-radius: 20px 20px 0 0;
                color: white;
                text-align: center;
            }

            .header-title {
                font-size: 1.8rem;
                font-weight: bold;
            }

            .header-subtitle {
                font-size: 1rem;
                opacity: 0.85;
            }

            .chat-box {
                background-color: rgb(255, 255, 255);
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
                color: black;
            }

            .chip {
                background-color: #5b3cc4;
                padding: 0.4rem 0.8rem;
                border-radius: 999px;
                margin: 0.2rem;
                display: inline-block;
                font-size: 0.85rem;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)





def render_header():
    """Display the application header."""
    st.markdown("""
        <div class="header-container">
            <div class="header-title">🤖 Library Assistant By Wexor AI</div>
            <div class="header-subtitle">Ask me about books, research materials, and library services</div>
        </div>
    """, unsafe_allow_html=True)

def render_intro():
    """Display introductory message with suggested tags."""
    st.markdown("""
        <div class="chat-box">
            Hello! I'm your Library Assistant. I can help you find books, research materials, and answer questions about our library services. What would you like to know today?
            <div style="margin-top: 0.5rem;">
                <span class="chip">AI books</span>
                <span class="chip">Blockchain</span>
                <span class="chip">Library hours</span>
                <span class="chip">E-books</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_result_item(item: dict):
    """Display a single search result item with full-text link, PDF download, and OpenAlex reference."""
    title = item.get("display_name", "No Title")
    authors = [auth["author"]["display_name"] for auth in item.get("authorships", [])]
    abstract_text = convert_abstract(item.get("abstract_inverted_index"))
    published_date = item.get("publication_date", "Unknown")
    work_type = get_work_type(item.get("type"))

    landing_page = item.get("primary_location", {}).get("landing_page_url")
    openalex_url = item.get("id", "#")
    fulltext_url = landing_page if landing_page else openalex_url

    # Get PDF (if any)
    pdf_url = item.get("primary_location", {}).get("pdf_url")

    # OpenAlex reference URL
    openalex_url = item.get("id", "#")

    # Render result
    st.markdown(f"### [{title}]({fulltext_url})")
    st.markdown(f"**Type**: `{work_type}` | **Published**: {published_date}")
    st.markdown(f"**Authors**: {', '.join(authors) if authors else 'Unknown'}")
    st.markdown(abstract_text)

    # Action links
    links = []
    if pdf_url:
        links.append(f"[📥 PDF Download]({pdf_url})")
    if openalex_url:
        links.append(f"[🔗 OpenAlex Link]({openalex_url})")
    if links:
        st.markdown(" | ".join(links), unsafe_allow_html=True)

    st.markdown("---")



# ======================
# MAIN APP FLOW
# ======================
def main():
    setup_page_config()
    load_custom_css()
    render_header()
    render_intro()

    query = st.text_input("🔍 Enter your research query:")

    if query:
        try:
            with st.spinner("🔍 Extracting keywords..."):
                keywords = extract_keywords(query)

            with st.spinner("🔎 Searching OpenAlex..."):
                results = search_openalex(keywords)

            if results:
                with st.spinner("🤖 Generating summary using LLaMA 3.3..."):
                    summary = summarize_with_llm(query, results)

                st.subheader("🧠 Summary")
                st.markdown(summary)
                st.markdown("---")

                st.subheader("📄 Related Works from OpenAlex")
                for item in results:
                    render_result_item(item)
            else:
                st.warning("No results found for the extracted keywords.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
