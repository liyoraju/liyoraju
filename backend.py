import requests
import re
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set this in your environment or when you import backend.py
LLAMA_MODEL = "meta-llama/llama-3-8b-instruct"


def set_api_key(key: str):
    global OPENROUTER_API_KEY
    OPENROUTER_API_KEY = key


def llama_completion_via_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "❌ Missing OpenRouter API Key"
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


def extract_keywords(prompt: str) -> str:
    instruction = "Extract 2–5 academic search keywords from this research question. Respond ONLY with a comma-separated list:"
    raw_output = llama_completion_via_openrouter(f"{instruction}\n\n{prompt}")
    keywords_line = raw_output.split("\n")[0]
    return re.sub(r"[^a-zA-Z0-9,\- ]+", "", keywords_line).strip()


def search_openalex(keywords: str, per_page=5) -> list:
    url = "https://api.openalex.org/works"
    params = {
        "search": keywords,
        "per_page": per_page,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        raise Exception(f"OpenAlex API error: {response.status_code}")


def convert_abstract(abstract_inverted_index: dict, word_limit=60) -> str:
    if not abstract_inverted_index:
        return ""
    pos_word_pairs = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            pos_word_pairs.append((pos, word))
    pos_word_pairs.sort()
    words = [word for _, word in pos_word_pairs]
    truncated = " ".join(words[:word_limit])
    return truncated + "..." if len(words) > word_limit else truncated


def get_work_type(type_str: str) -> str:
    if type_str:
        return type_str.replace("https://openalex.org/", "").replace("-", " ").capitalize()
    return "Unknown"


def summarize_with_llm(user_query: str, docs: list) -> str:
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