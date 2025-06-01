from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from backend import (  # Replace with your actual filename
    set_api_key, 
    extract_keywords, 
    search_openalex, 
    convert_abstract, 
    get_work_type, 
    summarize_with_llm
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Set your API key on startup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY:
    set_api_key(OPENROUTER_API_KEY)

@app.route('/api/research', methods=['POST'])
def research_query():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if len(query) < 5:
            return jsonify({'error': 'Query too short. Please provide more details.'}), 400
        
        # Step 1: Extract keywords
        keywords = extract_keywords(query)
        
        # Step 2: Search OpenAlex
        papers = search_openalex(keywords, per_page=10)
        
        if not papers:
            return jsonify({
                'query': query,
                'keywords': keywords,
                'papers': [],
                'summary': 'No relevant papers found for your query. Try different keywords.',
                'total_results': 0
            })
        
        # Step 3: Process papers for frontend
        processed_papers = []
        for paper in papers:
            processed_paper = {
                'title': paper.get('display_name', 'No Title'),
                'authors': [auth['author']['display_name'] for auth in paper.get('authorships', [])],
                'publication_year': paper.get('publication_year'),
                'type': get_work_type(paper.get('type')),
                'abstract': convert_abstract(paper.get('abstract_inverted_index'), word_limit=100),
                'doi': paper.get('doi'),
                'url': paper.get('primary_location', {}).get('landing_page_url'),
                'citation_count': paper.get('cited_by_count', 0)
            }
            processed_papers.append(processed_paper)
        
        # Step 4: Generate AI summary
        summary = summarize_with_llm(query, papers[:5])  # Use top 5 papers for summary
        
        return jsonify({
            'query': query,
            'keywords': keywords,
            'papers': processed_papers,
            'summary': summary,
            'total_results': len(processed_papers)
        })
        
    except Exception as e:
        return jsonify({'error': f'Research failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'api_key_configured': bool(OPENROUTER_API_KEY)
    })

@app.route('/api/keywords', methods=['POST'])
def extract_keywords_only():
    """Endpoint to just extract keywords from a query"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        keywords = extract_keywords(query)
        return jsonify({'keywords': keywords})
        
    except Exception as e:
        return jsonify({'error': f'Keyword extraction failed: {str(e)}'}), 500

if __name__ == '__main__':
    if not OPENROUTER_API_KEY:
        print("⚠️  Warning: OPENROUTER_API_KEY not set!")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)