from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import html2text
import logging

app = Flask(__name__)
CORS(app)

# Enhanced logging
logging.basicConfig(level=logging.DEBUG)

# Add configuration for proxy server
app.config['PREFERRED_URL_SCHEME'] = 'https'

@app.route('/urlcatch')  # No trailing slash
@app.route('/urlcatch/')  # With trailing slash
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/urlcatch/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Log the incoming request
        app.logger.debug(f"Received request at {request.path}")
        app.logger.debug(f"Request headers: {request.headers}")
        
        if not request.is_json:
            app.logger.error("Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            app.logger.error("No JSON data in request")
            return jsonify({'error': 'No data provided'}), 400

        urls = [data.get(f'url{i}') for i in range(1, 4) if data.get(f'url{i}')]
        if not urls:
            app.logger.error("No URLs provided")
            return jsonify({'error': 'No URLs provided'}), 400

        analysis_type = data.get('analysis_type', 'basic')
        custom_prompt = data.get('custom_prompt')
        
        contents = [fetch_url_content(url) for url in urls if url]
        if not contents:
            app.logger.error("Failed to fetch content from URLs")
            return jsonify({'error': 'Failed to fetch content from URLs'}), 400
        
        result = process_content(contents, analysis_type, custom_prompt)
        return jsonify({'result': result})
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Configure Google Generative AI
GENAI_API_KEY = "AIzaSyDceI3mqdAoSIPkGpYgbttbuJ-YUwoLk3E"
genai.configure(api_key=GENAI_API_KEY)

# Set up the Generative AI model configuration
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1000,
}

# Define prompts dictionary
prompts = {
    'basic': """Analyze these contents and provide a clear summary:

{}

FORMAT YOUR RESPONSE LIKE THIS:
# Summary
• Key point 1
• Key point 2
• Key point 3

# Main Insights
• First insight
• Second insight
• Third insight""",

    'contextual': """Analyze these contents and provide a contextual summary:

{}

FORMAT YOUR RESPONSE LIKE THIS:
# Key Takeaways
• Point 1
• Point 2
• Point 3

# Main Themes
• Theme 1
• Theme 2
• Theme 3

# Notable Information
• Notable point 1
• Notable point 2
• Notable point 3""",

    'comparison': """Compare and analyze these contents:

{}

FORMAT YOUR RESPONSE LIKE THIS:
# Key Differences
• Difference 1
• Difference 2
• Difference 3

# Common Themes
• Theme 1
• Theme 2
• Theme 3

# Unique Points
• Point 1
• Point 2
• Point 3""",

    'actionable': """Analyze these contents and provide actionable insights:

{}

FORMAT YOUR RESPONSE LIKE THIS:
# Actionable Advice
1. First Action
   • Why: [Explanation]
   • How: [Steps]

2. Second Action
   • Why: [Explanation]
   • How: [Steps]

3. Third Action
   • Why: [Explanation]
   • How: [Steps]

# Implementation Steps
1. Immediate steps
2. Short-term steps
3. Long-term steps""",

    'social': """Create social media friendly summaries for these contents:

{}

FORMAT YOUR RESPONSE LIKE THIS:
# Social Media Summaries
📱 [First summary in 280 chars]
#hashtag1 #hashtag2

📱 [Second summary in 280 chars]
#hashtag1 #hashtag2

# Key Points for Sharing
• Point 1
• Point 2
• Point 3""",

    'analysis': """Provide a deep analysis of these contents:

{}

FORMAT YOUR RESPONSE LIKE THIS:
# Critical Analysis
## Main Arguments
• Argument 1
• Argument 2
• Argument 3

## Evidence Quality
• Evidence point 1
• Evidence point 2

## Potential Biases
• Bias 1
• Bias 2

# Strengths and Weaknesses
• ✓ Strength 1
• ✓ Strength 2
• × Weakness 1
• × Weakness 2"""
}

def clean_text(text):
    """Clean and format extracted text"""
    cleaned = ' '.join(text.split())
    cleaned = ' '.join([word for word in cleaned.split() if not (word.startswith('http') and len(word) > 30)])
    return cleaned

def fetch_url_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'aside']):
            element.decompose()
            
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        text = h.handle(str(soup))
        
        cleaned_text = clean_text(text)
        return cleaned_text[:8000]
        
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def process_content(contents, analysis_type, custom_prompt=None):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    if custom_prompt:
        content_text = '\n\n'.join([f'CONTENT {i+1}:\n{content}' for i, content in enumerate(contents)])
        prompt = f"""Analyze the following content based on this instruction:
{custom_prompt}

{content_text}

Format your response with clear headers and bullet points where appropriate."""
    else:
        content_sections = '\n\n'.join([f'CONTENT {i+1}:\n{content}' for i, content in enumerate(contents)])
        prompt = prompts.get(analysis_type, prompts['basic']).format(content_sections)
    
    response = model.generate_content(prompt)
    return response.text

if __name__ == '__main__':
    app.run(debug=True, port = 5010)