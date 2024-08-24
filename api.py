from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx 
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

# Download the 'punkt' and 'stopwords' resources
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

# Configure CORS settings
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def fetch_webpage_content(url):
    """Fetches the webpage content asynchronously."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text content from all <p> tags
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/summarize/')
async def summarize_website(url: str = Form(...)):
    try:
        # Fetch the webpage content asynchronously
        webpage_content = await fetch_webpage_content(url)

        # Tokenize the text into sentences and words
        sentences = sent_tokenize(webpage_content)
        words = word_tokenize(webpage_content)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Calculate word frequencies
        word_freq = FreqDist(filtered_words)

        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]

        # Get the top N sentences with the highest scores as the summary
        summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)

        # Combine the summary sentences into the final summary
        summary = ' '.join(summary_sentences)

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
