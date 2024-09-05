

import spacy 
from transformers import pipeline

def summarizer(rowdoc):
    # Load the summarization pipeline with a specified model
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    # Load the Spacy English model
    nlp = spacy.load('en_core_web_sm')
    
    # Process the text using Spacy
    text = nlp(rowdoc)
    
    # Convert Spacy Doc object back to string
    text_str = text.text
    
    # Summarize the text
    summary_result = summarizer(text_str, max_length=300, min_length=100, do_sample=False)
    
    # Extract summary text from the result
    summary = summary_result[0]['summary_text']
    
    # Extract tokens from the Spacy Doc object
    tokens = [token.text for token in text]
    original_text = ' '.join(tokens)
    return summary, original_text
