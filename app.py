import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

# Load POS tagging model and tokenizer
pos_model_name = "bert-base-cased"  # Replace with your POS tagging model name on Hugging Face Hub
pos_model = AutoModelForTokenClassification.from_pretrained(pos_model_name)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_name)

# Load Sentiment analysis model and tokenizer
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment analysis model
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

# Define POS tag names (update based on your fine-tuned model if needed)
LABEL_NAMES = [
    "O", "B-NP", "I-NP", "B-VP", "I-VP", "B-PP", "I-PP", ".", "CC", "CD", "DT", "EX",
    "FW", "IN", "JJ", "JJR", "JJS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS",
    "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "VB", "VBD", "VBG", "VBN",
    "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"
]

# POS and Sentiment pipeline function
def analyze_sentence(sentence):
    # POS Tagging
    inputs = pos_tokenizer(sentence, return_tensors="pt", is_split_into_words=True)
    word_ids = inputs.word_ids()
    tokens = pos_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = pos_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Align predictions
    pos_results = []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id != prev_word_id:
            pos_results.append((tokens[idx], LABEL_NAMES[predictions[idx]]))
        prev_word_id = word_id

    # Sentiment Analysis
    sentiment_inputs = sentiment_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        sentiment_outputs = sentiment_model(**sentiment_inputs)
    sentiment_prediction = torch.argmax(sentiment_outputs.logits, dim=-1).item()
    sentiment_label = "Positive" if sentiment_prediction == 1 else "Negative"

    return pos_results, sentiment_label

# Streamlit App Interface
st.title("POS Tagging and Sentiment Analysis App 🚀")
st.write("Enter a sentence below to analyze its POS tags and sentiment.")

# Input text
sentence = st.text_input("Enter your sentence here:", "The movie was fantastic and the actors did a great job.")

if st.button("Analyze"):
    if sentence.strip():
        pos_tags, sentiment = analyze_sentence(sentence)
        
        # Display POS tagging results
        st.subheader("POS Tagging Results:")
        for word, tag in pos_tags:
            st.write(f"**{word}**: {tag}")
        
        # Display Sentiment Analysis result
        st.subheader("Sentiment Analysis Result:")
        st.write(f"**Sentiment**: {sentiment}")
    else:
        st.error("Please enter a valid sentence!")
