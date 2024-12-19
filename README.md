
# POS Tagging and Sentiment Analysis

This project integrates **Part-of-Speech (POS) Tagging** and **Sentiment Analysis** into a single pipeline using Hugging Face Transformer models. It identifies grammatical roles of words in a sentence and classifies the sentence sentiment as **Positive** or **Negative**. The project leverages **CoNLL** and **IMDB datasets** for training and testing, providing accurate results for real-world text inputs.

---

## Overview

### Features:
1. **POS Tagging**:
   - Tags each word in the sentence with its grammatical role.
   - Example tags include:
     - **DT**: Determiner
     - **NN**: Noun
     - **VBD**: Verb (Past tense)
     - **JJ**: Adjective
   - Trained on the **CoNLL dataset**, a widely used resource for POS tagging tasks.

2. **Sentiment Analysis**:
   - Analyzes the overall sentiment of a sentence.
   - Possible outputs:
     - **Positive**
     - **Negative**
   - Trained on the **IMDB dataset**, which contains labeled movie reviews for sentiment classification.

3. **Interactive User Interface**:
   - Powered by **Gradio** for easy-to-use text input and output visualization.
   - Results include:
     - POS Tags displayed word-by-word.
     - Overall sentiment classification.

---

## Datasets Used

1. **CoNLL Dataset**:
   - A dataset primarily used for Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
   - Contains labeled words with their grammatical roles and named entity tags.

2. **IMDB Dataset**:
   - A popular dataset for binary sentiment classification.
   - Consists of 50,000 labeled movie reviews for **Positive** and **Negative** sentiment analysis.

---

## How It Works

The application processes user-input sentences and provides:
- **POS Tags**: Word-level grammatical roles.
- **Sentiment Analysis**: Sentence-level sentiment (Positive/Negative).

### Example Input:
```
This was the best movie I have ever seen.
```

### Example Output:
```
**POS Tags:**
This -> DT  
was -> VBD  
the -> DT  
best -> JJS  
movie -> NN  
I -> PRP  
have -> VBP  
ever -> RB  
seen -> VBN  
. -> .

**Sentiment Analysis:**
Sentiment: Positive
```

---

## Live Demo

Try the project live on Hugging Face Spaces:

[**Live Demo Link**](https://huggingface.co/spaces/sncmedikonduru/Sentiment_POSAnalysis)

---

## Setup Instructions

Follow these steps to run the project locally:

### 1. Clone the Repository
```bash
git clone https://huggingface.co/spaces/sncmedikonduru/Sentiment_POSAnalysis
cd Sentiment_POSAnalysis
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Gradio application:
```bash
python app.py
```

### 4. Access the Application
Open the browser and navigate to:
```
http://localhost:8501
```

---

## Acknowledgments

This project is made possible thanks to:
- [Hugging Face](https://huggingface.co) for the Transformers library and pre-trained models.
- [Gradio](https://gradio.app) for building interactive UIs.

