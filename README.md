# Simple-Rnn-IMDB-Sentiment
# Simple RNN - IMDB Movie Review Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](PASTE-YOUR-LIVE-APP-LINK-HERE)

**Live Demo:** [PASTE-YOUR-LIVE-APP-LINK-HERE](https://simple-rnn-imdb-sentiment-jstscbn6gpqc2ummv8qkta.streamlit.app/)

A sentiment classifier built with a **Simple RNN** in TensorFlow/Keras. It is trained on the IMDB movie review dataset and predicts whether a review is **positive** or **negative**. A **Streamlit** app lets you type in your own review and get a prediction instantly.

## Features

- Simple RNN model trained on the IMDB dataset (50,000 labelled reviews)
- Word embedding layer to turn words into dense vectors
- Interactive Streamlit web app for live predictions
- Notebooks that walk through embeddings, training, and prediction step by step

## Model Architecture

```
Input (padded word indices)
        |
Embedding layer
        |
SimpleRNN layer
        |
Dense layer (sigmoid)  ->  Positive / Negative
```

The trained model is saved as `simple_rnn_imdb.h5`.

## Project Structure

```
.
├── main.py                # Streamlit app
├── simple_rnn_imdb.h5     # Trained model
├── simplernn.ipynb        # Model building and training
├── emmending.ipynb        # Word embedding experiments
├── predication.ipynb      # Prediction on custom reviews
├── requirements.txt       # Dependencies
└── README.md
```

## Run Locally

```bash
git clone https://github.com/TheR0HIT/Simple-Rnn-IMDB-Sentiment.git
cd Simple-Rnn-IMDB-Sentiment

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
streamlit run main.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How It Works

1. Each review is converted to a sequence of word indices using the IMDB word index.
2. Sequences are padded to a fixed length.
3. The Embedding layer maps each word to a vector.
4. The Simple RNN reads the sequence and captures word order.
5. A sigmoid output gives a score between 0 and 1. A score above 0.5 is **Positive**, otherwise **Negative**.

## Example

| Review | Prediction |
|---|---|
| "This movie was fantastic, I loved every minute." | Positive |
| "Boring story and terrible acting." | Negative |

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit (deployed on Streamlit Community Cloud)

## Future Improvements

- Replace Simple RNN with LSTM / GRU for better long-range context
- Add Bidirectional layers

## Author

**Rohit Raj**
GitHub: [@TheR0HIT](https://github.com/TheR0HIT)
