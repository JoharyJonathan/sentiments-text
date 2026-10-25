# Sentiment Analysis with LSTM

A text sentiment classification project built with Python and TensorFlow. This project uses a Recurrent Neural Network (RNN) with a Long Short-Term Memory (LSTM) layer to classify movie reviews as **positive** or **negative**.

## Overview

The goal of this project is to explore Natural Language Processing (NLP) and deep learning by building a model that can identify the sentiment expressed in a text.

The model is trained using the IMDB movie reviews dataset and can be used to analyze new text inputs through a command-line interface.

The application loads a trained model, processes the user's input, and predicts its sentiment.

## Features

- **Binary sentiment classification:** classify text as positive or negative.
- **LSTM neural network:** use recurrent deep learning to process sequences of words.
- **Text preprocessing:** clean HTML tags, tokenize text, and pad sequences to a fixed length.
- **Interactive CLI:** enter text directly into the terminal and receive a prediction.
- **Prediction probabilities:** display the model's output probabilities for both sentiment classes.
- **Pre-trained model:** load the saved model without having to train it again.

## Technologies

- **Python** — application logic and model integration.
- **TensorFlow / Keras** — neural network implementation and inference.
- **NumPy** — numerical operations and prediction processing.
- **Pandas** — dataset loading and processing.
- **LSTM (RNN)** — sequence-based text classification.
- **Docker** — containerization support.

## How It Works

The prediction pipeline follows these steps:

1. **Text input:** the user enters a movie review or another text.
2. **Text cleaning:** HTML tags are removed.
3. **Tokenization:** words are converted into numerical sequences using a tokenizer.
4. **Sequence padding:** the sequence is padded or truncated to a fixed length.
5. **Model inference:** the processed sequence is passed to the trained LSTM model.
6. **Sentiment prediction:** the model selects the most likely class.
7. **Output:** the predicted sentiment and the output probabilities are displayed in the terminal.

```text
User Input
    |
    v
Text Preprocessing
    |
    v
Tokenization
    |
    v
Sequence Padding
    |
    v
Trained LSTM Model
    |
    v
Sentiment Classification
    |
    +------> Positive
    |
    +------> Negative
```

## Project Structure

```text
sentiments-text/
├── IMDB reviews/
│   └── IMDB Dataset.csv
├── train.py
├── test.py
├── sentiment_imdb_lstm_model.h5
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

- `train.py` — trains the sentiment classification model.
- `test.py` — loads the trained model and runs interactive predictions.
- `sentiment_imdb_lstm_model.h5` — saved TensorFlow/Keras model.
- `IMDB reviews/` — contains the dataset used for training and tokenizer initialization.
- `requirements.txt` — lists the Python dependencies.
- `Dockerfile` — defines the Docker image configuration.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python
- pip
- Git

Docker is optional if you prefer running the project in a container.

### 1. Clone the repository

```bash
git clone https://github.com/JoharyJonathan/sentiments-text.git
cd sentiments-text
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python test.py
```

The application prompts you to enter a text for sentiment analysis.

Enter a movie review or another text, then press Enter to obtain the prediction.

To exit the application, type:

```text
quitter
```

or:

```text
exit
```

## Example

Example of the expected interaction:

```text
--- Launch Sentiment Analysis ---

Enter a movie review:
This movie was fantastic! The acting was great and the story was beautiful.

Predicted sentiment: POSITIVE
```

The prediction above is an illustrative example. Actual results depend on the trained model and the input text.

## Dataset

The project uses the IMDB movie reviews dataset, which contains movie reviews labeled with sentiment categories.

The model is designed for binary classification:

- `0` — Negative
- `1` — Positive

The tokenizer configuration used for inference must remain consistent with the configuration used during training.

## Limitations

- Predictions are not guaranteed to be correct.
- The model may struggle with sarcasm, irony, ambiguous expressions, or mixed sentiments.
- Performance can vary depending on the input text and how closely it resembles the training data.
- The project focuses on binary sentiment classification rather than emotion detection or neutral sentiment classification.

## Future Improvements

Potential improvements include:

- Evaluate the model using accuracy, precision, recall, and F1-score.
- Add a confusion matrix to analyze classification errors.
- Save and load the tokenizer separately instead of reconstructing it from the dataset.
- Improve preprocessing and test different sequence lengths and model architectures.
- Develop a web interface or REST API to make the model accessible beyond the command line.

## Author

**Johary Jonathan**

* GitHub: [JoharyJonathan](https://github.com/JoharyJonathan)
* LinkedIn : https://www.linkedin.com/in/johary-rabarison-130bb0327/

## License

Please refer to the repository's license information before redistributing or reusing this project.