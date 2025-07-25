import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st

@st.cache_resource
def load_assets():
    """Loads the pre-trained model and tokenizer."""
    model = load_model('hamlet_BiLSTM.h5', compile=False)

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

def predict_next_words(model, tokenizer, text, num_words=1):
    max_sequence_len = model.input_shape[1]

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences(
            [token_list], maxlen=max_sequence_len, padding='pre'
        )
        predicted_probs = model.predict(sequence, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        output_word = tokenizer.index_word.get(predicted_index, "")
        text += " " + output_word
    return text

st.title("Hamlet Next Word Predictor")
st.write("Enter a seed text and the model will predict the next word(s).")

seed_text = st.text_input("Enter your text:", "So nightly toyles the subiect of the")
num_to_predict = st.slider("Number of words to predict:", 1, 10, 1)

if st.button("Predict"):
    if seed_text:
        result = predict_next_words(model, tokenizer, seed_text, num_to_predict)
        st.success(f"**Result:** {result}")
    else:
        st.warning("Please enter some text.")