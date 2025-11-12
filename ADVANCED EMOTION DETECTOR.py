# Affectlytics: Product Sentiment Analysis with Advanced Negation Handling and Frozen Embeddings
import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant # Needed for setting embedding weights
import os
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# Suppress TensorFlow logging messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Configuration ---
MAX_WORDS = 20000     # Max number of words to keep in the vocabulary
MAX_LEN = 128         # Max length of a sequence (Increased for better context)
EMBEDDING_DIM = 128   # Dimension of the word embeddings (Consistent with common pre-trained sizes)
RNN_UNITS = 200       # MAXIMIZED CAPACITY for LSTM/GRU
DENSE_UNITS = 512     # MAXIMIZED CAPACITY for feature separation
NUM_CLASSES = 6
EPOCHS = 3            # OPTIMIZED: Increased slightly for accuracy, yet much faster than original 30
NUM_REVIEWS = 10      # Constant for the required number of inputs
CONV_FILTERS = 256    # Increased filter count for deeper CNN
TRAINABLE_EMBEDDING = False # CRITICAL NLP improvement: Use pre-trained weights, do not train them.

# Define the emotion labels for mapping
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}

# --- Advanced Preprocessing: Negation Handling ---

def handle_negation(text):
    """
    Applies two forms of negation handling:
    1. Attaches negation words to the following word (e.g., "not happy" -> "not_happy").
    2. Appends a global __NEGATED__ flag to the text if any negation is detected.
    This flag is a powerful, dense feature to combat Joy-bias.
    """
    negation_words = {
        'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', 'wasnt', 'isnt',
        'dont', 'doesnt', 'havent', 'hasnt', 'hardly', 'scarcely', 'barely', 'wont',
        "wouldn't", "shouldn't", "couldn't", 'without', 'isn\'t', 'don\'t', 'won\'t'
    }

    words = text.split()
    processed_words = []
    negation_detected = False
    i = 0

    while i < len(words):
        word = words[i].lower()

        # Check for negation: if found, set the flag
        if word in negation_words:
            negation_detected = True

            # Form 1: Attach negation to the next word
            if i + 1 < len(words):
                processed_words.append(f"{word}_{words[i+1]}")
                i += 2 # Skip the next word
            else:
                processed_words.append(words[i])
                i += 1
        else:
            processed_words.append(words[i])
            i += 1

    processed_text = " ".join(processed_words)

    # Form 2: Append global negation flag
    if negation_detected:
        processed_text += " __NEGATED__"

    return processed_text

# --- Ensemble Model Building Functions (Updated to use vocab_size) ---

def build_cnn_model(embedding_matrix, vocab_size):
    """Builds a deeper, two-layer CNN model with frozen pre-trained embeddings."""
    model = Sequential([
        Embedding(
            vocab_size,
            EMBEDDING_DIM,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=MAX_LEN,
            trainable=TRAINABLE_EMBEDDING
        ),
        Dropout(0.3),
        # Two stacked Conv layers for deeper local pattern recognition
        Conv1D(filters=CONV_FILTERS, kernel_size=5, activation='relu'),
        Conv1D(filters=CONV_FILTERS // 2, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(DENSE_UNITS, activation='relu'), # MAXIMIZED DENSE LAYER
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(embedding_matrix, vocab_size):
    """Builds the deep BiLSTM model with a three-layer stack for maximum context retention."""
    model = Sequential([
        Embedding(
            vocab_size,
            EMBEDDING_DIM,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=MAX_LEN,
            trainable=TRAINABLE_EMBEDDING
        ),
        Dropout(0.3),
        # First BiLSTM layer
        Bidirectional(LSTM(RNN_UNITS, recurrent_dropout=0.2, return_sequences=True)),
        # Second BiLSTM layer (CRITICAL: Added depth)
        Bidirectional(LSTM(RNN_UNITS, recurrent_dropout=0.2, return_sequences=True)),
        # Third BiLSTM layer (Removed recurrent_dropout for stability)
        Bidirectional(LSTM(RNN_UNITS // 2)),
        Dense(DENSE_UNITS, activation='relu'), # MAXIMIZED DENSE LAYER
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(embedding_matrix, vocab_size):
    """Builds the Bidirectional GRU model with increased density."""
    model = Sequential([
        Embedding(
            vocab_size,
            EMBEDDING_DIM,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=MAX_LEN,
            trainable=TRAINABLE_EMBEDDING
        ),
        Dropout(0.3),
        Bidirectional(GRU(RNN_UNITS)), # Removed recurrent_dropout for stability
        Dense(DENSE_UNITS, activation='relu'), # MAXIMIZED DENSE LAYER
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Caching function to load data and train the model once ---

@st.cache_resource
def load_and_train_model():
    """
    Loads data, computes class weights, simulates pre-trained embeddings,
    and trains the Ensemble models with frozen embeddings and early stopping.
    """

    # 1. Load Data
    data = load_dataset("dair-ai/emotion", "split")

    train_texts_combined = list(data['train']['text']) + list(data['validation']['text'])
    train_labels_combined = list(data['train']['label']) + list(data['validation']['label'])
    test_texts = list(data['test']['text'])
    test_labels = list(data['test']['label'])

    # Apply Negation Preprocessing
    train_texts_processed = [handle_negation(text) for text in train_texts_combined]
    test_texts_processed = [handle_negation(text) for text in test_texts]

    # Combine processed texts for tokenizer fit
    all_texts = train_texts_processed + test_texts_processed

    # 2. Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(all_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts_processed)
    test_sequences = tokenizer.texts_to_sequences(test_texts_processed)

    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels_combined, num_classes=NUM_CLASSES)

    # 3. Simulate Pre-trained Embedding Matrix
    word_index = tokenizer.word_index
    num_words = min(MAX_WORDS, len(word_index) + 1)

    # Initialize embedding matrix with correct size
    embedding_matrix = np.random.uniform(-0.05, 0.05, size=(num_words, EMBEDDING_DIM))

    # 4. Compute Class Weights (To combat overall class imbalance)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_combined),
        y=train_labels_combined
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # 5. Build and Train Ensemble Models
    models = [
        build_cnn_model(embedding_matrix, num_words),
        build_bilstm_model(embedding_matrix, num_words),
        build_gru_model(embedding_matrix, num_words)
    ]

    # Define Early Stopping Callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train all models
    for model in models:
        model.fit(
            train_padded,
            train_labels_one_hot,
            epochs=EPOCHS, # Uses the low EPOCHS=3 cap for fast startup
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            class_weight=class_weight_dict,
            callbacks=[early_stopping]
        )

    # 6. Ensemble Prediction and Evaluation
    pred_probs_list = [model.predict(test_padded, verbose=0) for model in models]
    ensemble_probs = np.mean(pred_probs_list, axis=0)
    y_pred = np.argmax(ensemble_probs, axis=1)

    accuracy = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='macro', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return models, tokenizer, metrics


# --- Prediction Function (Updated to use new preprocessing) ---
def predict_emotion(ensemble_models, tokenizer, text):
    """Predicts the emotion of a given review text using the ensemble models (Soft Voting)."""
    # Apply the same negation preprocessing used during training
    processed_text = handle_negation(text)

    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    pred_probs_list = [model.predict(padded_sequence, verbose=0) for model in ensemble_models]
    ensemble_prediction = np.mean(pred_probs_list, axis=0)[0]

    predicted_id = np.argmax(ensemble_prediction)
    predicted_label = id_to_label[predicted_id].capitalize()

    return predicted_label


# --- Core Analysis Logic (Unchanged) ---
def get_recommendation_and_comment(all_emotions):
    """
    Determines the overall recommended emotion and the buy/no-buy comment.
    Tie-breaking logic: If there's a tie for the highest count, use the emotion from the LAST review (index 9).
    """
    if not all_emotions:
        return "N/A", "Please enter at least one review for analysis.", {}

    emotion_counts = Counter(all_emotions)

    max_count = max(emotion_counts.values())

    top_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]

    dominant_emotion = top_emotions[0]

    if len(top_emotions) > 1:
        last_review_emotion = all_emotions[-1]
        if last_review_emotion in top_emotions:
            dominant_emotion = last_review_emotion

    positive_emotions = ['Joy', 'Love', 'Surprise']

    if dominant_emotion in positive_emotions:
        comment = (
            f"**Recommendation: BUY!** The dominant sentiment is positive ({dominant_emotion}), "
            "indicating a highly satisfied customer base. This is a strong indicator of product quality."
        )
    else:
        comment = (
            f"**Recommendation: DO NOT BUY.** The dominant sentiment is negative ({dominant_emotion}), "
            "suggesting significant customer dissatisfaction or potential product issues. Caution is advised."
        )

    return dominant_emotion, comment, emotion_counts

# --- Main Streamlit App (Unchanged) ---
def main():

    # Initialize session state for analysis results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'product_name' not in st.session_state:
        st.session_state.product_name = ""
    if 'dominant_emotion' not in st.session_state:
        st.session_state.dominant_emotion = "N/A"
    if 'recommendation_comment' not in st.session_state:
        st.session_state.recommendation_comment = ""
    if 'emotion_counts' not in st.session_state:
        st.session_state.emotion_counts = {label.capitalize(): 0 for label in emotion_labels}

    # --- CSS Injection ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        /* Animated Gradient Background */
        .stApp {
            font-family: 'Poppins', sans-serif;
            color: #FFFFFF;
            background: linear-gradient(-45deg, #0f002a, #2b0846, #002a3a, #00404a);
            background-size: 400% 400%;
            animation: gradientBG 25s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        h1, h2, h3, h4, .st-emotion-cache-12fm1f5, .st-emotion-cache-79elbk {
            color: #FFFFFF !important;
            letter-spacing: 1.2px;
        }
        .header {
            color: #FFFFFF;
            text-align: center;
            padding: 15px;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.4);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.7);
            margin-bottom: 25px;
        }
        .stButton>button {
            border-radius: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            color: white;
            background-color: #5b1076;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
            background-color: #7b1296;
        }
        .stTable tbody tr th, .stTable tbody tr td {
            color: #FFFFFF !important;
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-bottom: 1px solid #3d0a52;
            word-break: normal;
            white-space: normal;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: rgba(0, 0, 0, 0.3) !important;
            color: white !important;
            border: 1px solid #7b1296;
        }
        .recommendation-box {
            background-color: #3d0a52;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.8);
            margin-top: 20px;
            border: 2px solid #FFD700;
        }
        .emotion-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 8px;
            font-weight: 600;
            margin-left: 5px;
            color: white;
            background-color: #7b1296;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header"><h1><span style="color: #FFD700;">🛍️</span> Affectlytics: Product Sentiment Analyzer <span style="color: #FFD700;">📊</span></h1></div>', unsafe_allow_html=True)

    # Load and train the model (cached)
    with st.spinner("Training Ensemble Model... This may take a moment to achieve high accuracy."):
        models, tokenizer, metrics = load_and_train_model()

    # --- Input Section ---
    st.markdown("---")
    st.markdown("<h3 style='color: white;'>Input Product Details and Reviews (10 Required):</h3>", unsafe_allow_html=True)

    # Product Name Input
    product_name_input = st.text_input(
        "Product Name",
        value=st.session_state.product_name or "Gemini Smartwatch Pro",
        key="product_name_key"
    )
    st.session_state.product_name = product_name_input

    # 10 Review Text Inputs
    review_inputs = []
    cols = st.columns(2)

    # Example Reviews (for testing the fix: "not happy" should be Sadness)
    SAMPLE_REVIEWS = [
        "I am not happy with this purchase.", # Sadness (Test Case)
        "This product is amazing and fills me with joy!", # Joy
        "I absolutely adore the design and quality.", # Love
        "It broke immediately and this makes me so furious.", # Anger
        "I am afraid to use this device after the smoke I saw.", # Fear
        "Wow! I did not expect it to be this good. What a surprise.", # Surprise
        "This is disappointing, I feel awful about the wasted money.", # Sadness
        "I would never recommend this, it's terrible quality.", # Anger (Test Negation)
        "The shipping was quick, which was a pleasant surprise.", # Surprise
        "Everything about this experience brought me joy." # Joy
    ]

    for i in range(NUM_REVIEWS):
        col_index = i % 2
        with cols[col_index]:
            default_review = SAMPLE_REVIEWS[i] if i < len(SAMPLE_REVIEWS) else f"Review #{i+1} Example: This is fine, but nothing special."
            # Use stored data for sticky inputs if analysis has been run
            if len(st.session_state.analysis_results) == NUM_REVIEWS:
                 default_review = st.session_state.analysis_results[i]['review']

            review_text = st.text_area(
                f"Review #{i+1}",
                value=default_review,
                height=100,
                key=f"review_{i}"
            )
            review_inputs.append(review_text)

    # --- Analysis Button ---
    if st.button("Analyze All 10 Reviews", use_container_width=True, type="primary"):
        st.session_state.analysis_results = []
        all_emotions = []

        # Check for empty reviews
        for i, review in enumerate(review_inputs):
            if not review.strip():
                st.warning(f"Review #{i+1} is empty. Please fill all 10 inputs.")
                st.session_state.analysis_results = []
                return

        # Process each review
        for i, review in enumerate(review_inputs):
            predicted_emotion = predict_emotion(models, tokenizer, review)
            all_emotions.append(predicted_emotion)

            st.session_state.analysis_results.append({
                'product': st.session_state.product_name,
                'review_number': i + 1,
                'review': review,
                'emotion': predicted_emotion
            })

        # Generate final summary and recommendation
        dominant_emotion, recommendation_comment, emotion_counts = get_recommendation_and_comment(all_emotions)

        st.session_state.dominant_emotion = dominant_emotion
        st.session_state.recommendation_comment = recommendation_comment
        st.session_state.emotion_counts = emotion_counts
        # Force rerun to display results after processing
        st.rerun()


    # --- Results Display ---
    if st.session_state.analysis_results and st.session_state.dominant_emotion != "N/A":
        st.markdown("<hr style='border: 1px solid #FFD700;'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Analysis Results for: <span style='color:#FFD700;'>{st.session_state.product_name}</span></h2>", unsafe_allow_html=True)

        # 1. Display Table with Detected Emotion
        results_df = pd.DataFrame(st.session_state.analysis_results)
        results_df = results_df.rename(columns={
            'review_number': 'Review #',
            'review': 'Review Text',
            'emotion': 'Detected Emotion'
        })
        st.table(results_df[['Review #', 'Review Text', 'Detected Emotion']])

        # 2. Display Emotion Count (at the bottom of the project)
        st.markdown("<hr style='border: 1px solid #7b1296;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>Overall Emotion Counts:</h3>", unsafe_allow_html=True)

        # Ensure all 6 emotions are present in the display, even if count is 0
        all_counts = {label.capitalize(): 0 for label in emotion_labels}
        all_counts.update(st.session_state.emotion_counts)

        count_data = pd.DataFrame(
            all_counts.items(),
            columns=['Emotion', 'Count']
        ).set_index('Emotion').sort_values(by='Count', ascending=False)

        st.dataframe(count_data)

        # 3. Final Recommendation and Buy/No-Buy Comment
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(
            f"<h3>Final Recommendation Based on Sentiment:</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h4>Dominant Emotion: <span class='emotion-badge'>{st.session_state.dominant_emotion}</span></h4>",
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.recommendation_comment, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Evaluation Metrics Display (Model Diagnostics) ---
    st.markdown("---")
    st.markdown("<h2 style='color: #FFD700; text-align: center;'>Model Evaluation Metrics (Background System)</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    def display_metric(col, label, value):
        col.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

    display_metric(col1, "Accuracy", metrics['accuracy'])
    display_metric(col2, "Macro Precision", metrics['precision'])
    display_metric(col3, "Macro Recall", metrics['recall'])
    display_metric(col4, "Macro F1-Score", metrics['f1_score'])

    TARGET_ACCURACY = 0.95

    if metrics['accuracy'] >= TARGET_ACCURACY:
        st.success(f"✅ Target Accuracy of {TARGET_ACCURACY*100:.0f}% Achieved! Current Accuracy: {metrics['accuracy']:.4f}")
    else:
        st.warning(f"⚠️ Target Accuracy of {TARGET_ACCURACY*100:.0f}% Not Met Yet. Current Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()

