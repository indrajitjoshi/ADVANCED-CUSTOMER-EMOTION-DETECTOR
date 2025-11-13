import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
import os
from collections import Counter
import re

# Suppress TensorFlow logging messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Configuration ---
MAX_WORDS = 20000     # Max number of words to keep in the vocabulary
MAX_LEN = 128         # Max length of a sequence (increased for context)
EMBEDDING_DIM = 128   # Dimension of the word embeddings
RNN_UNITS = 200       # Capacity of LSTM/GRU cells
DENSE_UNITS = 512     # High capacity for complex feature separation
NUM_CLASSES = 6
EPOCHS = 10           # Increased for stable convergence and 95% accuracy goal
NUM_REVIEWS = 10      
TRAINABLE_EMBEDDING = True # Allow fine-tuning for better anti-bias integration

# Define the emotion labels for mapping
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}

# Custom Samples designed for clear classification tests
SAMPLE_REVIEWS = {
    "sadness": "I am not happy with this purchase. It makes me feel miserable and disappointed.",
    "joy": "This product is amazing and fills me with joy! I am absolutely ecstatic.",
    "love": "I absolutely adore the design and quality, I'm completely in love.",
    "anger": "It broke immediately and this makes me so furious and upset. I hate it.",
    "fear": "I am afraid to use this device after the smoke I saw, it is worrying.",
    "surprise": "Wow! I truly did not expect it to be this good. What a pleasant surprise." # Clear unexpectedness
}

# --- Preprocessing Function (Negation Handling) ---

def handle_negation(texts):
    """
    Handles negation by modifying phrases and appending a global negation flag.
    - Replaces space between a negator and the following word with an underscore.
    - Appends __NEGATED__ token to the end of the text if negation is detected.
    """
    negation_words = ['not', 'no', 'never', 'don\'t', 'isn\'t', 'wasn\'t', 'wouldn\'t', 
                      'couldn\'t', 'won\'t', 'can\'t', 'do not', 'did not', 'will not',
                      'hardly', 'scarcely']
    
    processed_texts = []
    
    for text in texts:
        did_negate = False
        words = text.split()
        
        # 1. Look for explicit negation words ('not', 'never', etc.)
        for i, word in enumerate(words):
            # Check for multi-word negators (e.g., 'do not')
            if word.lower() in ['do', 'did', 'will', 'was', 'is'] and i + 1 < len(words) and words[i+1].lower() == 'not':
                 # Treat 'do not' as one unit affecting the next word
                 if i + 2 < len(words):
                    words[i+2] = f"{word}_{words[i+1]}_{words[i+2]}"
                    words[i], words[i+1] = '', '' # Remove negators
                    did_negate = True
            
            # Check for standard contractions or single-word negators
            elif word.lower() in negation_words and i + 1 < len(words):
                # Create a specialized token (e.g., 'not_happy')
                words[i+1] = f"{word}_{words[i+1]}"
                words[i] = '' # Remove the negator itself after modifying the next word
                did_negate = True
            
            # 2. Look for prefixes (un, in, etc.)
            if re.match(r'^(un|in|im|ir)\w+', word.lower()):
                 did_negate = True
        
        # Rejoin words and remove empty strings from explicit negation
        new_text = " ".join(filter(None, words))
        
        # 3. Append global flag to signal negation presence
        if did_negate:
            new_text = f"{new_text} __NEGATED__"
            
        processed_texts.append(new_text)
        
    return processed_texts


# --- Ensemble Model Building Functions ---

def create_embedding_layer(num_words, embedding_matrix=None):
    """Creates the Embedding layer, initialized with pre-trained vectors if provided."""
    return Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        input_length=MAX_LEN,
        trainable=TRAINABLE_EMBEDDING
    )


def build_cnn_model(num_words, embedding_matrix):
    """Builds the deep CNN model (best for local features like negation)."""
    model = Sequential([
        create_embedding_layer(num_words, embedding_matrix),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'), # Deeper stack
        GlobalMaxPooling1D(),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(num_words, embedding_matrix):
    """Builds the stable two-layer BiLSTM model (best for long-range context)."""
    model = Sequential([
        create_embedding_layer(num_words, embedding_matrix),
        Dropout(0.3),
        Bidirectional(LSTM(RNN_UNITS, return_sequences=True, dropout=0.1)), # Stable Recurrent Dropout
        Bidirectional(LSTM(RNN_UNITS)),                                    # Final layer
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(num_words, embedding_matrix):
    """Builds the Bidirectional GRU model (efficient sequence processing)."""
    model = Sequential([
        create_embedding_layer(num_words, embedding_matrix),
        Dropout(0.3),
        Bidirectional(GRU(RNN_UNITS, dropout=0.1)),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Caching function to load data and train the model once ---

@st.cache_resource
def load_and_train_model():
    """Loads data, trains the Ensemble of three models, and evaluates them."""
    st.info("Loading and pre-processing dataset...")
    
    # 1. Load Data
    data = load_dataset("dair-ai/emotion", "split")
    
    # Combine train and validation splits for maximum data utilization
    train_texts = list(data['train']['text']) + list(data['validation']['text'])
    train_labels = list(data['train']['label']) + list(data['validation']['label'])
    test_texts = list(data['test']['text'])
    test_labels = list(data['test']['label'])
    
    # Apply Negation Preprocessing to all data
    st.info("Applying custom negation preprocessing...")
    train_texts = handle_negation(train_texts)
    test_texts = handle_negation(test_texts)
    
    all_texts = train_texts + test_texts

    # 2. Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(all_texts)
    
    # Determine the actual vocabulary size
    num_words = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
    
    # 3. Class Weighting (Anti-Bias Fix)
    st.info("Calculating class weights to counteract dataset bias...")
    total_samples = len(train_labels)
    class_counts = np.bincount(train_labels)
    class_weights = {}
    for i in range(NUM_CLASSES):
        class_weights[i] = total_samples / (NUM_CLASSES * class_counts[i])
        
    # 4. Simulated Transfer Learning Initialization (For Semantic Stability)
    st.info("Initializing embedding matrix (Simulated Transfer Learning)...")
    embedding_matrix = np.random.uniform(-0.05, 0.05, (num_words, EMBEDDING_DIM))
    
    # Set a unique initialization for our special tokens to force attention
    negated_index = tokenizer.word_index.get('__negated__', 0)
    if negated_index < num_words:
        embedding_matrix[negated_index] = np.random.uniform(-0.1, 0.1, EMBEDDING_DIM)


    # 5. Build and Train Ensemble Models
    st.info(f"Building and training ensemble of 3 deep learning models for up to {EPOCHS} epochs...")
    
    models = [
        build_cnn_model(num_words, embedding_matrix),
        build_bilstm_model(num_words, embedding_matrix),
        build_gru_model(num_words, embedding_matrix)
    ]

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=0 # Run silently
    )

    # Train all models silently
    for i, model in enumerate(models):
        st.info(f"Training Model {i+1}...")
        try:
            model.fit(
                train_padded, 
                train_labels_one_hot,
                epochs=EPOCHS, 
                batch_size=32,
                validation_split=0.1,
                class_weight=class_weights, # Apply anti-bias weights
                callbacks=[early_stopping],
                verbose=0 # Run silently
            )
        except Exception as e:
            st.error(f"Error during training Model {i+1}: {e}")
            # Non-critical models can be skipped if one fails
            continue
            
    # 6. Ensemble Prediction and Evaluation
    
    # Get predictions (probabilities) from all models
    pred_probs_list = [model.predict(test_padded, verbose=0) for model in models if model is not None]
    
    if not pred_probs_list:
        st.error("All models failed to train. Cannot proceed with evaluation.")
        return models, tokenizer, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
    # Soft Voting: Average the probabilities across all models
    ensemble_probs = np.mean(pred_probs_list, axis=0)
    
    # Final prediction based on ensemble average
    y_pred = np.argmax(ensemble_probs, axis=1)
    
    # Calculate Metrics based on ensemble prediction
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

    st.success(f"Model Training Complete! Ensemble Accuracy: {accuracy:.4f}")
    return models, tokenizer, metrics


# --- Prediction Function ---
def predict_emotion(ensemble_models, tokenizer, text):
    """Predicts the emotion of a given review text using the ensemble models (Soft Voting)."""
    # 1. Apply the same negation preprocessing as training
    preprocessed_text = handle_negation([text])[0]
    
    # 2. Tokenize and pad
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict using ensemble
    pred_probs_list = [model.predict(padded_sequence, verbose=0) for model in ensemble_models if model is not None]
    
    if not pred_probs_list:
        return "ERROR", None
        
    ensemble_prediction = np.mean(pred_probs_list, axis=0)[0] 
    
    predicted_id = np.argmax(ensemble_prediction)
    predicted_label = id_to_label[predicted_id].capitalize()
    
    return predicted_label


# --- Core Analysis Logic ---
def get_recommendation_and_comment(all_emotions):
    """
    Determines the overall recommended emotion and the buy/no-buy comment.
    Tie-breaking logic: If there's a tie for the highest count, use the emotion from the LAST review (index 9).
    """
    if not all_emotions:
        return "N/A", "Please enter at least one review for analysis.", {}

    emotion_counts = Counter(all_emotions)
    
    # 1. Find the maximum count
    max_count = max(emotion_counts.values())
    
    # 2. Find all emotions that share the maximum count
    top_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]

    # 3. Determine the final dominant emotion (Tie-breaking: use the last review's emotion)
    dominant_emotion = top_emotions[0] # Default choice
    
    if len(top_emotions) > 1:
        last_review_emotion = all_emotions[-1]
        if last_review_emotion in top_emotions:
            dominant_emotion = last_review_emotion
        # If the last review emotion wasn't part of the tie, we stick to the default (alphabetically first)

    # 4. Generate Comment
    # We must consider 'Surprise' as ambiguous. Only 'Joy' and 'Love' are strong BUY signals.
    positive_buy_emotions = ['Joy', 'Love']
    
    if dominant_emotion in positive_buy_emotions:
        comment = (
            f"**Recommendation: BUY!** The dominant sentiment is strongly positive ({dominant_emotion}), "
            "indicating a highly satisfied customer base. This is a strong indicator of product quality."
        )
    elif dominant_emotion == 'Surprise':
        comment = (
            f"**Recommendation: CAUTIOUS BUY.** The dominant sentiment is 'Surprise'. While positive, "
            "it suggests a wide range of unexpected outcomes. Review customer feedback carefully."
        )
    else:
        # Sadness, Anger, Fear
        comment = (
            f"**Recommendation: DO NOT BUY.** The dominant sentiment is negative ({dominant_emotion}), "
            "suggesting significant customer dissatisfaction or potential product issues. Caution is advised."
        )

    return dominant_emotion, comment, emotion_counts

# --- Main Streamlit App ---
def main():
    
    # Initialize session state for analysis results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'product_name' not in st.session_state:
        st.session_state.product_name = "Affectlytics Product"
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False


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

        h1, h2, h3, h4 { color: #FFFFFF !important; letter-spacing: 1.2px; }
        .header {
            color: #FFFFFF; text-align: center; padding: 15px; border-radius: 12px;
            background: rgba(0, 0, 0, 0.4); box-shadow: 0 4px 10px rgba(0, 0, 0, 0.7);
            margin-bottom: 25px;
        }
        .stButton>button {
            border-radius: 10px; transition: all 0.3s ease; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            color: white; background-color: #5b1076;
        }
        .stButton>button:hover {
            transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6); background-color: #7b1296;
        }
        .stTable tbody tr th, .stTable tbody tr td {
            color: #FFFFFF !important; background-color: rgba(0, 0, 0, 0.5) !important;
            border-bottom: 1px solid #3d0a52; word-break: normal; white-space: normal;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: rgba(0, 0, 0, 0.3) !important; color: white !important;
            border: 1px solid #7b1296;
        }
        /* Specific styling for the recommendation box */
        .recommendation-box {
            background-color: #3d0a52; padding: 20px; border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.8); margin-top: 20px; border: 2px solid #FFD700;
        }
        .emotion-badge {
            display: inline-block; padding: 5px 10px; border-radius: 8px; font-weight: 600;
            margin-left: 5px; color: white; background-color: #7b1296;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header"><h1><span style="color: #FFD700;">🛍️</span> Affectlytics: Product Sentiment Analyzer <span style="color: #FFD700;">📊</span></h1></div>', unsafe_allow_html=True)
    
    # Load and train the model (cached)
    try:
        models, tokenizer, metrics = load_and_train_model()
        st.session_state.is_trained = True
    except Exception as e:
        st.error(f"Failed to load or train model: {e}")
        st.session_state.is_trained = False
        return

    # --- Input Section ---
    st.markdown("---")
    st.markdown("<h3 style='color: white;'>Input Product Details and Reviews (10 Required):</h3>", unsafe_allow_html=True)
    
    # Product Name Input
    product_name_input = st.text_input(
        "Product Name", 
        value=st.session_state.product_name,
        key="product_name_key"
    )
    st.session_state.product_name = product_name_input
    
    # 10 Review Text Inputs
    review_inputs = []
    
    # Use columns for a cleaner 2-column layout for reviews
    cols = st.columns(2)

    for i in range(NUM_REVIEWS):
        col_index = i % 2
        with cols[col_index]:
            default_review = SAMPLE_REVIEWS[emotion_labels[i % 6]] if i < 6 else f"Review #{i+1} Example: This is fine, but nothing special."
            # Use sticky input value if available
            sticky_value = st.session_state.analysis_results[i]['review'] if len(st.session_state.analysis_results) == NUM_REVIEWS else default_review
            
            review_text = st.text_area(
                f"Review #{i+1}", 
                value=sticky_value, 
                height=100, 
                key=f"review_{i}"
            )
            review_inputs.append(review_text)

    # --- Analysis Button ---
    if st.button("Analyze All 10 Reviews", use_container_width=True, type="primary"):
        if not st.session_state.is_trained:
            st.warning("Model is not fully initialized. Please wait a moment and try again.")
            return

        st.session_state.analysis_results = []
        all_emotions = []

        # Process each review
        for i, review in enumerate(review_inputs):
            if not review.strip():
                st.warning(f"Review #{i+1} is empty. Please fill all 10 inputs.")
                st.session_state.analysis_results = [] # Clear results if incomplete
                return

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


    # --- Results Display ---
    if st.session_state.analysis_results:
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
        
        count_data = pd.DataFrame(
            st.session_state.emotion_counts.items(), 
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


    # --- Evaluation Metrics Display ---
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

    if st.session_state.is_trained:
        display_metric(col1, "Accuracy", metrics['accuracy'])
        display_metric(col2, "Macro Precision", metrics['precision'])
        display_metric(col3, "Macro Recall", metrics['recall'])
        display_metric(col4, "Macro F1-Score", metrics['f1_score'])

        TARGET_ACCURACY = 0.95
        
        if metrics['accuracy'] >= TARGET_ACCURACY:
            st.success(f"✅ Target Accuracy of {TARGET_ACCURACY*100:.0f}% Achieved! Current Accuracy: {metrics['accuracy']:.4f}")
        else:
            st.warning(f"⚠️ Target Accuracy of {TARGET_ACCURACY*100:.0f}% Not Met Yet. Current Accuracy: {metrics['accuracy']:.4f}")
    else:
        st.warning("Metrics unavailable: Model failed to initialize or train.")

if __name__ == "__main__":
    main()
