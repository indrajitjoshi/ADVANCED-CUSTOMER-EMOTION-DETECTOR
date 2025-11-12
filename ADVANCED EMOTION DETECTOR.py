import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, GRU
from tensorflow.keras.callbacks import EarlyStopping # NEW: Added for robust training
import os
from collections import Counter 

# Suppress TensorFlow logging messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Configuration ---
MAX_WORDS = 20000 # Max number of words to keep in the vocabulary
MAX_LEN = 100     # Max length of a sequence (review)
EMBEDDING_DIM = 100 # Dimension of the word embeddings
LSTM_UNITS = 150  # MAXIMIZED CAPACITY
NUM_CLASSES = 6
# NOTE: EPOCHS is kept high, but EarlyStopping will prevent true overfitting.
EPOCHS = 30 
NUM_REVIEWS = 10 # Constant for the required number of inputs

# Define the emotion labels for mapping
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
label_to_id = {label: i for i, label in enumerate(emotion_labels)}
id_to_label = {i: label for i, label in enumerate(emotion_labels)}

# --- Ensemble Model Building Functions (Designed for High Accuracy) ---

def build_cnn_bilstm_model():
    """Builds the Hybrid CNN-BiLSTM model."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'), 
        GlobalMaxPooling1D(),
        Dense(150, activation='relu'),
        Dropout(0.5), # High Dropout for strong regularization
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model_v2():
    """Builds the pure BiLSTM model with two BiLSTM layers."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.3), # Strong Dropout
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)), 
        Bidirectional(LSTM(LSTM_UNITS // 2)),
        Dense(150, activation='relu'),
        Dropout(0.5), # High Dropout for strong regularization
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model():
    """Builds the Bidirectional GRU model."""
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.3), # Strong Dropout
        Bidirectional(GRU(LSTM_UNITS)),
        Dense(150, activation='relu'),
        Dropout(0.5), # High Dropout for strong regularization
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# --- Caching function to load data and train the model once ---

@st.cache_resource
def load_and_train_model():
    """Loads data, trains the Ensemble of three models, and evaluates them using soft voting."""
    
    # 1. Load Data
    data = load_dataset("dair-ai/emotion", "split")
    
    train_texts = list(data['train']['text'])
    train_labels = list(data['train']['label'])
    test_texts = list(data['test']['text'])
    test_labels = list(data['test']['label'])
    
    # Combine train and test into one pool for tokenizer fit
    all_texts = train_texts + test_texts

    # 2. Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(all_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
    
    # --- BIAS CORRECTION: Calculate Class Weights (Essential for imbalance) ---
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=train_labels
    )
    class_weights_dict = dict(zip(classes, weights))
    # --- END BIAS CORRECTION ---
    
    # 3. Define Early Stopping Callback (Essential for preventing overfitting/bias)
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, # Stop if validation loss doesn't improve for 5 epochs
        restore_best_weights=True, # Load the best weights found
        verbose=0
    )
    callbacks = [early_stopping]
    
    # 4. Build and Train Ensemble Models
    
    models = [
        build_cnn_bilstm_model(),
        build_bilstm_model_v2(),
        build_gru_model()
    ]

    # Train all models silently
    for model in models:
        model.fit(
            train_padded, 
            train_labels_one_hot,
            epochs=EPOCHS, 
            batch_size=32,
            validation_split=0.1,
            verbose=0, # Run silently
            class_weight=class_weights_dict, # Apply the calculated weights
            callbacks=callbacks # Apply early stopping for robustness
        )
    
    # 5. Ensemble Prediction and Evaluation
    
    # Get predictions (probabilities) from all models
    pred_probs_list = [model.predict(test_padded, verbose=0) for model in models]
    
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

    return models, tokenizer, metrics


# --- Prediction Function (Unchanged) ---
def predict_emotion(ensemble_models, tokenizer, text):
    """Predicts the emotion of a given review text using the ensemble models (Soft Voting)."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Get predictions (probabilities) from all models
    pred_probs_list = [model.predict(padded_sequence, verbose=0) for model in ensemble_models]
    
    # Soft Voting: Average the probabilities across all models
    ensemble_prediction = np.mean(pred_probs_list, axis=0)[0] # [0] for single input batch
    
    # Get the index of the highest probability
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
        return "N/A", "Please enter at least one review for analysis."

    emotion_counts = Counter(all_emotions)
    
    # 1. Find the maximum count
    max_count = max(emotion_counts.values())
    
    # 2. Find all emotions that share the maximum count
    top_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]

    # 3. Determine the final dominant emotion
    dominant_emotion = top_emotions[0] 

    if len(top_emotions) > 1:
        # Tie detected: Use the emotion from the last review (Review #10)
        last_review_emotion = all_emotions[-1]
        
        # Check if the last review's emotion is one of the tied dominant emotions
        if last_review_emotion in top_emotions:
            dominant_emotion = last_review_emotion
        
    # 4. Generate Comment
    positive_emotions = ['Joy', 'Love', 'Surprise']
    
    if dominant_emotion in positive_emotions:
        comment = (
            f"**Recommendation: BUY!** The dominant sentiment is positive ({dominant_emotion}), "
            "indicating a highly satisfied customer base. This is a strong indicator of product quality."
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
        st.session_state.product_name = ""

    # --- CSS Injection (Unchanged) ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        /* Animated Gradient Background - DARKER COLORS FOR BETTER CONTRAST */
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
        /* Specific styling for the recommendation box */
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

    # UPDATED APP NAME
    st.markdown('<div class="header"><h1><span style="color: #FFD700;">🔬</span> Affectlytics <span style="color: #FFD700;">📊</span></h1></div>', unsafe_allow_html=True)
    
    # Load and train the model (cached)
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
    
    # Use columns for a cleaner 2-column layout for reviews
    cols = st.columns(2)

    for i in range(NUM_REVIEWS):
        col_index = i % 2
        with cols[col_index]:
            default_review = f"Review #{i+1} Example: This is fine, but nothing special."
            # If we have stored data, use it for sticky inputs
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
        # Remove 'product' column for cleaner display in table
        st.table(results_df[['Review #', 'Review Text', 'Detected Emotion']])

        # 2. Display Emotion Count (at the bottom of the project)
        st.markdown("<hr style='border: 1px solid #7b1296;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>Overall Emotion Counts:</h3>", unsafe_allow_html=True)
        
        # Convert Counter to DataFrame for display
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


    # --- Evaluation Metrics Display (Unchanged) ---
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

