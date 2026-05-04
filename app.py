from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import tensorflow as tf
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
MODEL_PATH = MODELS_DIR / "sentiment_model.keras"
VECTORIZER_PATH = MODELS_DIR / "text_vectorizer"


EMOJI_MAP = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
    ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
    ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
    '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
    ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}


def clean_text(text: str) -> str:
    normalized = text
    for emoji, meaning in EMOJI_MAP.items():
        normalized = normalized.replace(emoji, meaning)
    normalized = re.sub(r'http\S+', '', normalized)
    normalized = re.sub(r'@\w+', '', normalized)
    normalized = re.sub(r'#\w+', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def build_sentiment_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(200,), dtype='int32'),
        tf.keras.layers.Embedding(20000, 128),
        tf.keras.layers.SpatialDropout1D(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    return model


def load_artifacts() -> tuple[tf.keras.Model, tf.keras.Model]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    def copy_to_temp_h5(path: Path) -> Path:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        shutil.copyfile(path, temp_path)
        return temp_path

    with MODEL_PATH.open('rb') as model_file:
        model_signature = model_file.read(8)

    if model_signature.startswith(b'\x89HDF\r\n\x1a\n'):
        temp_model_path = copy_to_temp_h5(MODEL_PATH)
        model = build_sentiment_model()
        try:
            model.load_weights(temp_model_path)
        finally:
            if temp_model_path.exists():
                temp_model_path.unlink()
    else:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Prioritize SavedModel directory (has full state) over .keras file (vocabulary not persisted)
    vectorizer_source = None
    if VECTORIZER_PATH.exists() and VECTORIZER_PATH.is_dir():
        vectorizer_source = VECTORIZER_PATH
    else:
        vectorizer_file = MODELS_DIR / 'text_vectorizer.keras'
        if vectorizer_file.exists():
            vectorizer_source = vectorizer_file

    if vectorizer_source is None:
        raise FileNotFoundError('Missing vectorizer artifact in Models/')

    # Load from SavedModel directory or .keras file
    if isinstance(vectorizer_source, Path) and vectorizer_source.is_dir():
        # Use TFSMLayer for legacy SavedModel format (Keras 3 doesn't support direct SavedModel loading)
        vectorizer_model = tf.keras.layers.TFSMLayer(str(vectorizer_source), call_endpoint='serving_default')
    else:
        # Try loading as .keras or HDF5 file
        try:
            vectorizer_model = tf.keras.models.load_model(str(vectorizer_source), compile=False)
        except Exception:
            # If .keras fails, try treating as HDF5
            if vectorizer_source.is_file():
                temp_vectorizer_path = copy_to_temp_h5(vectorizer_source)
                try:
                    vectorizer_model = tf.keras.models.load_model(temp_vectorizer_path, compile=False)
                finally:
                    if temp_vectorizer_path.exists():
                        temp_vectorizer_path.unlink()
            else:
                raise

    return model, vectorizer_model


MODEL, VECTORIZER_MODEL = load_artifacts()


def predict_sentiment(text: str) -> dict[str, object]:
    cleaned_text = clean_text(text)
    vectorized = VECTORIZER_MODEL(tf.constant([cleaned_text]))
    
    # Extract tensor from TFSMLayer dict output
    if isinstance(vectorized, dict):
        # TFSMLayer returns dict; extract the actual output
        vectorized_text = list(vectorized.values())[0]
    else:
        vectorized_text = vectorized
    
    score = float(MODEL.predict(vectorized_text, verbose=0)[0][0])
    is_positive = score >= 0.5
    label = 'Positive' if is_positive else 'Negative'
    confidence = score if is_positive else 1 - score

    return {
        'label': label,
        'score': score,
        'confidence': round(confidence * 100, 2),
        'cleaned_text': cleaned_text,
    }


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024


@app.get('/')
def index() -> str:
    return render_template(
        'index.html',
        prediction=None,
        text_value='',
        error=None,
        examples=[
            'I love how this product solved my problem so quickly.',
            'This update broke everything and the experience is frustrating.',
            'The service was fine, but I expected a little more polish.',
        ],
    )


@app.post('/predict')
def predict() -> str:
    text_value = request.form.get('text', '').strip()

    if not text_value:
        return render_template(
            'index.html',
            prediction=None,
            text_value='',
            error='Please enter a sentence or short paragraph to analyze.',
            examples=[
                'I love how this product solved my problem so quickly.',
                'This update broke everything and the experience is frustrating.',
                'The service was fine, but I expected a little more polish.',
            ],
        )

    prediction = predict_sentiment(text_value)
    return render_template(
        'index.html',
        prediction=prediction,
        text_value=text_value,
        error=None,
        examples=[
            'I love how this product solved my problem so quickly.',
            'This update broke everything and the experience is frustrating.',
            'The service was fine, but I expected a little more polish.',
        ],
    )


@app.post('/api/predict')
def api_predict() -> tuple[dict[str, object], int]:
    payload = request.get_json(silent=True) or {}
    text_value = str(payload.get('text', '')).strip()

    if not text_value:
        return {'error': 'Missing text field.'}, 400

    return predict_sentiment(text_value), 200


@app.get('/health')
def health() -> tuple[dict[str, str], int]:
    return {'status': 'ok'}, 200


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=debug_mode)