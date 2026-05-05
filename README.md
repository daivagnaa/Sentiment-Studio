# Sentiment Analysis Project

This project implements a comprehensive sentiment analysis solution using deep learning with LSTM (Long Short-Term Memory) neural networks. The project analyzes the sentiment of text as either Positive or Negative, trained on 1.6 million real tweets. It includes data preprocessing, model training, and deployment through a professional Flask web application with REST API support.

## 🚀 Live Demo

**Web Application**: [Sentiment Analysis Studio](https://sentiment-studio.onrender.com)
## Project Structure

```
Sentiment Analysis/
│
├── README.md                        # Project documentation
├── app.py                          # Flask application with routes and inference
├── requirements.txt                # Python dependencies
├── test_inference.py              # Inference testing script
├── inspect_vectorizer.py          # Model artifact inspection utility
│
├── Data/
│   └── training.1600000.processed.noemoticon.csv   # 1.6M tweets dataset
│
├── Models/
│   ├── sentiment_model.keras      # Saved sentiment classifier model
│   ├── text_vectorizer.keras      # Text preprocessing layer
│   └── text_vectorizer/           # SavedModel format vectorizer
│       ├── saved_model.pb
│       ├── keras_metadata.pb
│       ├── assets/
│       └── variables/
│
├── templates/
│   ├── base.html                  # Base Jinja2 template
│   └── index.html                 # Main web interface
│
└── static/
    └── css/
        └── style.css              # Professional styling and animations
```

## Dataset

The project uses **1.6 Million Real Tweets Dataset** with pre-processed sentiment labels:
- **Total Samples**: 1,600,000 tweets
- **Features**: Raw tweet text (cleaned and normalized)
- **Target Variable**: Sentiment (0: Negative, 4: Positive, encoded as 0-1 for binary classification)
- **Data Format**: CSV with text and sentiment columns
- **Preprocessing**: Emoji handling, URL removal, mention/hashtag filtering, standardization

## Model Architecture

**Deep Learning Neural Network**:
```
Input Layer
    ↓
Embedding Layer (20,000 tokens, 128 dimensions)
    ↓
SpatialDropout1D (0.3 dropout rate)
    ↓
Bidirectional LSTM (128 units, return sequences)
    ↓
GlobalMaxPooling1D
    ↓
Dense Layer (128 units, ReLU activation, 0.6 dropout)
    ↓
Dense Layer (64 units, ReLU activation, 0.4 dropout)
    ↓
Dense Layer (32 units, ReLU activation, 0.2 dropout)
    ↓
Output Layer (1 unit, Sigmoid activation)
    ↓
Prediction (0-1 confidence score)
```

**Key Components**:
- **Embedding Layer**: Converts text tokens to dense vectors
- **Bidirectional LSTM**: Captures contextual information from both directions
- **Dropout Layers**: Prevents overfitting with decreasing rates
- **Sigmoid Output**: Produces probability scores for binary classification

## Workflow

1. **Data Collection and Analysis**
   - Load 1.6 million tweet samples
   - Analyze sentiment distribution
   - Explore text patterns and characteristics

2. **Data Preprocessing**
   - Text normalization and cleaning
   - Emoji replacement with semantic meanings
   - URL, mention, and hashtag removal
   - Tokenization and sequence padding

3. **Model Training**
   - Initialize deep LSTM architecture
   - Train on preprocessed tweet data
   - Use categorical cross-entropy loss
   - Adam optimizer for parameter updates
   - Batch training with validation splits

4. **Text Vectorization**
   - Create TextVectorization layer
   - Learn vocabulary from training data
   - Generate fixed-length sequences (200 tokens)
   - Standardize text with lowercase and punctuation removal

5. **Model Evaluation**
   - Validation accuracy on held-out test set
   - Binary classification metrics
   - Confidence score analysis
   - Inference speed benchmarking

6. **Model Deployment**
   - Save trained model and vectorizer
   - Create Flask REST API
   - Build professional web interface
   - Deploy with production-ready configuration

## Key Features

### 🤖 Machine Learning Pipeline
- **Deep LSTM Architecture**: Bidirectional sequence modeling for context understanding
- **Text Vectorization**: Efficient token-to-sequence conversion with vocabulary learning
- **Emoji Replacement**: 30+ emoji symbols converted to semantic text representations
- **Advanced Preprocessing**: URL removal, mention/hashtag filtering, standardization
- **Dropout Regularization**: Progressive dropout rates to prevent overfitting

### 💻 Prediction Systems

1. **REST API Endpoint** (`/api/predict`):
   - JSON-based predictions
   - Accepts raw text input
   - Returns sentiment label and confidence score
   - CORS-enabled for cross-origin requests
   ```bash
   curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
   ```

2. **Professional Web Interface** (`Flask Web App`):
   - 🎨 **Modern Dark Theme**: Gradient backgrounds with glassmorphism effects
   - 📱 **Fully Responsive**: Optimized for desktop, tablet, and mobile
   - ⚡ **Real-time Analysis**: Instant sentiment predictions with smooth transitions
   - 🎯 **Interactive Examples**: Pre-loaded sentiment samples for quick testing
   - 📊 **Visual Confidence Meter**: Gradient bar showing sentiment strength
   - 💡 **Text Preprocessing Display**: Shows cleaned text after processing
   - 🎭 **Emoji Indicators**: Visual representation of sentiment with emojis
   - 📈 **Detailed Metrics**: Model score, confidence percentage, processed text
   - 🔄 **Smooth UX**: Hover effects, animations, and loading states

## Installation & Usage

### Prerequisites
```bash
pip install flask tensorflow keras numpy
```

### 🌐 Access Web Application

#### Online Demo
Visit the live demo: **[Sentiment Analysis Studio]**(https://sentiment-studio.onrender.com)

#### 💻 Run Locally

**Step 1: Clone/Navigate to Project**
```bash
cd "e:\Data Science\Projects\Sentiment Analysis"
```

**Step 2: Activate Virtual Environment**
```bash
# Create venv (if not exists)
python -3.12 -m venv .venv

# Activate venv
.\.venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Launch Flask Application**
```bash
python app.py
```

**Step 5: Access Web Interface**
Open browser to: `http://127.0.0.1:5000`

### 🧪 Running Inference Tests
```bash
python test_inference.py
```

## 🎯 Model Performance

The LSTM classifier demonstrates strong performance on sentiment classification:

| Metric | Value |
|--------|-------|
| **Architecture** | Bidirectional LSTM with Embedding |
| **Training Samples** | 1,600,000 tweets |
| **Vocabulary Size** | 20,000 tokens |
| **Sequence Length** | 200 tokens (max) |
| **Test Accuracy** | High-precision classification |
| **Inference Speed** | ~50ms per sample |
| **Model Size** | 34.5 MB (sentiment_model) |

## 📝 API Usage

### POST /api/predict
**Request**:
```json
{
  "text": "I absolutely love this amazing product!"
}
```

**Response**:
```json
{
  "label": "Positive",
  "score": 0.9899,
  "confidence": 98.99,
  "cleaned_text": "I absolutely love this amazing product!"
}
```

### POST /predict (Web Form)
Submit sentiment analysis through the web interface form.

### GET /health
Check API health status:
```bash
curl http://127.0.0.1:5000/health
```
Response: `{"status": "ok"}`

## 🛠 Technologies Used

- **Python 3.12**: Core programming language
- **TensorFlow 2.21.0**: Deep learning framework
- **Keras 3.14.0**: Neural network API
- **NumPy 2.4.4**: Numerical computing
- **Flask 3.1.3**: Web framework
- **Jinja2**: Template engine
- **LSTM**: Sequence-to-sequence modeling
- **TextVectorization**: Token preprocessing layer
- **Pickle/Pickle-like**: Model serialization
- **CSS3**: Modern styling and animations
- **JavaScript**: Interactive UI features

## 🌐 Deployment

### Local Development
- **Framework**: Flask development server
- **Port**: 5000 (default)
- **Debug Mode**: Can be enabled via environment variable

### Production Deployment Options
- **Heroku**: Procfile-ready configuration
- **AWS**: EC2 or Elastic Beanstalk
- **Azure**: App Service or Container Instances
- **Google Cloud**: Cloud Run or App Engine
- **PythonAnywhere**: Python hosting platform

**Recommended for Production**:
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🔮 Future Enhancements

- [ ] 📊 Add sentiment intensity levels (Very Negative, Negative, Neutral, Positive, Very Positive)
- [ ] 🎯 Implement aspect-based sentiment analysis
- [ ] 🔍 Add sarcasm and irony detection
- [ ] 📈 Include confidence calibration
- [ ] 🌍 Multi-language sentiment analysis
- [ ] 📱 Create mobile application (React Native/Flutter)
- [ ] 🔐 Add user authentication and history tracking
- [ ] 📊 Batch processing for large text files
- [ ] 🎨 Advanced visualization dashboard
- [ ] ☁️ Docker containerization
- [ ] 🚀 GraphQL API support
- [ ] 💾 Implement caching layer for repeated queries
- [ ] 🤖 Fine-tune on domain-specific data
- [ ] 📊 Add explainability features (attention visualization)

## 🧠 Model Training Notes

### Training Data
- **Dataset**: 1.6 Million tweets with sentiment labels
- **Split**: 80% training, 20% validation/testing
- **Preprocessing**: Standardization, tokenization, padding to 200 tokens
- **Augmentation**: Emoji replacement for better context

### Hyperparameters
- **Embedding Dimension**: 128
- **LSTM Units**: 128 (bidirectional)
- **Dropout Rates**: 0.3, 0.6, 0.4, 0.2 (progressive)
- **Dense Units**: 128 → 64 → 32
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 32-64 (during training)

## 🔐 Security & Disclaimers

⚠️ **Sentiment Analysis Disclaimer**:
- Results are AI predictions and may not perfectly reflect human sentiment
- Use for informational purposes only
- Not suitable for critical decision-making without human review
- Model trained on Twitter data; performance may vary on other text sources

## 📝 Text Preprocessing

Input text is automatically processed:
1. **Emoji Replacement**: Emojis converted to text (e.g., 😊 → smile)
2. **URL Removal**: All hyperlinks stripped
3. **Mention Removal**: @username references removed
4. **Hashtag Removal**: # symbols and associated text removed
5. **Whitespace Normalization**: Multiple spaces collapsed to single space
6. **Case Normalization**: Converted to lowercase
7. **Punctuation Handling**: Standardized punctuation handling

## 👨‍💻 Developer

**Daivagna Parmar**
- 📧 **Email**: [devparmar1895@gmail.com](mailto:devparmar1895@gmail.com)
- 🔗 **GitHub**: [@daivagnaa](https://github.com/daivagnaa)
- 💼 **LinkedIn** : [Daivagna Parmar](https://in.linkedin.com/in/daivagna-parmar-949315316)

## 📜 License

This project is open-source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports and feature requests
- Propose improvements to the model or interface
- Share feedback and suggestions
- Create pull requests with enhancements

---

*This project demonstrates a complete deep learning pipeline from data preprocessing to model deployment, showcasing practical implementation of sentiment analysis using LSTM neural networks with a beautiful, production-ready web interface.*

**⭐ Star this repository if you found it helpful!**

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Activate virtual environment: `.\.venv\Scripts\activate`
- [ ] Run Flask app: `python app.py`
- [ ] Open browser: `http://127.0.0.1:5000`
- [ ] Test inference: `python test_inference.py`
- [ ] Explore API: POST to `/api/predict`
- [ ] Update live demo link in README

---

**Last Updated**: May 5, 2026  
**Version**: 1.0  
**Status**: ✅ Production Ready
