# 🎬 Movie Sentiment Analysis API

Predicts whether movie reviews are **Fresh** or **Rotten** based on Rotten Tomatoes critic data.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Model** | Logistic Regression |
| **Accuracy** | 79.6% |
| **ROC-AUC** | 87.9% |
| **Training Data** | 278,000 Top Critic Reviews |
| **Inference Time** | <1ms |
| **Threshold** | 0.45 (optimized) |

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/AceKinnn/sentiment-analysis-rotten-tomatoes.git
cd sentiment-analysis-rotten-tomatoes/backend

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py