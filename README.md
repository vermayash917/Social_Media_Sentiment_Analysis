# Sentiment Analysis on Social Media Posts

## Introduction

This repository contains the implementation of a **multimodal sentiment analysis pipeline**, designed to classify sentiments expressed in social media posts. The project focuses on analyzing **images**, **captions**, and **comments** to classify emotions into three categories: **Happy**, **Sad**, and **Angry**. By combining visual and textual modalities, this framework leverages cutting-edge deep learning models to provide a comprehensive understanding of sentiment in social media content.

The framework addresses a critical research gap by effectively integrating visual and textual data using state-of-the-art models, including **CLIP** for image analysis and **BERTweet** for text-based sentiment classification. A **late fusion technique** is employed to merge predictions from both modalities, ensuring robust sentiment predictions for multimodal content.

---

## Features

- **Multimodal Analysis**: Combines visual and textual data for more accurate sentiment analysis.
- **Image Sentiment Analysis**: Uses **CLIP embeddings** with a custom neural network for classification.
- **Text Sentiment Analysis**: Employs a fine-tuned **BERTweet model** for analyzing captions and comments.
- **Late Fusion**: Integrates predictions from both modalities to deliver unified sentiment insights.
- **Scalable Framework**: Modular design supports easy adaptation for new datasets or additional sentiment classes.

---

## Models Used

### CLIP
- Extracts visual features from images and maps them to a high-dimensional embedding space.
- Paired with a custom neural network for sentiment classification.

### BERTweet
- Pre-trained transformer model optimized for social media text.
- Fine-tuned for sentiment classification of captions and comments.

---

## Dataset

- **Images**:
  - Instagram posts categorized into three sentiments: **Happy**, **Sad**, and **Angry**.
  - **Training set**: 5,799 images per class.
  - **Testing set**: 1,343 images per class.

- **Text**:
  - Corresponding captions and comments labeled for sentiment classification.

---

## Architecture

### Image Sentiment Analysis
- Visual features are extracted using **CLIP**.
- Features are passed through a lightweight custom neural network for classification.

### Text Sentiment Analysis
- Captions and comments are preprocessed to handle social media-specific elements like emojis, hashtags, and mentions.
- Sentiments are predicted using a fine-tuned **BERTweet model**.

### Multimodal Fusion
- A **late fusion technique** combines logits from the image and caption models.
- The weighted average of the logits produces a unified sentiment prediction.

---

## Workflow

### Preprocessing
- **Text**: Emojis are converted to textual representations, hashtags and mentions are retained, and URLs are removed.
- **Images**: Processed using CLIP's preprocessing pipeline to ensure consistency.

### Model Training
- **Image classifier** trained on CLIP embeddings.
- **BERTweet model** fine-tuned on labeled textual data.

### Sentiment Prediction
- Individual predictions are made for images, captions, and comments.
- Combined predictions are calculated using **late fusion**.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vermayash917/Social_Media_Sentiment_Analysis.git
   cd Social_Media_Sentiment_Analysis
