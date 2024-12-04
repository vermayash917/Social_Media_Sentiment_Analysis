import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image
import clip
import torch.nn as nn
import os

# Device setup: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned BERTweet model and tokenizer
bertweet_tokenizer = BertTokenizer.from_pretrained(
    "models/BERTweet_model/fine_tuned_bertweet_tokenizer_balanced"
)
bertweet_model = BertForSequenceClassification.from_pretrained(
    "models/BERTweet_model/fine_tuned_bertweet_model_balanced"
).to(device)

# Load the CLIP model for extracting image embeddings
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# Define the architecture of the CLIP sentiment model
class CLIPSentimentModel(nn.Module):
    def __init__(self, clip_embedding_dim, hidden_dim, num_classes):
        super(CLIPSentimentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# Instantiate the model architecture
clip_sentiment_model = CLIPSentimentModel(clip_embedding_dim=512, hidden_dim=256, num_classes=3).to(device)

# Load the pre-trained CLIP sentiment model weights
state_dict = torch.load("models/CLIP_model/clip_sentiment_model.pth", map_location=device)
clip_sentiment_model.load_state_dict(state_dict)
clip_sentiment_model.eval()

# Image preprocessing pipeline (CLIP's preprocess method)
image_transforms = clip_preprocess


def get_text_sentiment(text):
    """
    Predict sentiment from text using the fine-tuned BERTweet model.
    Args:
        text (str): Input text (caption or comment).
    Returns:
        torch.Tensor: Logits for the sentiment classes.
    """
    tokens = bertweet_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = bertweet_model(**tokens).logits
    return logits


def get_image_sentiment(image_path):
    """
    Predict sentiment from an image using the CLIP sentiment model.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple(torch.Tensor, torch.Tensor): Image embedding and sentiment logits.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        logits = clip_sentiment_model(image_features)
    return image_features, logits


def analyze_multimodal(image_path, caption, comment):
    """
    Perform multimodal sentiment analysis for an image, caption, and comment.
    Args:
        image_path (str): Path to the input image.
        caption (str): Caption text.
        comment (str): Comment text.
    Returns:
        dict: Sentiment probabilities for individual and combined modalities.
    """
    # Get individual sentiments
    image_features, image_logits = get_image_sentiment(image_path)
    image_sentiment = torch.softmax(image_logits, dim=-1).cpu().numpy()

    caption_logits = get_text_sentiment(caption)
    caption_sentiment = torch.softmax(caption_logits, dim=-1).cpu().numpy()

    comment_logits = get_text_sentiment(comment)
    comment_sentiment = torch.softmax(comment_logits, dim=-1).cpu().numpy()

    # Late fusion: Combine image and caption logits
    combined_logits = 0.5 * image_logits + 0.5 * caption_logits
    image_caption_sentiment = torch.softmax(combined_logits, dim=-1).cpu().numpy()

    return {
        "image_sentiment": image_sentiment,
        "caption_sentiment": caption_sentiment,
        "comment_sentiment": comment_sentiment,
        "image_caption_sentiment": image_caption_sentiment,
    }
