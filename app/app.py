import sys
import os
from flask import Flask, render_template, request
from scripts.multimodal_pipeline import analyze_multimodal

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Explicitly set the template folder path
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    """
    Render the homepage with the input form for sentiment analysis.
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Process user input and perform sentiment analysis.
    """
    # Get uploaded image and text inputs
    image = request.files.get('image')
    caption = request.form.get('caption')
    comment = request.form.get('comment')

    if not image or not caption or not comment:
        return render_template("index.html", error="Please provide all inputs: image, caption, and comment.")

    # Ensure the 'static' directory exists for saving uploaded images
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Save the uploaded image temporarily
    image_path = os.path.join(static_dir, "uploaded_image.jpg")
    try:
        image.save(image_path)
    except Exception as e:
        return render_template("index.html", error=f"Failed to save the uploaded image: {e}")

    # Perform sentiment analysis
    try:
        results = analyze_multimodal(image_path, caption, comment)
    except Exception as e:
        return render_template("index.html", error=f"Error during sentiment analysis: {e}")

    # Format results for easier display
    formatted_results = {
        "image_sentiment": format_probabilities(results["image_sentiment"]),
        "caption_sentiment": format_probabilities(results["caption_sentiment"]),
        "comment_sentiment": format_probabilities(results["comment_sentiment"]),
        "image_caption_sentiment": format_probabilities(results["image_caption_sentiment"]),
    }

    # Pass results to the result page
    return render_template(
        'result.html',
        results=formatted_results
    )


def format_probabilities(probabilities):
    """
    Format probabilities into a human-readable dictionary.
    Args:
        probabilities (np.ndarray): Probabilities for each sentiment class.
    Returns:
        dict: Formatted probabilities with sentiment labels.
    """
    labels = ['happy', 'sad', 'anger']
    return {label: round(float(prob), 2) for label, prob in zip(labels, probabilities[0])}


if __name__ == "__main__":
    app.run(debug=True)
