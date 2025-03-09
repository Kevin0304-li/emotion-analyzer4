# SentiMind - Context-Aware Sentiment Analysis Website

This is a modern web interface for the Context-Aware Sentiment Analysis model that allows users to test and visualize the model's capabilities in a user-friendly way.

## Features

- **ChatGPT-like Interface**: Clean, modern UI inspired by ChatGPT and other LLM interfaces
- **Real-time Analysis**: Analyze text and see results instantly
- **Visualization**: View emotion scores with interactive charts and graphs
- **Context Demonstration**: Try different versions of phrases to see how context changes emotional interpretation
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Mode**: Toggle between light and dark themes

## Components

The website consists of:

1. **Flask Backend** (`app.py`): Serves the web interface and handles sentiment analysis requests
2. **HTML Templates** (`templates/`): Main pages for the interface
3. **CSS Styling** (`static/css/`): Modern, responsive styling with dark/light mode support
4. **JavaScript** (`static/js/`): Client-side interaction and visualization
5. **Model Integration**: Connects to the sentiment analysis model (real, demo, or mock)

## Running the Website

### Quick Start (Windows)

Simply double-click `launch_sentimind.bat` to start the website.

### Manual Start

```bash
# Install requirements
pip install -r requirements.txt

# Start the website
python run_website.py
```

Then open your browser and navigate to: `http://localhost:5000`

## Model Options

The website can work with three different model types:

1. **Real Model**: The fully trained context-aware sentiment model (if training has completed)
2. **Demo Model**: A simplified BERT-based model for demonstration purposes
3. **Mock Model**: A simple rule-based model for quick testing when BERT models aren't available

When you first launch the website, if no model is found, you'll be prompted to choose which type to use.

## Testing the Model

Try these examples to see how context changes sentiment interpretation:

- "I will kill you!"
- "I will kill you, haha, just kidding my friend."
- "The sunset was absolutely breathtaking today."
- "I can't believe they would do something so cruel and heartless."

## File Structure

```
├── app.py                 # Flask backend server
├── run_website.py         # Helper script to run the website
├── launch_sentimind.bat   # Windows batch launcher
├── mock_model.py          # Rule-based model for demonstration
├── demo_model.py          # Simplified BERT model
├── templates/
│   ├── index.html         # Main interface page
│   └── about.html         # About page with model explanation
├── static/
│   ├── css/
│   │   └── style.css      # CSS styling
│   ├── js/
│   │   └── script.js      # Client-side JavaScript
│   └── img/
│       └── create_avatar.py  # Script to generate avatar image
```

## Integration with the Sentiment Analysis Model

The website integrates with the sentiment analysis model through the `app.py` file, which:

1. Loads the appropriate model (real, demo, or mock)
2. Tokenizes the input text
3. Runs the sentiment analysis
4. Returns the results to the frontend for display

## Customization

You can customize the website by:

- Modifying the CSS themes in `static/css/style.css`
- Adding new emotion visualization in `static/js/script.js`
- Extending the model explanation in `templates/about.html` 