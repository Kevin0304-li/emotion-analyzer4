# Context-Aware Sentiment Analysis Model

This project implements a context-aware sentiment analysis model that considers three key aspects of text:
1. **Semantic content** - The literal meaning of the words and phrases
2. **Context** - The broader situational context in which the text appears
3. **Reaction** - How the model itself responds to the text

The model is designed to recognize that the same text (e.g., "I will kill you") could have very different emotional meanings depending on the relationship between the speaker and listener (e.g., between friends as a joke vs. between strangers as a threat).

## Project Structure

- `sentiment_analysis_model.py` - Main model implementation using BERT with context-aware components
- `process_goemotions.py` - Utility for processing the GoEmotions dataset and combining it with our annotated data
- `predict.py` - Script for using the trained model to make predictions on new text
- `app.py` - Flask web application for the UI interface
- `demo_model.py` - Simplified demo model
- `mock_model.py` - Mock model for testing
- `requirements.txt` - Python dependencies required for the project

## Dataset

**Note:** The dataset files are not included in this repository due to their size. You will need to:

1. Download the GoEmotions dataset and place it in a `full_dataset` directory
2. Create a small annotated dataset in `annotated_data.csv` 
3. Run the processing script to prepare the data:
   ```
   python process_goemotions.py --goemotions_files full_dataset/goemotions_1.csv full_dataset/goemotions_2.csv full_dataset/goemotions_3.csv --annotated_file annotated_data.csv --output_dir processed_data --balance
   ```

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Process and prepare the datasets as described above

3. Train the model:
   ```
   python sentiment_analysis_model.py
   ```

4. Make predictions:
   ```
   python predict.py --model_dir model_output --input_file test_texts.txt --output_file results.txt
   ```

5. Run the web interface:
   ```
   python run_website.py
   ```

## Web Interface

The project includes a modern web interface similar to ChatGPT for interacting with the model:

1. To launch the web interface on Windows, use the `launch_sentimind.bat` file
2. The interface allows real-time sentiment analysis with visualizations
3. See `README_WEBSITE.md` for more details about the web interface

## Acknowledgements

This project uses the GoEmotions dataset along with custom annotated data to create a context-aware sentiment analysis model.

## Model Architecture

The model uses a transformer-based architecture (BERT) with three specialized components:

1. **Semantic Layer** - Extracts semantic understanding from the text through BERT embeddings
2. **Context Layer** - Processes sequence outputs to understand contextual nuances
3. **Reaction Layer** - Models potential reactions to the text

These components are integrated through an integration layer that combines all features before the final classification.

## Training Techniques

- **Cross-validation** - K-fold validation to ensure the model works well on unseen data
- **Regularization** - Weight decay and dropout to prevent overfitting
- **Learning Rate Scheduling** - Linear schedule with warmup for optimal training

## Making Predictions

Once trained, you can use the model in interactive mode:
```
python predict.py --model_dir model_output --interactive
```

Or analyze text from a file:
```
python predict.py --model_dir model_output --input_file texts_to_analyze.txt --output_file results.txt
```

## Example

For the example "I will kill you", the model considers:
- The semantic content of the words themselves
- The potential context (through its context-aware training)
- How a human might react to such a statement in different contexts

This enables the model to distinguish between a friendly joke and a genuine threat based on surrounding context cues.

## Performance

The model's performance is evaluated using:
- Accuracy
- F1 score (weighted)
- Cross-validation results across multiple folds

Training results are visualized and saved as `training_results.png` during the training process.

## Contributing

To enhance this model:
1. Add more contextually-annotated data examples
2. Experiment with different transformer architectures (RoBERTa, ALBERT, etc.)
3. Implement more sophisticated context modeling techniques 