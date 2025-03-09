import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_goemotions(file_paths):
    """
    Load and process GoEmotions dataset files
    
    Args:
        file_paths: List of paths to GoEmotions CSV files
        
    Returns:
        Processed DataFrame with text and emotion labels
    """
    dfs = []
    for file_path in file_paths:
        logger.info(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    if len(dfs) > 1:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        combined_df = dfs[0]
    
    logger.info(f"Loaded {len(combined_df)} records from GoEmotions dataset")
    
    # Get list of emotion columns (excluding metadata columns)
    emotion_columns = [col for col in combined_df.columns 
                      if col not in ['text', 'id', 'author', 'subreddit', 
                                    'link_id', 'parent_id', 'created_utc', 
                                    'rater_id', 'example_very_unclear']]
    
    # Convert multi-label format to single dominant emotion
    # For each row, get the emotion with the highest value (1)
    logger.info("Converting multi-label format to single dominant emotion...")
    
    def get_dominant_emotion(row):
        emotions = [col for col in emotion_columns if row[col] == 1]
        if not emotions:
            return 'neutral'  # Default if no emotion is marked
        if len(emotions) == 1:
            return emotions[0]
        # If multiple emotions are marked, pick one with priority
        # Priority order can be customized
        priority = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        for emotion in priority:
            if emotion in emotions:
                return emotion
        return emotions[0]  # If none in priority list, take the first one
    
    # Apply the function to get the dominant emotion for each text
    combined_df['dominant_emotion'] = combined_df.apply(get_dominant_emotion, axis=1)
    
    # Create a new DataFrame with text and emotion
    result_df = pd.DataFrame({
        'text': combined_df['text'],
        'emotion': combined_df['dominant_emotion']
    })
    
    # Remove texts with 'neutral' emotion if desired
    # result_df = result_df[result_df['emotion'] != 'neutral']
    
    logger.info(f"Processed {len(result_df)} records with dominant emotions")
    
    return result_df

def combine_with_annotated_data(goemotions_df, annotated_csv):
    """
    Combine GoEmotions data with our custom annotated data
    
    Args:
        goemotions_df: Processed GoEmotions DataFrame
        annotated_csv: Path to our custom annotated CSV file
        
    Returns:
        Combined DataFrame
    """
    logger.info(f"Loading annotated data from {annotated_csv}...")
    annotated_df = pd.read_csv(annotated_csv)
    
    logger.info(f"Loaded {len(annotated_df)} annotated records")
    
    # Combine datasets
    combined_df = pd.concat([goemotions_df, annotated_df], ignore_index=True)
    
    # Map GoEmotions emotion labels to match our annotated data format if needed
    # This mapping should be adjusted based on your specific emotion labels
    emotion_mapping = {
        'admiration': 'trust',
        'amusement': 'joy',
        'annoyance': 'anger',
        'approval': 'trust',
        'caring': 'trust',
        'confusion': 'surprise',
        'curiosity': 'anticipation',
        'desire': 'anticipation',
        'disappointment': 'sadness',
        'disapproval': 'disgust',
        'embarrassment': 'shame',
        'excitement': 'joy',
        'gratitude': 'trust',
        'grief': 'sadness',
        'love': 'joy',
        'nervousness': 'fear',
        'optimism': 'anticipation',
        'pride': 'joy',
        'realization': 'surprise',
        'relief': 'joy',
        'remorse': 'sadness',
        # Keep others as is: joy, sadness, anger, fear, surprise, disgust, trust, anticipation
    }
    
    # Apply mapping
    combined_df['emotion'] = combined_df['emotion'].apply(
        lambda x: emotion_mapping.get(x, x)
    )
    
    # Get emotion distribution
    emotion_counts = combined_df['emotion'].value_counts()
    logger.info("Emotion distribution in combined dataset:")
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count}")
    
    return combined_df

def balance_dataset(df, max_per_class=None):
    """
    Balance the dataset by downsampling or upsampling
    
    Args:
        df: Input DataFrame
        max_per_class: Maximum samples per emotion class (if None, use size of smallest class)
        
    Returns:
        Balanced DataFrame
    """
    emotion_counts = df['emotion'].value_counts()
    
    if max_per_class is None:
        # Use the smallest class size as maximum
        max_per_class = emotion_counts.min()
    
    logger.info(f"Balancing dataset with {max_per_class} samples per class...")
    
    balanced_dfs = []
    for emotion in emotion_counts.index:
        emotion_df = df[df['emotion'] == emotion]
        
        if len(emotion_df) > max_per_class:
            # Downsample
            emotion_df = emotion_df.sample(max_per_class, random_state=42)
        elif len(emotion_df) < max_per_class:
            # Upsample (duplicate samples)
            emotion_df = emotion_df.sample(max_per_class, replace=True, random_state=42)
        
        balanced_dfs.append(emotion_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    logger.info(f"Balanced dataset created with {len(balanced_df)} total samples")
    
    return balanced_df

def split_and_save_dataset(df, output_dir, test_size=0.1, val_size=0.1):
    """
    Split the dataset into train, validation, and test sets and save to files
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save output files
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a stratified split
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['emotion'], random_state=42
    )
    
    # Further split train into train and validation
    train_df, val_df = train_test_split(
        train_df, test_size=val_size/(1-test_size), 
        stratify=train_df['emotion'], random_state=42
    )
    
    logger.info(f"Split dataset into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")
    
    # Save to CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save the full dataset as well
    df.to_csv(os.path.join(output_dir, 'full_dataset.csv'), index=False)
    
    logger.info(f"Saved processed datasets to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process GoEmotions dataset for sentiment analysis')
    parser.add_argument('--goemotions_files', nargs='+', required=True,
                        help='Paths to GoEmotions CSV files')
    parser.add_argument('--annotated_file', required=True,
                        help='Path to our custom annotated data CSV file')
    parser.add_argument('--output_dir', default='processed_data',
                        help='Directory to save processed datasets')
    parser.add_argument('--max_per_class', type=int, default=None,
                        help='Maximum samples per emotion class for balancing')
    parser.add_argument('--balance', action='store_true',
                        help='Whether to balance the dataset')
    
    args = parser.parse_args()
    
    # Process GoEmotions data
    goemotions_df = load_goemotions(args.goemotions_files)
    
    # Combine with our annotated data
    combined_df = combine_with_annotated_data(goemotions_df, args.annotated_file)
    
    # Balance dataset if requested
    if args.balance:
        combined_df = balance_dataset(combined_df, args.max_per_class)
    
    # Split and save dataset
    split_and_save_dataset(combined_df, args.output_dir)

if __name__ == "__main__":
    main() 