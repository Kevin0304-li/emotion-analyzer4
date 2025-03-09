import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define constants
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
NUM_FOLDS = 5  # For cross-validation

class ContextAwareEmotionDataset(Dataset):
    """
    Custom dataset for our context-aware emotion analysis model.
    This dataset will prepare data considering text semantics, context, and potential reactions.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text with special tokens
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ContextAwareSentimentModel(nn.Module):
    """
    A BERT-based model with additional components for context, 
    semantic understanding, and reaction awareness.
    """
    def __init__(self, num_classes, dropout_prob=0.1):
        super(ContextAwareSentimentModel, self).__init__()
        
        # BERT base model for semantic understanding
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # Semantic understanding component
        self.semantic_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Context awareness component
        self.context_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Reaction modeling component
        self.reaction_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Integration layer that combines all components
        self.integration_layer = nn.Linear(hidden_size * 3 // 2, hidden_size)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Regularization: Weight decay will be applied during optimizer setup
        
    def forward(self, input_ids, attention_mask):
        # Pass input through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the [CLS] token representation (semantic understanding)
        cls_output = outputs.last_hidden_state[:, 0, :]
        sequence_output = outputs.last_hidden_state
        
        # Process through the semantic layer
        semantic_features = self.semantic_layer(cls_output)
        
        # Process through the context layer (using attention mechanism)
        # Here we use a simple approach by taking the mean of sequence outputs
        # for context awareness
        context_features = self.context_layer(torch.mean(sequence_output, dim=1))
        
        # Process through the reaction layer
        # Here we model potential reactions from the same sequence output
        # In a more advanced implementation, we could include specific reaction data
        reaction_features = self.reaction_layer(cls_output)
        
        # Concatenate all features
        combined_features = torch.cat([semantic_features, context_features, reaction_features], dim=1)
        
        # Integration
        integrated_features = self.integration_layer(combined_features)
        integrated_features = torch.relu(integrated_features)
        
        # Classification
        logits = self.classifier(integrated_features)
        
        return logits

def prepare_data(annotated_csv, goemotions_csv_list):
    """
    Prepare and combine the annotated data and GoEmotions datasets
    """
    logger.info("Loading and preparing data...")
    
    # Load annotated data
    annotated_df = pd.read_csv(annotated_csv)
    
    # Map emotions to indices
    unique_emotions = annotated_df['emotion'].unique()
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
    
    # Convert emotions to indices
    annotated_labels = [emotion_to_idx[emotion] for emotion in annotated_df['emotion']]
    
    # Create lists for texts and labels
    texts = annotated_df['text'].tolist()
    labels = annotated_labels
    
    # Load GoEmotions data if needed for additional training
    # This would be implemented based on the structure of the GoEmotions data
    # For now, just using the annotated data
    
    logger.info(f"Prepared {len(texts)} samples with {len(unique_emotions)} emotion classes")
    return texts, labels, emotion_to_idx, idx_to_emotion

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs):
    """
    Train and evaluate the model
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_f1s = []
    
    # Train the model
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        # Calculate average loss over the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validate the model
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1s.append(val_f1)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: ")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Val F1 Score: {val_f1:.4f}")
    
    return train_losses, val_losses, val_f1s

def cross_validation(texts, labels, tokenizer, model_class, num_classes, device, n_splits=5):
    """
    Perform k-fold cross-validation
    """
    logger.info(f"Performing {n_splits}-fold cross-validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        logger.info(f"Training fold {fold+1}/{n_splits}")
        
        # Split data for this fold
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Create datasets
        train_dataset = ContextAwareEmotionDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        
        val_dataset = ContextAwareEmotionDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE
        )
        
        # Initialize model
        model = model_class(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize optimizer with weight decay for regularization
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # Total training steps
        total_steps = len(train_dataloader) * EPOCHS
        
        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,  # 10% warmup
            num_training_steps=total_steps
        )
        
        # Train and evaluate
        train_losses, val_losses, val_f1s = train_and_evaluate(
            model, train_dataloader, val_dataloader, optimizer, scheduler, device, EPOCHS
        )
        
        fold_results.append({
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1s': val_f1s,
            'final_f1': val_f1s[-1]
        })
        
    # Find best model
    best_model_idx = np.argmax([result['final_f1'] for result in fold_results])
    best_model = fold_results[best_model_idx]['model']
    
    logger.info(f"Best model found in fold {best_model_idx+1} with F1 score: {fold_results[best_model_idx]['final_f1']:.4f}")
    
    return best_model, fold_results

def plot_training_results(fold_results):
    """
    Plot training and validation losses and F1 scores
    """
    plt.figure(figsize=(12, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_losses'], label=f'Fold {i+1} Train')
        plt.plot(result['val_losses'], label=f'Fold {i+1} Val')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation F1 score
    plt.subplot(2, 1, 2)
    for i, result in enumerate(fold_results):
        plt.plot(result['val_f1s'], label=f'Fold {i+1}')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def save_model(model, tokenizer, emotion_to_idx, idx_to_emotion, save_dir='model_output'):
    """
    Save the trained model and related artifacts
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save emotion mappings
    pd.DataFrame({
        'emotion': list(emotion_to_idx.keys()),
        'index': list(emotion_to_idx.values())
    }).to_csv(os.path.join(save_dir, 'emotion_mappings.csv'), index=False)
    
    logger.info(f"Model and artifacts saved to {save_dir}")

def main():
    """
    Main function to run the training pipeline
    """
    logger.info("Starting the context-aware sentiment analysis model training")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare data
    texts, labels, emotion_to_idx, idx_to_emotion = prepare_data(
        annotated_csv='annotated_data.csv',
        goemotions_csv_list=[
            'full_dataset/goemotions_1.csv',
            'full_dataset/goemotions_2.csv',
            'full_dataset/goemotions_3.csv'
        ]
    )
    
    # Cross-validation
    best_model, fold_results = cross_validation(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        model_class=ContextAwareSentimentModel,
        num_classes=len(emotion_to_idx),
        device=device,
        n_splits=NUM_FOLDS
    )
    
    # Plot training results
    plot_training_results(fold_results)
    
    # Save the best model
    save_model(
        model=best_model,
        tokenizer=tokenizer,
        emotion_to_idx=emotion_to_idx,
        idx_to_emotion=idx_to_emotion
    )
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 