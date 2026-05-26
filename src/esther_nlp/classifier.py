"""Classifier module containing PyTorch MLP and training pipeline."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold

# Pinned NLP Task Settings
MASK_TOKEN = "[MASK]"
TARGET_TOKENS: Dict[str, str] = {
    "המן": "Haman", "והמן": "Haman", "להמן": "Haman", "האגגי": "Haman",
    "מרדכי": "Mordecai", "ומרדכי": "Mordecai", "למרדכי": "Mordecai",
    "במרדכי": "Mordecai", "היהודי": "Mordecai", "יהודי": "Mordecai",
}
CLASS_TO_ID: Dict[str, int] = {"Haman": 0, "Mordecai": 1}
ID_TO_CLASS: Dict[int, str] = {0: "Haman", 1: "Mordecai"}


class ZereshMLP(nn.Module):
    """Tiny Multi-Layer Perceptron (MLP) for classification."""

    def __init__(self, d_in: int, d_hid: int = 4, n_cls: int = 2):
        """Initializes model layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_hid, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        return self.net(x)


def build_classifier_examples(
    all_verses: List[str],
    tokenizer
) -> List[Tuple[List[str], int, str, int]]:
    """Builds masked dataset examples from verses.

    Filters for verses that mention exactly one target character class, masks
    the target tokens, and records labels.

    Args:
        all_verses: Flat list of verse strings.
        tokenizer: Trained BpeTokenizer wrapper.

    Returns:
        List of tuples: (masked_tokens, label_id, raw_verse, original_verse_index)
    """
    examples = []
    for idx, verse in enumerate(all_verses):
        tokens = tokenizer.encode_to_tokens(verse)
        triggers = {t: TARGET_TOKENS[t] for t in tokens if t in TARGET_TOKENS}
        classes = set(triggers.values())
        
        # Verify verse refers to exactly one character (Haman or Mordecai), not both or neither
        if len(classes) != 1:
            continue
            
        label = next(iter(classes))
        masked_tokens = [MASK_TOKEN if t in triggers else t for t in tokens]
        examples.append((masked_tokens, CLASS_TO_ID[label], verse, idx))
        
    return examples


def featurize_sentence(
    masked_tokens: List[str],
    w2v_model: Word2Vec,
    dim: int = 50
) -> np.ndarray:
    """Aggregates word vectors of non-masked tokens to represent a sentence.

    Extracts: mean, max, and min vector poolings + non-masked token count.
    Feature dimension = 3 * dim + 1 (default 151 dimensions).

    Args:
        masked_tokens: List of string tokens, potentially containing '[MASK]'.
        w2v_model: Trained Word2Vec model.
        dim: Embedding dimension size.

    Returns:
        1D numpy array of features.
    """
    vectors = [
        w2v_model.wv[t] 
        for t in masked_tokens 
        if t != MASK_TOKEN and t in w2v_model.wv
    ]
    
    if not vectors:
        return np.zeros(3 * dim + 1, dtype=np.float32)
        
    arr = np.stack(vectors)
    return np.concatenate([
        arr.mean(axis=0),
        arr.max(axis=0),
        arr.min(axis=0),
        np.array([len(vectors)], dtype=np.float32)
    ]).astype(np.float32)


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 0.1
) -> Tuple[float, List[float], np.ndarray, np.ndarray]:
    """Runs Stratified 5-Fold Cross-Validation with strict seed pinning.

    Args:
        X: Feature matrix of size (N, 151).
        y: Ground-truth label array of size (N,).
        n_splits: Number of cross-validation folds.
        random_state: Pinned seed for split shuffling.
        epochs: Number of neural network training epochs per fold.
        lr: Learning rate.
        weight_decay: L2 penalty.

    Returns:
        A tuple containing:
        - Mean accuracy across folds.
        - List of individual fold accuracies.
        - Array of out-of-fold validation predictions (N,).
        - Array of softmax confidence probabilities for the predicted class (N,).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_preds = np.full_like(y, -1)
    all_confs = np.zeros(len(y), dtype=np.float32)
    fold_accuracies = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Normalize features strictly within fold to prevent leakage
        mu = X[train_idx].mean(axis=0)
        std = X[train_idx].std(axis=0) + 1e-8
        X_train = (X[train_idx] - mu) / std
        X_test  = (X[test_idx] - mu) / std
        
        # Pinned random seed per fold weights initialization
        torch.manual_seed(random_state + fold_idx)
        model = ZereshMLP(d_in=X.shape[1], d_hid=4, n_cls=2)
        
        # Compute balanced class weights for training cross entropy
        cw = torch.tensor([
            len(y[train_idx]) / (2 * (y[train_idx] == c).sum()) 
            for c in range(2)
        ], dtype=torch.float32)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=cw)
        
        # Convert arrays to PyTorch Tensors
        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(y[train_idx], dtype=torch.long)
        
        # Training loop
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t), y_tr_t)
            loss.backward()
            optimizer.step()
            
        # Validation inference
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
            preds = probs.argmax(axis=1)
            
        fold_acc = float((preds == y[test_idx]).mean())
        fold_accuracies.append(fold_acc)
        
        all_preds[test_idx] = preds
        all_confs[test_idx] = probs[np.arange(len(preds)), preds]
        
    mean_accuracy = float(np.mean(fold_accuracies))
    return mean_accuracy, fold_accuracies, all_preds, all_confs
