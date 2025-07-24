"""
Ensemble Methods for Text Classification AutoML

Implements various ensemble strategies to combine multiple models for better performance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class VotingEnsemble:
    """Voting ensemble that combines predictions from multiple models."""
    
    def __init__(self, voting='soft', weights=None):
        """
        Initialize voting ensemble.
        
        Args:
            voting: 'hard' for majority vote, 'soft' for probability averaging
            weights: Optional weights for each model
        """
        self.voting = voting
        self.weights = weights
        self.models = []
        self.model_names = []
        self.is_fitted = False
        
    def add_model(self, model, name: str):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.model_names.append(name)
        
    def predict(self, X) -> np.ndarray:
        """Make predictions using voting ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        if self.voting == 'hard':
            return self._hard_voting_predict(X)
        else:
            return self._soft_voting_predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using soft voting."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # For models without predict_proba, use decision function or predictions
                pred = self._convert_to_proba(model, X)
            predictions.append(pred)
        
        # Weighted average of probabilities
        if self.weights is not None:
            weighted_preds = np.average(predictions, axis=0, weights=self.weights)
        else:
            weighted_preds = np.mean(predictions, axis=0)
            
        return weighted_preds
    
    def _hard_voting_predict(self, X) -> np.ndarray:
        """Hard voting prediction."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Apply weights if provided
        if self.weights is not None:
            # For hard voting with weights, convert to soft voting
            logger.warning("Hard voting with weights converted to soft voting")
            return self._soft_voting_predict(X)
        
        # Majority vote
        from scipy.stats import mode
        majority_vote, _ = mode(predictions, axis=0, keepdims=False)
        return majority_vote
    
    def _soft_voting_predict(self, X) -> np.ndarray:
        """Soft voting prediction."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def _convert_to_proba(self, model, X) -> np.ndarray:
        """Convert model predictions to probabilities."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        elif hasattr(model, 'decision_function'):
            # Convert decision function to probabilities
            decision = model.decision_function(X)
            if decision.ndim == 1:
                # Binary classification
                proba = np.vstack([1 - decision, decision]).T
            else:
                # Multi-class classification
                proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            return proba
        else:
            # Use predictions as hard probabilities
            pred = model.predict(X)
            n_classes = len(np.unique(pred))
            proba = np.zeros((len(pred), n_classes))
            proba[np.arange(len(pred)), pred] = 1.0
            return proba
    
    def fit_ensemble(self):
        """Mark ensemble as fitted."""
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")
        self.is_fitted = True
        logger.info(f"Voting ensemble fitted with {len(self.models)} models: {self.model_names}")


class StackingEnsemble:
    """Stacking ensemble that uses a meta-learner to combine model predictions."""
    
    def __init__(self, meta_learner=None, cv_folds=5):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_learner: Meta-learner model (default: LogisticRegression)
            cv_folds: Number of cross-validation folds for meta-features
        """
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)
        self.cv_folds = cv_folds
        self.base_models = []
        self.model_names = []
        self.is_fitted = False
        
    def add_base_model(self, model, name: str):
        """Add a base model to the stacking ensemble."""
        self.base_models.append(model)
        self.model_names.append(name)
        
    def fit(self, X, y):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Training labels
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models added to ensemble")
            
        # Generate meta-features using cross-validation
        logger.info(f"Generating meta-features using {self.cv_folds}-fold CV")
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-learner
        logger.info("Training meta-learner")
        self.meta_learner.fit(meta_features, y)
        
        # Retrain base models on full dataset
        logger.info("Retraining base models on full dataset")
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {self.model_names[i]}")
            model.fit(X, y)
            
        self.is_fitted = True
        logger.info(f"Stacking ensemble fitted with {len(self.base_models)} base models")
        
    def predict(self, X) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        # Generate meta-features from base models
        meta_features = np.column_stack([
            self._get_base_predictions(model, X) for model in self.base_models
        ])
        
        # Use meta-learner to make final prediction
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        # Generate meta-features from base models
        meta_features = np.column_stack([
            self._get_base_predictions(model, X) for model in self.base_models
        ])
        
        # Use meta-learner to predict probabilities
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(meta_features)
        else:
            # Convert predictions to probabilities
            pred = self.meta_learner.predict(meta_features)
            n_classes = len(np.unique(pred))
            proba = np.zeros((len(pred), n_classes))
            proba[np.arange(len(pred)), pred] = 1.0
            return proba
    
    def _generate_meta_features(self, X, y) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold = X[train_idx] if hasattr(X, '__getitem__') else X.iloc[train_idx]
            y_train_fold = y[train_idx] if hasattr(y, '__getitem__') else y.iloc[train_idx]
            X_val_fold = X[val_idx] if hasattr(X, '__getitem__') else X.iloc[val_idx]
            
            for i, model in enumerate(self.base_models):
                # Create a copy of the model for this fold
                model_copy = self._clone_model(model)
                
                if model_copy is None:
                    # If we can't clone the model, use the original (not ideal but works)
                    logger.warning(f"Using original model for fold {fold + 1}, model {i}")
                    predictions = self._get_base_predictions(model, X_val_fold)
                else:
                    # Train on fold training data
                    model_copy.fit(X_train_fold, y_train_fold)
                    
                    # Predict on fold validation data
                    predictions = self._get_base_predictions(model_copy, X_val_fold)
                
                meta_features[val_idx, i] = predictions
                
        return meta_features
    
    def _get_base_predictions(self, model, X) -> np.ndarray:
        """Get predictions from a base model."""
        if hasattr(model, 'predict_proba'):
            # Use probabilities of positive class for binary, max prob for multi-class
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]  # Positive class probability
            else:
                return np.max(proba, axis=1)  # Max probability
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(X)
            if decision.ndim == 1:
                return decision
            else:
                return np.max(decision, axis=1)
        else:
            # Use raw predictions
            return model.predict(X).astype(float)
    
    def _clone_model(self, model):
        """Create a copy of the model."""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For TextAutoML models, we can't clone them easily
            # Instead, skip stacking for such complex models
            logger.warning("Cannot clone TextAutoML model for stacking. Skipping this fold.")
            return None


class WeightedAveragingEnsemble:
    """Weighted averaging ensemble with automatic weight optimization."""
    
    def __init__(self, weight_method='accuracy', optimize_weights=True):
        """
        Initialize weighted averaging ensemble.
        
        Args:
            weight_method: Method to compute weights ('accuracy', 'loss', 'uniform')
            optimize_weights: Whether to optimize weights using validation data
        """
        self.weight_method = weight_method
        self.optimize_weights = optimize_weights
        self.models = []
        self.model_names = []
        self.weights = None
        self.is_fitted = False
        
    def add_model(self, model, name: str, validation_score: float = None):
        """
        Add a model to the ensemble.
        
        Args:
            model: The trained model
            name: Model name
            validation_score: Validation score for weight calculation
        """
        self.models.append({
            'model': model,
            'name': name,
            'validation_score': validation_score
        })
        self.model_names.append(name)
        
    def fit(self, X_val=None, y_val=None):
        """
        Fit the ensemble and compute optimal weights.
        
        Args:
            X_val: Validation features for weight optimization
            y_val: Validation labels for weight optimization
        """
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")
            
        if self.optimize_weights and X_val is not None and y_val is not None:
            self.weights = self._optimize_weights(X_val, y_val)
        else:
            self.weights = self._compute_simple_weights()
            
        self.is_fitted = True
        logger.info(f"Weighted ensemble fitted with weights: {dict(zip(self.model_names, self.weights))}")
        
    def predict(self, X) -> np.ndarray:
        """Make predictions using weighted averaging."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using weighted averaging."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = []
        for model_dict in self.models:
            model = model_dict['model']
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = self._convert_to_proba(model, X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred
    
    def _compute_simple_weights(self) -> np.ndarray:
        """Compute simple weights based on validation scores."""
        if self.weight_method == 'uniform':
            return np.ones(len(self.models)) / len(self.models)
        
        scores = []
        for model_dict in self.models:
            score = model_dict.get('validation_score', 1.0)
            if score is None:
                score = 1.0
            scores.append(score)
        
        scores = np.array(scores)
        
        if self.weight_method == 'accuracy':
            # Higher accuracy = higher weight
            weights = scores / np.sum(scores)
        elif self.weight_method == 'loss':
            # Lower loss = higher weight
            weights = (1.0 / (scores + 1e-8))
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(self.models)) / len(self.models)
            
        return weights
    
    def _optimize_weights(self, X_val, y_val) -> np.ndarray:
        """Optimize weights using validation data."""
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            
            # Get predictions from all models
            predictions = []
            for model_dict in self.models:
                model = model_dict['model']
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)
                else:
                    pred = self._convert_to_proba(model, X_val)
                predictions.append(pred)
            
            # Weighted average
            weighted_pred = np.average(predictions, axis=0, weights=weights)
            
            # Calculate loss (negative log-likelihood)
            epsilon = 1e-15
            weighted_pred = np.clip(weighted_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.log(weighted_pred[np.arange(len(y_val)), y_val]))
            
            return loss
        
        # Initialize with uniform weights
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            logger.info(f"Weight optimization successful. Final loss: {result.fun:.4f}")
            return optimal_weights
        else:
            logger.warning("Weight optimization failed. Using simple weights.")
            return self._compute_simple_weights()
    
    def _convert_to_proba(self, model, X) -> np.ndarray:
        """Convert model predictions to probabilities."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(X)
            if decision.ndim == 1:
                # Binary classification
                proba = np.vstack([1 - decision, decision]).T
            else:
                # Multi-class: softmax
                proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            return proba
        else:
            # Use predictions as hard probabilities
            pred = model.predict(X)
            n_classes = len(np.unique(pred))
            proba = np.zeros((len(pred), n_classes))
            proba[np.arange(len(pred)), pred] = 1.0
            return proba


class EnsembleBuilder:
    """Builder class to create and manage different ensemble types."""
    
    def __init__(self):
        self.models = []
        self.validation_scores = []
        self.model_names = []
        
    def add_model(self, model, name: str, validation_score: float = None):
        """Add a model to the ensemble builder."""
        self.models.append(model)
        self.model_names.append(name)
        self.validation_scores.append(validation_score)
        
    def build_voting_ensemble(self, voting='soft', weights=None) -> VotingEnsemble:
        """Build a voting ensemble."""
        ensemble = VotingEnsemble(voting=voting, weights=weights)
        for model, name in zip(self.models, self.model_names):
            ensemble.add_model(model, name)
        ensemble.fit_ensemble()
        return ensemble
    
    def build_stacking_ensemble(self, meta_learner=None, cv_folds=5) -> StackingEnsemble:
        """Build a stacking ensemble."""
        ensemble = StackingEnsemble(meta_learner=meta_learner, cv_folds=cv_folds)
        for model, name in zip(self.models, self.model_names):
            ensemble.add_base_model(model, name)
        return ensemble
    
    def build_weighted_ensemble(self, weight_method='accuracy', 
                              optimize_weights=True) -> WeightedAveragingEnsemble:
        """Build a weighted averaging ensemble."""
        ensemble = WeightedAveragingEnsemble(
            weight_method=weight_method, 
            optimize_weights=optimize_weights
        )
        for model, name, score in zip(self.models, self.model_names, self.validation_scores):
            ensemble.add_model(model, name, score)
        return ensemble
    
    def auto_select_best_ensemble(self, X_val, y_val) -> Tuple[Any, str, float]:
        """
        Automatically select the best ensemble method based on validation performance.
        
        Returns:
            Tuple of (best_ensemble, ensemble_name, validation_score)
        """
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
            
        ensemble_candidates = [
            (self.build_voting_ensemble(voting='soft'), 'voting_soft'),
            (self.build_voting_ensemble(voting='hard'), 'voting_hard'),
            (self.build_weighted_ensemble(weight_method='accuracy'), 'weighted_accuracy'),
            (self.build_weighted_ensemble(weight_method='uniform'), 'weighted_uniform'),
        ]
        
        # Skip stacking for TextAutoML models (too complex to clone properly)
        # if len(X_val) >= 100:  # Minimum data for reliable stacking
        #     stacking = self.build_stacking_ensemble()
        #     stacking.fit(X_val, y_val)
        #     ensemble_candidates.append((stacking, 'stacking'))
        
        best_ensemble = None
        best_name = None
        best_score = -1
        
        logger.info("Evaluating ensemble methods...")
        
        for ensemble, name in ensemble_candidates:
            try:
                if 'voting' in name:
                    # Voting ensembles don't need fitting with validation data
                    pass
                elif name != 'stacking':  # Stacking already fitted
                    ensemble.fit(X_val, y_val)
                
                predictions = ensemble.predict(X_val)
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, predictions)
                
                logger.info(f"{name}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_ensemble = ensemble
                    best_name = name
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                continue
                
        if best_ensemble is None:
            raise RuntimeError("All ensemble methods failed")
            
        logger.info(f"Best ensemble: {best_name} (score: {best_score:.4f})")
        return best_ensemble, best_name, best_score