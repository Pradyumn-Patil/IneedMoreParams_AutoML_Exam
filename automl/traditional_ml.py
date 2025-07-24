"""
Traditional Machine Learning Models for Text Classification

Implements various traditional ML approaches:
1. Support Vector Machines (SVM) with multiple kernels
2. Tree-based methods (XGBoost, LightGBM, Random Forest, CatBoost)
3. Naive Bayes variants
4. Gradient Boosting
5. Feature selection methods
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class TraditionalMLFactory:
    """Factory class for creating traditional ML models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Create a traditional ML model based on type."""
        if model_type == 'svm_linear':
            return SVMClassifier(kernel='linear', **kwargs)
        elif model_type == 'svm_rbf':
            return SVMClassifier(kernel='rbf', **kwargs)
        elif model_type == 'svm_poly':
            return SVMClassifier(kernel='poly', **kwargs)
        elif model_type == 'xgboost':
            return XGBoostClassifier(**kwargs)
        elif model_type == 'lightgbm':
            return LightGBMClassifier(**kwargs)
        elif model_type == 'catboost':
            return CatBoostClassifier(**kwargs)
        elif model_type == 'random_forest':
            return RandomForestWrapper(**kwargs)
        elif model_type == 'extra_trees':
            return ExtraTreesWrapper(**kwargs)
        elif model_type == 'gradient_boosting':
            return GradientBoostingWrapper(**kwargs)
        elif model_type == 'naive_bayes':
            return NaiveBayesClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class BaseTraditionalML:
    """Base class for traditional ML models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_selector = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_selection: Optional[Dict] = None,
            scale_features: bool = False):
        """Fit the model with optional preprocessing."""
        # Feature scaling
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            
        # Feature selection
        if feature_selection:
            self.feature_selector = self._create_feature_selector(feature_selection)
            X = self.feature_selector.fit_transform(X, y)
            logger.info(f"Selected {X.shape[1]} features from {X.shape[1]} original features")
            
        # Fit model
        self._fit_model(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        # Apply same preprocessing
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
            
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        # Apply same preprocessing
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision function
            decision = self.model.decision_function(X)
            if len(decision.shape) == 1:
                # Binary classification
                proba = np.vstack([1 - decision, decision]).T
            else:
                # Multi-class
                proba = self._softmax(decision)
            return proba
            
    def _softmax(self, x):
        """Compute softmax values."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
        
    def _create_feature_selector(self, config: Dict):
        """Create feature selector based on configuration."""
        method = config.get('method', 'chi2')
        k = config.get('k', 5000)
        
        if method == 'chi2':
            return SelectKBest(chi2, k=k)
        elif method == 'mutual_info':
            return SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            return SelectKBest(f_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit the specific model - to be implemented by subclasses."""
        raise NotImplementedError


class SVMClassifier(BaseTraditionalML):
    """Support Vector Machine classifier with multiple kernels."""
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 probability: bool = True,
                 max_iter: int = 10000,
                 random_state: int = 42):
        super().__init__(random_state)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight
        self.probability = probability
        self.max_iter = max_iter
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit SVM model."""
        if self.kernel == 'linear':
            # Use LinearSVC for efficiency with linear kernel
            self.model = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            self.model.fit(X, y)
            # Wrap with CalibratedClassifierCV for probability estimates
            if self.probability:
                self.model = CalibratedClassifierCV(self.model, cv=3)
                self.model.fit(X, y)
        else:
            # Use SVC for non-linear kernels
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                class_weight=self.class_weight,
                probability=self.probability,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            self.model.fit(X, y)


class XGBoostClassifier(BaseTraditionalML):
    """XGBoost classifier optimized for text classification."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 min_child_weight: int = 1,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 scale_pos_weight: Optional[float] = None,
                 random_state: int = 42,
                 n_jobs: int = -1):
        super().__init__(random_state)
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss'
        }
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit XGBoost model."""
        # Determine number of classes
        n_classes = len(np.unique(y))
        if n_classes == 2:
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = 'logloss'
            
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)


class LightGBMClassifier(BaseTraditionalML):
    """LightGBM classifier optimized for text classification."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 num_leaves: int = 31,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 min_child_samples: int = 20,
                 reg_alpha: float = 0,
                 reg_lambda: float = 0,
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1):
        super().__init__(random_state)
        self.params = {
            'n_estimators': n_estimators,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbosity': -1
        }
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit LightGBM model."""
        n_classes = len(np.unique(y))
        if n_classes == 2:
            self.params['objective'] = 'binary'
            self.params['metric'] = 'binary_logloss'
            
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)


class CatBoostClassifier(BaseTraditionalML):
    """CatBoost classifier optimized for text classification."""
    
    def __init__(self,
                 iterations: int = 100,
                 depth: int = 6,
                 learning_rate: float = 0.1,
                 l2_leaf_reg: float = 3.0,
                 border_count: int = 32,
                 class_weights: Optional[Union[str, Dict]] = 'balanced',
                 random_state: int = 42):
        super().__init__(random_state)
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
            
        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'border_count': border_count,
            'class_weights': class_weights,
            'random_state': random_state,
            'verbose': False
        }
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit CatBoost model."""
        self.model = cb.CatBoostClassifier(**self.params)
        self.model.fit(X, y)


class RandomForestWrapper(BaseTraditionalML):
    """Random Forest classifier wrapper."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, float] = 'sqrt',
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1):
        super().__init__(random_state)
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit Random Forest model."""
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)


class ExtraTreesWrapper(BaseTraditionalML):
    """Extra Trees classifier wrapper."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, float] = 'sqrt',
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1):
        super().__init__(random_state)
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit Extra Trees model."""
        self.model = ExtraTreesClassifier(**self.params)
        self.model.fit(X, y)


class GradientBoostingWrapper(BaseTraditionalML):
    """Gradient Boosting classifier wrapper."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 random_state: int = 42):
        super().__init__(random_state)
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'random_state': random_state
        }
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit Gradient Boosting model."""
        self.model = GradientBoostingClassifier(**self.params)
        self.model.fit(X, y)


class NaiveBayesClassifier(BaseTraditionalML):
    """Naive Bayes classifier with multiple variants."""
    
    def __init__(self,
                 variant: str = 'multinomial',
                 alpha: float = 1.0,
                 fit_prior: bool = True,
                 class_prior: Optional[np.ndarray] = None):
        super().__init__()
        self.variant = variant
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit Naive Bayes model."""
        if self.variant == 'multinomial':
            self.model = MultinomialNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior
            )
        elif self.variant == 'complement':
            self.model = ComplementNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior
            )
        elif self.variant == 'bernoulli':
            self.model = BernoulliNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior
            )
        else:
            raise ValueError(f"Unknown Naive Bayes variant: {self.variant}")
            
        # Ensure non-negative values for Multinomial/Complement NB
        if self.variant in ['multinomial', 'complement']:
            X = np.maximum(X, 0)
            
        self.model.fit(X, y)


def get_traditional_ml_search_space(model_type: str) -> Dict:
    """Get hyperparameter search space for traditional ML models."""
    if model_type == 'svm_linear':
        return {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }
    elif model_type == 'svm_rbf':
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        }
    elif model_type == 'xgboost':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }
    elif model_type == 'lightgbm':
        return {
            'n_estimators': [50, 100, 200],
            'num_leaves': [20, 31, 50],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'feature_fraction': [0.7, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.9, 1.0],
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    else:
        return {}