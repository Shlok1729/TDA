from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

class BaselineModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
class TDAModel(BaselineModel):
    def __init__(self, is_real_world=False):
        super().__init__()
        if is_real_world:
            self.model = XGBRegressor(
                n_estimators=200,    # More estimators for complex real patterns
                learning_rate=0.1,   # Faster learning to catch BTC-style trends
                max_depth=6,     
                reg_alpha=0.1,       # Lower L1 to keep more features active
                reg_lambda=0.1,      # Lower L2 for higher sensitivity
                random_state=42
            )
        else:
            # Optimal hyperparameters for synthetic sinusoidal structural noise
            self.model = XGBRegressor(
                n_estimators=100, 
                learning_rate=0.05,  # Lower learning rate
                max_depth=4,         # Shallower trees
                min_child_weight=3,  # Prevent very specific, tiny leaf nodes
                gamma=0.1,           # Minimum loss reduction for a split
                subsample=0.8,       # Bagging to prevent overfitting
                colsample_bytree=0.8,# Bagging features
                reg_alpha=0.5,       # L1 Regularization to drop useless TDA features
                reg_lambda=1.5,      # L2 Regularization 
                random_state=42
            )
        
def evaluate_model(y_true, y_pred):
    return {
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
