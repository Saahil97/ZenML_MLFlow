from zenml.steps import BaseParameters

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model_name: str = "LinearRegression"
    
