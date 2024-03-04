from pymc_experimental.model_builder import ModelBuilder

import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import arviz as az
import json
import mlflow.pyfunc as pyfunc

from typing import List, Union, Dict, Optional

class HeirarchicalModel(ModelBuilder):
  """
  Build a heirarchical model using PyMC
  Takes in the following parameters:
  - group_col: the column defining the group each observation belongs to 
  - feature_cols: the name of feature columns to be used in the model (must be present in dataframe)
  - target_col: the target variable column
  - has_intercept: whether or not to include an intercept in the model
  - features_with_group: whether or not to include group level features
  """

  def __init__(self, group_col: str, feature_cols: List[str], target_col: List[str], has_intercept: bool=True, individual_effects: Optional[List[str]]=None, model_config: Dict[str, Dict]=None, sampler_config: Dict[str, Dict]=None):
    self.group_col = group_col
    self.feature_cols = feature_cols
    self.target_col = target_col
    self.has_intercept = has_intercept
    self.individual_effects = individual_effects
    super().__init__(model_config=model_config, sampler_config=sampler_config)
  
  def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
    # Check the type of X and y and adjust access accordingly
    pass

  @staticmethod
  def get_default_sampler_config() -> Dict:
    """
    Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
    The sampler config dict is used to send parameters to the sampler .
    It will be used during fitting in case the user doesn't provide any sampler_config of their own.
    """
    sampler_config: Dict = {
      "draws": 1_000,
      "tune": 1_000,
      "chains": 4,
      "target_accept": 0.95,
    }
    return sampler_config
  
  @staticmethod
  def get_default_model_config(feature_cols: List[str], individual_effects: Optional[List[str]]) -> Dict:
    """Generates default model priors
    Input:
      feature_cols (list[str]): Features of dataset to be used in the model will recieve a global slope parameter
      individual_effects (list[str]): Features of dataset that will recieve an individual level parameter (should be subset of feature_cols)
    """
    
    default_beta_prior = {'dist': pm.Normal,'kwargs': {'mu': 0, 'sigma': 2}}
    feature_beta_priors = {f'{feature}_beta':  default_beta_prior for feature in feature_cols}
    
