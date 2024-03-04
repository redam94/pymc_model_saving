from pymc_experimental.model_builder import ModelBuilder

import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import arviz as az
import json
import mlflow.pyfunc as pyfunc

from typing import List, Union, Dict



class OLSModel(ModelBuilder):
  '''
  Build a simple OLS model using PyMC
  '''
  _model_type = "OLS"

  version = "0.0.1"

  def __init__(self, endog_label: str, exog_label: Union[str, List[str]], has_intercept: bool = True, **kwargs):
    super().__init__(**kwargs)
    self.endog_label = endog_label
    self.exog_label = exog_label if isinstance(exog_label, list) else [exog_label]
    self.has_intercept = has_intercept

  def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
   # Check the type of X and y and adjust access accordingly
    try:
      X_values = X[self.exog_label].values
    except KeyError as e:

      raise KeyError(
        f"""Column {e} not found in the input data. 
        Please insure {', '.join(self.exog_label)} are present 
        in the input data."""
      )
    
    
    y_values = y.values if isinstance(y, pd.Series) else y
    
    
    self._generate_and_preprocess_model_data(X_values, y_values)
    with pm.Model(coords=self.model_coords, coords_mutable=self.mutable_coords) as self.model:
      # Create mutable data containers
      x_data = pm.MutableData("x_data", self.X, dims=("index", "exog"))
      y_data = pm.MutableData("y_data", self.y, dims="index")

      # prior parameters
      if self.has_intercept:
        intercept_mu_prior = self.model_config.get("intercept_mu_prior", 0.0)
        intercept_sigma_prior = self.model_config.get("intercept_sigma_prior", 1.0)
      
      b_mu_prior = self.model_config.get("b_mu_prior", 0.0)
      b_sigma_prior = self.model_config.get("b_sigma_prior", 1.0)
      eps_prior = self.model_config.get("eps_prior", 1.0)

      # priors
      
      b = pm.Normal("b", mu=b_mu_prior, sigma=b_sigma_prior, dims="exog")
      eps = pm.HalfNormal("eps", eps_prior)
      if self.has_intercept:
        intercept = pm.Normal("intercept", mu=intercept_mu_prior, sigma=intercept_sigma_prior)
        obs = pm.Normal(self.endog_label, mu=intercept + pm.math.dot(x_data, b), sigma=eps, shape=x_data.shape[0], observed=y_data, dims="index")
      else:
        obs = pm.Normal(self.endog_label, mu=pm.math.dot(x_data, b), sigma=eps, shape=x_data.shape[0], observed=y_data, dims="index")

  def _data_setter(self, X: pd.DataFrame, y: pd.Series|None = None, **kwargs) -> None:
    

    with self.model:
      if isinstance(X, pd.DataFrame):
        X = X[self.exog_label].values
      
      pm.set_data({"x_data": X}, coords={"index": np.arange(X.shape[0])})
      
        
      if y is not None:
        pm.set_data({"y_data": y.values})
      else:
        pm.set_data({"y_data": np.zeros(X.shape[0])})
      

  @staticmethod
  def get_default_model_config() -> Dict:
    """
    Returns a class default config dict for model builder if no model_config is provided on class initialization.
    The model config dict is generally used to specify the prior values we want to build the model with.
    It supports more complex data structures like lists, dictionaries, etc.
    It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
    """
    print("called model config")
    model_config: Dict = {
      "intercept_mu_prior": 0.0,
      "intercept_sigma_prior": 1.0,
      "b_mu_prior": 0.0,
      "b_sigma_prior": 1.0,
      "eps_prior": 1.0,
    }
    return model_config
  

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
  
  @property
  def output_var(self):
    return self.endog_label
  
  @property
  def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
    return self.model_config
  
  def get_params(self, deep=True):
    return super().get_params(deep) | {'endog_label': self.endog_label, 'exog_label': self.exog_label, 'has_intercept': self.has_intercept}
  
  def _save_input_params(self, idata) -> None:
    """
    Saves any additional model parameters (other than the dataset) to the idata object.

    These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
    If you don't need to store any extra parameters, you can leave this method unimplemented.

    Example:
      For saving customer IDs provided as an 'customer_ids' input to the model:
      self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
      idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
    """

    idata.attrs['endog'] = json.dumps(self.endog_label)
    idata.attrs['exog'] = json.dumps(self.exog_label)
    idata.attrs['has_intercept'] = json.dumps(self.has_intercept)

  def _generate_and_preprocess_model_data(self, X, y) -> None:
    """
    Depending on the model, we might need to preprocess the data before fitting the model.
    all required preprocessing and conditional assignments should be defined here.
    """
    self.model_coords = {'exog': self.exog_label}  # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
    self.mutable_coords = {'index': np.arange(X.shape[0])}# as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
    # and usually we would need to do some preprocessing, or generate the coords from the data.
    self.X = X
    self.y = y

  def plot_trace(self):
    n = 3 if self.has_intercept else 2
    fig, ax = plt.subplots(n, 2, figsize=(10, 3*n))
    az.plot_trace(self.idata.posterior, axes=ax)
    return fig
    

  def summary(self):
    return az.summary(self.idata.posterior)
  
  def model_graph(self):
    return pm.model_to_graphviz(self.model)
      
  @classmethod
  def load(cls, fname: str):
    """ """

    filepath = Path(str(fname))
    idata = az.from_netcdf(filepath)
    model_config = cls._model_config_formatting(
      json.loads(idata.attrs["model_config"])
    )
    model = cls(
            json.loads(idata.attrs["endog"]),
            json.loads(idata.attrs["exog"]),
            has_intercept=json.loads(idata.attrs["has_intercept"]),
        )
    model.idata = idata
    dataset = idata.fit_data.to_dataframe()
    X = dataset.drop(columns=[model.output_var])
    y = dataset[model.output_var].values
    model.build_model(X, y)
        # All previously used data is in idata.
    if model.id != idata.attrs["id"]:
      raise ValueError(
        f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
      )

    return model
    

class OLSModelWrapper(pyfunc.PythonModel):
    def load_context(self, context):
        self.model: OLSModel = OLSModel.load(context.artifacts['model_path'])
      
    
    def predict(self, context, model_input):
        self.model: OLSModel = OLSModel.load(context.artifacts['model_path'])
        
        return self.model.predict(model_input)
    
    def summary(self, context, model_input):
        self.model: OLSModel = OLSModel.load(context.artifacts['model_path'])
        
        return self.model.summary()
    