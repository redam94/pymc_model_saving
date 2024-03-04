import logging
import os
import pickle
from typing import Any, Dict, Optional


import numpy as np
import pandas as pd
import pymc as pm
import yaml

import mlflow_wrapper
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "pymc"

PYMC_PREDICT = "predict"

SUPPORTED_PYMC_PREDICT_METHODS = [
    PYMC_PREDICT,
]

SERIALIZATION_FORMAT_NC = "nc"

SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_NC,
]

_logger = logging.getLogger(__name__)
__all__ = ["save_model", "OlsModelWrapper"]

def get_default_pip_requirements(include_cloudpickle=False):
    """Create list of default pip requirements for MLflow Models.

    Returns
    -------
    list of default pip requirements for MLflow Models produced by this flavor.
    Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
    that, at a minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("pymc")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """Return default Conda environment for MLflow Models.

    Returns
    -------
    The default Conda environment for MLflow Models produced by calls to
    :func:`save_model()` and :func:`log_model()`
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


def save_model(
    pymc_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_NC,
):
    """Save a ``pymc`` model to a path on the local file system.

    Parameters
    ----------
    pymc_model :
        Fitted ``pymc`` model object.
    path : str
        Local path where the model is to be saved.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    mlflow_model: mlflow.models.Model, optional (default=None)
        mlflow.models.Model configuration to which to add the python_function flavor.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)

        .. Warning:: if performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a ``pymc`` model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though.
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pymc", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                f"Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                f"{SUPPORTED_SERIALIZATION_FORMATS}."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.nc"
    model_data_path = os.path.join(path, model_data_subpath)
    _save_model(pymc_model, model_data_path, serialization_format=serialization_format)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="flavor",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        pymc_version=pm.__version__,
        serialization_format=serialization_format,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            include_cloudpickle = False # serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
            default_reqs = get_default_pip_requirements(include_cloudpickle)
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))

def _save_model(model, path, **kwargs):
    model.save(path)


def log_model(
    sktime_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_NC,
    **kwargs,
):
    """
    Log a ``sktime`` model as an MLflow artifact for the current run.

    Parameters
    ----------
    sktime_model : fitted ``sktime`` model
        Fitted ``sktime`` model object.
    artifact_path : str
        Run-relative artifact path to save the model to.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    registered_model_name : str, optional (default=None)
        If given, create a model version under ``registered_model_name``, also creating
        a registered model if one with the given name does not exist.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models import infer_signature

          train = df.drop_column("target_label")
          predictions = ...  # compute model predictions
          signature = infer_signature(train, predictions)

        .. Warning:: if performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a ``sktime`` model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though.
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    await_registration_for : int, optional (default=None)
        Number of seconds to wait for the model version to finish being created and is
        in ``READY`` status. By default, the function waits for five minutes. Specify 0
        or None to skip waiting.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["sktime", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    kwargs:
        Additional arguments for :py:class:`mlflow.models.model.Model`

    Returns
    -------
    A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
    metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_wrapper,
        registered_model_name=registered_model_name,
        sktime_model=sktime_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        serialization_format=serialization_format,
        **kwargs,
    )



