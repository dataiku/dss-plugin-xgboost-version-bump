# -*- coding: utf-8 -*-
import json
from glob import glob
import logging
import os

try:
    from dataiku.doctor.utils import model_io
except Exception as e:
    # TODO 12.2 or higher ?
    raise ImportError("Failed to import dataiku.doctor.utils.model_io. Check that your DSS version is higher than 12.2", e.message)


XGB_PREDICTION_TYPES = ["XGBOOST_CLASSIFICATION", "XGBOOST_REGRESSION"]


DIP_HOME = os.getenv("DIP_HOME")
assert DIP_HOME is not None, "Could not find DSS home environment variable: DIP_HOME"


class MODEL_ORIGIN:
    SAVED_MODEL = "Saved model"
    ANALYSIS = "Analysis"


def bump_saved_model(project_id, saved_model_id):
    """Bump all XGBoost model versions in the saved model

    :param str project_id: Project id containing the saved model
    :param str saved_model_id: Saved model id to bump
    :return List[tuple]: list of (clf_path, model_origin, algorithm) tuples for all bumped xgboost models
    """
    bumped_xgboost_models = []
    saved_model_folder = os.path.join(DIP_HOME, "saved_models", project_id, saved_model_id)
    for rmodeling_params_path in (
        glob("{}/versions/*/rmodeling_params.json".format(saved_model_folder)) +
        glob("{}/pversions/*/*/rmodeling_params.json".format(saved_model_folder))
    ):
        bumped_model = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.SAVED_MODEL)
        if bumped_model is not None:
            bumped_xgboost_models.append(bumped_model)
    
    return bumped_xgboost_models


def bump_analysis_models(project_id, analysis_id):
    """Bump all XGBoost model versions in the analysis

    :param str project_id: Project id containing the analysis
    :param str analysis_id: Analysis id to bump
    :return List[tuple]: list of (clf_path, model_origin, algorithm) tuples for all bumped xgboost models
    """
    bumped_xgboost_models = []
    analysis_folder = os.path.join(DIP_HOME, "analysis-data", project_id, analysis_id)
    for rmodeling_params_path in glob("{}/*/sessions/*/*/*/rmodeling_params.json".format(analysis_folder)):
        bumped_model = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.ANALYSIS)
        if bumped_model is not None:
            bumped_xgboost_models.append(bumped_model)
     
    return bumped_xgboost_models


def bump_all_analyses_and_saved_models(project_id):
    """Bump all XGBoost models in the project

    :param str project_id: Project id containing the analysis
    :return List[tuple]: list of (clf_path, model_origin, algorithm) tuples for all bumped xgboost models
    """
    bumped_xgboost_models = []
    saved_models_folder = os.path.join(DIP_HOME, "saved_models", project_id)
    for rmodeling_params_path in (
        glob("{}/*/versions/*/rmodeling_params.json".format(saved_models_folder)) +
        glob("{}/*/pversions/*/*/rmodeling_params.json".format(saved_models_folder))
    ):
        bumped_model = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.SAVED_MODEL)
        if bumped_model is not None:
            bumped_xgboost_models.append(bumped_model)
 
    analysis_data_folder = os.path.join(DIP_HOME, "analysis-data", project_id)
    for rmodeling_params_path in glob("{}/*/*/sessions/*/*/*/rmodeling_params.json".format(analysis_data_folder)):
        bumped_model = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.ANALYSIS)
        if bumped_model is not None:
            bumped_xgboost_models.append(bumped_model)
     
    return bumped_xgboost_models


def _bump_xgboost_model(model_folder):
    """
    Bump XGBoost model in model folder from 'clf.pkl' format to new backwards compatible format
    with 'xgboost_clf_attributes.json' (Classifier attributes needed for instantiation) and 
    'xgboost_booster.bin' (Booster serialized in backwards compatible format) files.

    :param str model_folder: path to model folder container 
    :return bool: Whether the model was successfully bumped
    """
    logging.info("Bumping XGBoost model in: %s", model_folder)
    try:
        clf = model_io.load_model_from_folder(model_folder)
        model_io.dump_model_to_folder(model_folder, clf)
    except:
        logging.warn("Failed to bump XGBoost model: ", exc_info=True)
        return False
    
    return True


def _possibly_bump_model(rmodeling_params_path, model_origin):
    """
    Check:
    - if model is an XGBoost one
    - if it has an old 'clf.pkl' format file 
    and if so try to bump it into the new backwards compatible format.

    :param str rmodeling_params_path: path to the rmodeling_params.json file of the model
    :param MODEL_ORIGIN model_origin: Whether model is a Saved Model or an Analysis model
    :return tuple | None: (clf_path, model_origin, algorithm) tuple if successfully bumped,
                          None otherwise
    """
    with open(rmodeling_params_path) as f:
        modeling_params = json.load(f)
        algorithm = modeling_params.get("algorithm")

    if algorithm in XGB_PREDICTION_TYPES:
        model_folder = os.path.dirname(rmodeling_params_path)
        clf_path = os.path.join(model_folder, "clf.pkl")
        if os.path.exists(clf_path):
            success = _bump_xgboost_model(model_folder)
            # success = True
            if success:
                return (clf_path, model_origin, algorithm)
    
    return None
