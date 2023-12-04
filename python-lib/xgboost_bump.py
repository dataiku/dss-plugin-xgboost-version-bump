# -*- coding: utf-8 -*-
import datetime
import json
from glob import glob
import logging
import os
import shutil

try:
    from dataiku.doctor.utils import model_io
except Exception as e:
    raise ImportError("Failed to import dataiku.doctor.utils.model_io. Check that your DSS version is higher than 13.0.0", e.message)

try:
    from dataiku.base.folder_context import build_folder_context
except Exception as e:
    raise ImportError("Failed to import dataiku.base.folder_context.build_folder_context. Check that your DSS version is higher than 13.0.0", e.message)

try:
    from dataiku.doctor.prediction.dku_xgboost import XGBOOST_BOOSTER_FILENAME
    from dataiku.doctor.prediction.dku_xgboost import XGBOOST_CLF_ATTRIBUTES_FILENAME
except Exception as e:
    raise ImportError("Failed to import from dataiku.doctor.utils.dku_xgboost. Check that your DSS version is higher than 13.0.0", e.message)


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
    :return List[tuple]: list of (model_path, model_origin, algorithm, bump_result) tuples for all xgboost models
    """
    logging.info("Bumping all XGBoost models in saved model: %s > %s", project_id, saved_model_id)
    all_bump_results = []
    saved_model_folder = os.path.join(DIP_HOME, "saved_models", project_id, saved_model_id)
    for rmodeling_params_path in (
        glob("{}/versions/*/rmodeling_params.json".format(saved_model_folder)) +
        glob("{}/pversions/*/*/rmodeling_params.json".format(saved_model_folder))
    ):
        bump_result = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.SAVED_MODEL)
        if bump_result is not None:
            all_bump_results.append(bump_result)
    
    return all_bump_results


def bump_analysis_models(project_id, analysis_id):
    """Bump all XGBoost model versions in the analysis

    :param str project_id: Project id containing the analysis
    :param str analysis_id: Analysis id to bump
    :return List[tuple]: list of (model_path, model_origin, algorithm, bump_result) tuples for all xgboost models
    """
    logging.info("Bumping all XGBoost models in analysis: %s > %s", project_id, analysis_id)
    all_bump_results = []
    analysis_folder = os.path.join(DIP_HOME, "analysis-data", project_id, analysis_id)
    for rmodeling_params_path in glob("{}/*/sessions/*/*/*/rmodeling_params.json".format(analysis_folder)):
        bump_result = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.ANALYSIS)
        if bump_result is not None:
            all_bump_results.append(bump_result)
     
    return all_bump_results


def bump_all_analyses_and_saved_models(project_id):
    """Bump all XGBoost models in the project

    :param str project_id: Project id containing the analysis
    :return List[tuple]: list of (clf_path, model_origin, algorithm, bump_result) tuples for all xgboost models
    """
    logging.info("Bumping all XGBoost models in project: %s", project_id)
    all_bump_results = []
    saved_models_folder = os.path.join(DIP_HOME, "saved_models", project_id)
    for rmodeling_params_path in (
        glob("{}/*/versions/*/rmodeling_params.json".format(saved_models_folder)) +
        glob("{}/*/pversions/*/*/rmodeling_params.json".format(saved_models_folder))
    ):
        bump_result = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.SAVED_MODEL)
        if bump_result is not None:
            all_bump_results.append(bump_result)
 
    analysis_data_folder = os.path.join(DIP_HOME, "analysis-data", project_id)
    for rmodeling_params_path in glob("{}/*/*/sessions/*/*/*/rmodeling_params.json".format(analysis_data_folder)):
        bump_result = _possibly_bump_model(rmodeling_params_path, MODEL_ORIGIN.ANALYSIS)
        if bump_result is not None:
            all_bump_results.append(bump_result)
     
    return all_bump_results


def _possibly_bump_model(rmodeling_params_path, model_origin):
    """
    Check:
    - if model is an XGBoost one
    - if it has not already been bumped to the new backwards compatible format

    Then bump XGBoost model in model folder from 'clf.pkl' format to new backwards compatible format
    with 'xgboost_clf_attributes.json' (Classifier attributes needed for instantiation) and 
    'xgboost_booster.bin' (Booster serialized in backwards compatible format) files.

    :param str rmodeling_params_path: path to the rmodeling_params.json file of the model
    :param MODEL_ORIGIN model_origin: Whether model is a Saved Model or an Analysis model
    :return tuple | None: (model_path, model_origin, algorithm, bump_result) tuple if it is an
                          XGBoost model, None otherwise
    """
    model_folder = os.path.dirname(rmodeling_params_path)
    with open(rmodeling_params_path) as f:
        modeling_params = json.load(f)
        algorithm = modeling_params.get("algorithm")

    if algorithm not in XGB_PREDICTION_TYPES:
        return None

    # If model has already been bumped, skip
    xgb_clf_attributes_path = os.path.join(model_folder, XGBOOST_CLF_ATTRIBUTES_FILENAME)
    xgb_booster_path = os.path.join(model_folder, XGBOOST_BOOSTER_FILENAME)
    if os.path.exists(xgb_clf_attributes_path) and os.path.exists(xgb_booster_path):
        logging.warn("Bumped Xgboost files already exist: '%s', '%s'", xgb_booster_path, xgb_clf_attributes_path)
        return (xgb_clf_attributes_path, model_origin, algorithm, "SKIPPED")

    # Try bumping model from clf.pkl
    clf_path = os.path.join(model_folder, "clf.pkl")

    # Create a backup copy of the clf file in case something goes wrong
    backup_clf_path = clf_path + ".bak" + str(datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    if os.path.exists(clf_path):
        shutil.copy2(clf_path, backup_clf_path, follow_symlinks=False)

    try:
        folder_context = build_folder_context(model_folder)
        clf = model_io.load_model_from_folder(folder_context)
        model_io.dump_model_to_folder(folder_context, clf)
        logging.info("Successfully bumped XGBoost model: %s", clf_path)
        output = (clf_path, model_origin, algorithm, "SUCCESS")
    except:
        logging.warn("Failed to bump XGBoost model: '%s'", clf_path, exc_info=True)
        output = (clf_path, model_origin, algorithm, "FAIL")
    else:
        # Cleanup backup copy if we have successfully bumped the files
        if os.path.exists(backup_clf_path):
            os.remove(backup_clf_path)

    return output
