from result_table_utils import create_result_table
import xgboost_upgrade

from dataiku.runnables import Runnable


class XGBoostSavedModelUpgrader(Runnable):

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.saved_model_id = config["saved_model_id"]


    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current progress
        """
        upgraded_models = xgboost_upgrade.upgrade_saved_model(self.project_key, self.saved_model_id)
        return create_result_table(upgraded_models)
