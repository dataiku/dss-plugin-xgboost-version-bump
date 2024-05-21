from result_table_utils import create_result_table
import xgboost_upgrade

from dataiku.runnables import ResultTable
from dataiku.runnables import Runnable


class XGBoostProjectModelsUpgrader(Runnable):

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key

    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current progress
        """
        upgraded_models = xgboost_upgrade.upgrade_all_analyses_and_saved_models(self.project_key)
        return create_result_table(upgraded_models)
        