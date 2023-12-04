from result_table_utils import create_result_table
import xgboost_bump

import dataiku
from dataiku.runnables import ResultTable
from dataiku.runnables import Runnable


class XGBoostInstanceModelsBumper(Runnable):

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        client = dataiku.api_client()
        self.project_keys = client.list_project_keys()


    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current progress
        """
        bumped_models = [ 
            bumped_model
            for project_key in self.project_keys
            for bumped_model in xgboost_bump.bump_all_analyses_and_saved_models(project_key)
        ]
        return create_result_table(bumped_models)
