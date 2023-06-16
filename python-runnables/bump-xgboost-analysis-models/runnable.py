from result_table_utils import create_result_table
import xgboost_bump

from dataiku.runnables import Runnable


class XGBoostAnalysisModelsBumper(Runnable):

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.analysis_id = config["analysis_id"]

    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current progress
        """
        bumped_models = xgboost_bump.bump_analysis_models(self.project_key, self.analysis_id)
        return create_result_table(bumped_models)
        