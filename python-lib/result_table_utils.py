from dataiku.runnables import Runnable, ResultTable


def create_result_table(bumped_xgboost_models):
    rt = ResultTable()

    rt.add_column("clf_path", "Model file path", "STRING")
    rt.add_column("model_origin", "Model origin", "STRING")
    rt.add_column("algorithm", "Algorithm", "STRING")

    for model_info in bumped_xgboost_models:
        rt.add_record(model_info)

    return rt
