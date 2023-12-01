from dataiku.runnables import Runnable, ResultTable


def create_result_table(bump_results):
    rt = ResultTable()

    rt.add_column("model_path", "Model file path", "STRING")
    rt.add_column("model_origin", "Model origin", "STRING")
    rt.add_column("algorithm", "Algorithm", "STRING")
    rt.add_column("bump_result", "Bump result", "STRING")

    for bump_result in bump_results:
        rt.add_record(bump_result)

    return rt
