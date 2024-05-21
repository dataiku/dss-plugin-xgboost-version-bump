from dataiku.runnables import Runnable, ResultTable


def create_result_table(upgrade_results):
    rt = ResultTable()

    rt.add_column("model_path", "Model file path", "STRING")
    rt.add_column("model_origin", "Model origin", "STRING")
    rt.add_column("algorithm", "Algorithm", "STRING")
    rt.add_column("upgrade_result", "Upgrade result", "STRING")

    for upgrade_result in upgrade_results:
        rt.add_record(upgrade_result)

    return rt
