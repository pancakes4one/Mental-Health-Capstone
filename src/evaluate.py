import mlflow

EXPERIMENT_NAME = "MentalHealth_Capstone"


def get_best_run(metric="metrics.f1_score"):
    """Search all runs in the experiment and return the one with the best metric."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric} DESC"]
    )

    best_run = runs_df.iloc[0]
    return best_run


if __name__ == "__main__":
    best_run = get_best_run()

    print("--- Best Run ---")
    print(f"Run name:  {best_run['tags.mlflow.runName']}")
    print(f"Run ID:    {best_run['run_id']}")
    print(f"Accuracy:  {best_run['metrics.accuracy']:.4f}")
    print(f"Precision: {best_run['metrics.precision']:.4f}")
    print(f"Recall:    {best_run['metrics.recall']:.4f}")
    print(f"F1 Score:  {best_run['metrics.f1_score']:.4f}")
    print(f"AUC-ROC:   {best_run['metrics.auc_roc']:.4f}")
