from .stages import (
    evaluate_stage,
    optimize_stage,
    prepare_data,
    preprocess_data,
    train_stage,
)

STAGES = {
    "prepare" : prepare_data,
    "preprocess" : preprocess_data,
    "train" : train_stage,
    "optimize" : optimize_stage,
    "evaluate" : evaluate_stage,
}

def run_pipeline (stage : str = "all") -> None:
    if stage == "all":
        for stage_name, stage_function in STAGES.items():
            print(f"Running stage: {stage_name}")
            stage_function()

        return

    STAGES[stage]()
