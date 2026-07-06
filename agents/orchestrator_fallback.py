# Use only if agent.py's ADK imports break and there's no time left to fix them.
# Same decision logic as the ADK orchestrator, just without the framework.
# You lose the ADK scoring row but keep a working, explainable pipeline.

import sys
sys.path.insert(0, "../mcp_server")
from data_tools_server import profile_data, clean_data, train_models, explain_model


def run_pipeline(csv_path: str, target_column: str) -> dict:
    profile = profile_data(csv_path, target_column)

    working_csv = csv_path
    cleaning_report = None
    if profile["needs_cleaning"]:
        cleaning_report = clean_data(csv_path, target_column)
        working_csv = cleaning_report["cleaned_csv_path"]

    training = train_models(working_csv, target_column)
    explanation = explain_model(training["model_path"], working_csv, target_column)

    return {
        "profile": profile,
        "cleaning": cleaning_report,
        "training": training,
        "explanation": explanation,
    }


if __name__ == "__main__":
    import json
    result = run_pipeline("your_data.csv", "target_column_name")
    print(json.dumps(result, indent=2))
