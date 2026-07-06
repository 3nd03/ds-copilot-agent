# Decisions Log

- ydata-profiling: no wheel for Python 3.14 yet, pip install failed. Installed rest of requirements.txt without it. profile_data() already wraps the import in try/except and sets report_path to "skipped (...)", so this is a soft dependency, not a blocker.
- ADK import test (google.adk.agents.LlmAgent, MCPToolset): passed on first try, no fix needed.
- orchestrator_fallback.run_pipeline: was gating clean_data on profile["needs_cleaning"], but clean_data also does categorical encoding, which train_models always needs (this dataset has no missing/dupes but plenty of string columns). Changed run_pipeline to always call clean_data before training.
- train_models: `y.dtype == object` check never matched because pandas 3.0 loads text columns as its new native "str" dtype, not "object", so the target ("Yes"/"No") never got LabelEncoded and the model trained/predicted on raw strings, which later broke SHAP's numba masking. Changed the check to `not pd.api.types.is_numeric_dtype(y)`.
