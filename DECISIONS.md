# Decisions Log

- ydata-profiling: no wheel for Python 3.14 yet, pip install failed. Installed rest of requirements.txt without it. profile_data() already wraps the import in try/except and sets report_path to "skipped (...)", so this is a soft dependency, not a blocker.
- ADK import test (google.adk.agents.LlmAgent, MCPToolset): passed on first try, no fix needed.
