# FLAG: ADK had breaking changes at 2.0. Verify these import paths against
# `pip show google-adk` version and adk-docs before running. If MCPToolset's
# import path has moved, the fallback is orchestrator_fallback.py in this folder,
# which calls the same MCP tools directly without ADK.

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

MODEL = "gemini-3-flash-preview"

mcp_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=["mcp_server/data_tools_server.py"],
    )
)

profiler_agent = LlmAgent(
    name="ProfilerAgent",
    model=MODEL,
    description="Profiles a raw CSV and decides whether cleaning is needed.",
    instruction="""Call profile_data on the given csv_path and target_column.
    Report rows, missing values, duplicates, and target balance in plain terms.
    State clearly whether needs_cleaning is true or false, this decides the next step.""",
    tools=[mcp_tools],
    output_key="profile_result",
)

modeler_agent = LlmAgent(
    name="ModelerAgent",
    model=MODEL,
    description="Cleans data if needed, then trains and selects the best model.",
    instruction="""Given the profile result in state, if needs_cleaning is true call
    clean_data first and use the cleaned_csv_path for the next step, otherwise use the
    original csv_path. Then call train_models. Report task_type, both model scores, and
    which model won and why.""",
    tools=[mcp_tools],
    output_key="model_result",
)

explainer_agent = LlmAgent(
    name="ExplainerAgent",
    model=MODEL,
    description="Explains the trained model's predictions with SHAP.",
    instruction="""Take the model_path from the model result in state and call
    explain_model. Summarise the top 5 features and what they mean for the business
    problem, in plain language a non-technical stakeholder could read.""",
    tools=[mcp_tools],
    output_key="explanation_result",
)

root_agent = LlmAgent(
    name="DSCopilotOrchestrator",
    model=MODEL,
    description="Coordinates profiling, modeling, and explanation for a business dataset.",
    instruction="""You are a data science orchestrator. Given a csv_path and target_column:
    1. Delegate to ProfilerAgent first, always.
    2. Delegate to ModelerAgent next, using the profiling outcome to decide if cleaning ran.
    3. Delegate to ExplainerAgent last.
    Do not skip steps out of order. Summarise the full pipeline result for the user at the end,
    written for a business stakeholder, not an engineer.""",
    sub_agents=[profiler_agent, modeler_agent, explainer_agent],
)
