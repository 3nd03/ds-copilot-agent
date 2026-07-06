# [Project name]

## Problem
- [1-2 sentences: what business cost does this solve]

## Solution
- [1-2 sentences: what the agent does end to end]

## Architecture
- [diagram: Orchestrator -> Profiler/Modeler/Explainer sub-agents -> MCP server -> tools]
- Orchestrator: decides step order, can skip cleaning
- MCP server: profile_data, clean_data, train_models, explain_model
- Security: Policy Server gates train_models on row count and blocked-tool list

## Course concepts demonstrated
- Multi-agent system (Google ADK): [where, file reference]
- MCP Server: [where, file reference]
- Agent Skills / progressive disclosure: [if applicable, else remove]
- Security: Policy Server structural gating: [file reference]
- Deployability: [discussed, not deployed, why]

## Setup
```
pip install -r requirements.txt
```
- [Google AI Studio API key env var name and where to set it]
- [how to run the MCP server standalone for testing]
- [how to run the Streamlit app]

## Usage
- [upload CSV, enter target column, click run, what you'll see]

## Known limitations
- [ADK version sensitivity, dataset size limits, anything you cut for time]

## Demo dataset
- [Telco Customer Churn or your chosen dataset, link, why it fits Agents for Business]
