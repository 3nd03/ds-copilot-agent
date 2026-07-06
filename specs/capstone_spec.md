# Spec: DS Copilot Agent

## Problem
- [what business problem, who has it, why it costs money/time]

## Goal
- [what the agent must do, one sentence]

## Non-goals
- [explicitly out of scope, e.g. deployment to GCP, hyperparameter tuning]

## Inputs / Outputs
- Input: CSV + target column name
- Output: [profile summary, model scores, feature explanation, in what format]

## Success criteria
- [what "working" means, e.g. runs end to end on a 5k-row churn dataset in under 2 minutes]

## Architecture (one paragraph + diagram reference)
- [orchestrator delegates to 3 sub-agents via MCP tools, see README diagram]

## Risks
- [ADK API instability, SHAP compute time on large datasets, etc.]
