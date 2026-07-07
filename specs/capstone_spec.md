# Spec: DS Copilot Agent

## Problem
Businesses with customer data want to know who is about to leave and why, but profiling, cleaning, modeling, and explaining a dataset properly takes a data scientist hours each time.

## Goal
Given a CSV and a target column, decide whether cleaning is needed, train and compare two models, and explain the winner's predictions with SHAP.

## Non-goals
- Deployment to a live public endpoint
- Hyperparameter tuning beyond default sklearn settings
- Regression targets, only classification is tested

## Inputs / Outputs
- Input: CSV file, target column name
- Output: profiling summary, model scores for both candidates, top SHAP features, shown in the Streamlit app

## Success criteria
Runs end to end on the Telco Customer Churn dataset (7,043 rows) without error, correctly skips or runs cleaning based on the data, produces a trained model and a SHAP explanation that matches domain intuition.

## Architecture
Root orchestrator delegates to three sub-agents (profiling, modeling, explaining), each calling tools on an MCP server. See README for the full diagram.

## Risks
- ADK's API changed at 2.0, breaking the message format and connection class this code depends on
- Gemini's free tier caps at 20 requests/day, limiting live testing of the ADK path
- The orchestrator's step order is LLM-decided, not guaranteed to repeat exactly the same way twice