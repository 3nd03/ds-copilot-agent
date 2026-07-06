"""MCP server exposing EDA / cleaning / training / explainability as tools.
Run standalone: python data_tools_server.py
Consumed by ADK agents via MCPToolset (see agents/agent.py).
"""
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import shap

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

server = Server("ds-copilot")


def _load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _is_classification(y: pd.Series) -> bool:
    return y.dtype == object or y.nunique() <= 10


def profile_data(csv_path: str, target_column: str) -> dict:
    """Lightweight profiling summary the orchestrator reasons over.
    Full ydata-profiling HTML report is also saved for the writeup/demo.
    """
    df = _load(csv_path)
    missing = df.isna().mean().round(3).to_dict()
    dupes = int(df.duplicated().sum())
    target_balance = df[target_column].value_counts(normalize=True).round(3).to_dict()

    try:
        from ydata_profiling import ProfileReport
        ProfileReport(df, title="EDA Report", minimal=True).to_file(ARTIFACTS / "eda_report.html")
        report_path = str(ARTIFACTS / "eda_report.html")
    except Exception as e:
        report_path = f"skipped ({e})"

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_pct_by_column": missing,
        "duplicate_rows": dupes,
        "target_balance": target_balance,
        "needs_cleaning": dupes > 0 or any(v > 0 for v in missing.values()),
        "full_report_path": report_path,
    }


def clean_data(csv_path: str, target_column: str) -> dict:
    """Drops duplicates, imputes missing values, encodes categoricals. Saves cleaned CSV."""
    df = _load(csv_path)
    before = len(df)
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if col != target_column:
            df[col] = LabelEncoder().fit_transform(df[col])

    out_path = ARTIFACTS / "cleaned.csv"
    df.to_csv(out_path, index=False)
    return {
        "rows_dropped": before - len(df),
        "cleaned_csv_path": str(out_path),
        "columns_encoded": list(df.select_dtypes(include="int64").columns),
    }


def train_models(csv_path: str, target_column: str) -> dict:
    """Trains a baseline and a random forest, picks the better one by held-out score."""
    df = _load(csv_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    classification = _is_classification(y)

    if classification and not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if classification:
        baseline, rf = LogisticRegression(max_iter=1000), RandomForestClassifier(random_state=42)
        score_fn, score_name = lambda m: f1_score(y_test, m.predict(X_test), average="weighted"), "f1_weighted"
    else:
        baseline, rf = LinearRegression(), RandomForestRegressor(random_state=42)
        score_fn, score_name = lambda m: r2_score(y_test, m.predict(X_test)), "r2"

    results = {}
    for name, model in [("baseline", baseline), ("random_forest", rf)]:
        model.fit(X_train, y_train)
        results[name] = round(score_fn(model), 4)

    best_name = max(results, key=results.get)
    best_model = rf if best_name == "random_forest" else baseline
    model_path = ARTIFACTS / "best_model.joblib"
    joblib.dump({"model": best_model, "features": list(X.columns), "classification": classification}, model_path)

    return {
        "task_type": "classification" if classification else "regression",
        "scores": {k: {score_name: v} for k, v in results.items()},
        "best_model": best_name,
        "model_path": str(model_path),
    }


def explain_model(model_path: str, csv_path: str, target_column: str) -> dict:
    """SHAP feature importance for the trained model. Saves a summary plot."""
    bundle = joblib.load(model_path)
    model, features = bundle["model"], bundle["features"]
    df = _load(csv_path)
    X = df[features].sample(min(200, len(df)), random_state=42)

    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X, show=False)
    plot_path = ARTIFACTS / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    importance = pd.Series(abs(shap_values.values).mean(axis=0), index=features).sort_values(ascending=False)
    return {
        "top_features": importance.head(5).round(4).to_dict(),
        "plot_path": str(plot_path),
    }


TOOLS = {
    "profile_data": profile_data,
    "clean_data": clean_data,
    "train_models": train_models,
    "explain_model": explain_model,
}

TOOL_SCHEMAS = [
    Tool(
        name="profile_data",
        description="Profile a CSV: missing values, duplicates, target balance. Run this first, always.",
        inputSchema={"type": "object", "properties": {
            "csv_path": {"type": "string"}, "target_column": {"type": "string"}},
            "required": ["csv_path", "target_column"]},
    ),
    Tool(
        name="clean_data",
        description="Clean a CSV: dedupe, impute, encode categoricals. Only needed if profiling flags issues.",
        inputSchema={"type": "object", "properties": {
            "csv_path": {"type": "string"}, "target_column": {"type": "string"}},
            "required": ["csv_path", "target_column"]},
    ),
    Tool(
        name="train_models",
        description="Train baseline + random forest, return scores and best model path.",
        inputSchema={"type": "object", "properties": {
            "csv_path": {"type": "string"}, "target_column": {"type": "string"}},
            "required": ["csv_path", "target_column"]},
    ),
    Tool(
        name="explain_model",
        description="SHAP explainability for a trained model. Run only after train_models.",
        inputSchema={"type": "object", "properties": {
            "model_path": {"type": "string"}, "csv_path": {"type": "string"}, "target_column": {"type": "string"}},
            "required": ["model_path", "csv_path", "target_column"]},
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOL_SCHEMAS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in TOOLS:
        return [TextContent(type="text", text=f"Error: unknown tool {name}")]
    try:
        result = TOOLS[name](**arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read, write):
        init_options = server.create_initialization_options()
        await server.run(read, write, init_options)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
