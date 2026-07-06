"""Structural + semantic gating in front of the MCP tools.
Structural: deterministic rules from policies.yaml, no LLM, fast.
Semantic: an LLM call checking intent for the one genuinely risky action (training on
unvetted data). Mirrors the Day 4/5 Policy Server pattern: separate execution from governance.
"""
import yaml
from pathlib import Path
from google import genai

POLICIES_PATH = Path(__file__).parent / "policies.yaml"


class PolicyService:
    def __init__(self, env: str = "local", project_id: str | None = None, location: str = "us-central1"):
        self.config = yaml.safe_load(POLICIES_PATH.read_text())
        self.env = env
        self.project_id = project_id
        self.location = location

    def is_tool_allowed(self, tool_name: str) -> bool:
        env_config = self.config.get("environments", {}).get(self.env, {})
        return tool_name not in env_config.get("blocked_tools", [])

    def check_row_count(self, tool_name: str, row_count: int) -> tuple[bool, str]:
        # Structural gate specific to this project: refuse to train on too little data,
        # a random forest on 20 rows produces a confidently wrong model.
        min_rows = self.config.get("min_rows_for_training", 50)
        if tool_name == "train_models" and row_count < min_rows:
            return False, f"Refused: {row_count} rows is below the {min_rows}-row training floor."
        return True, ""

    async def check_action_semantic(self, action_description: str) -> tuple[bool, str]:
        if not self.project_id:
            return True, "semantic check skipped, no project_id configured"
        client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        prompt = (
            "Evaluate if this data science action risks exposing PII or violates basic data "
            f"handling policy. Reply VIOLATION or OK, one word, then a one-line reason.\n{action_description}"
        )
        response = client.models.generate_content(model="gemini-3-pro", contents=prompt)
        verdict = response.text.strip()
        return not verdict.upper().startswith("VIOLATION"), verdict

    def gate(self, tool_name: str, arguments: dict, row_count: int | None = None) -> tuple[bool, str]:
        if not self.is_tool_allowed(tool_name):
            return False, f"Refused: {tool_name} is blocked in env '{self.env}'."
        if row_count is not None:
            ok, reason = self.check_row_count(tool_name, row_count)
            if not ok:
                return False, reason
        return True, "allowed"
