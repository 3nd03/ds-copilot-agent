import asyncio
import sys
import streamlit as st

sys.path.insert(0, "..")
sys.path.insert(0, "../agents")

st.set_page_config(page_title="DS Copilot Agent", layout="wide")
st.title("Data Science Copilot Agent")
st.caption("Upload a CSV, pick a target column. The orchestrator profiles, cleans if needed, trains, and explains.")

uploaded = st.file_uploader("Dataset (CSV)", type="csv")
target_column = st.text_input("Target column name")
run_clicked = st.button("Run agent pipeline", disabled=not (uploaded and target_column))

if run_clicked:
    csv_path = "artifacts/upload.csv"
    import os
    os.makedirs("artifacts", exist_ok=True)
    with open(csv_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Running orchestrator..."):
        try:
            from agent import root_agent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService

            async def run_adk():
                session_service = InMemorySessionService()
                runner = Runner(agent=root_agent, app_name="ds_copilot", session_service=session_service)
                session = await session_service.create_session(app_name="ds_copilot", user_id="oscar")
                events = []
                async for event in runner.run_async(
                    user_id="oscar", session_id=session.id,
                    new_message=f"csv_path={csv_path}, target_column={target_column}",
                ):
                    events.append(event)
                return events[-1]

            result = asyncio.run(run_adk())
            st.success("Ran via ADK orchestrator.")
            st.write(result)

        except Exception as e:
            st.warning(f"ADK path failed ({e}), falling back to direct pipeline.")
            from orchestrator_fallback import run_pipeline
            result = run_pipeline(csv_path, target_column)

            st.subheader("Profile")
            st.json(result["profile"])
            if result["cleaning"]:
                st.subheader("Cleaning")
                st.json(result["cleaning"])
            st.subheader("Model Training")
            st.json(result["training"])
            st.subheader("Explainability")
            st.json(result["explanation"]["top_features"])
            st.image(result["explanation"]["plot_path"])
