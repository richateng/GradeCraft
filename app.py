import streamlit as st
import asyncio
from pathlib import Path
from src.config.settings import settings
from src.core.orchestrator import run_gradecraft_swarm_pipeline
import os

st.set_page_config(page_title="GradeCraft - Async Swarm", page_icon="📝", layout="wide")
st.title("GradeCraft 📝")
st.markdown("Automated answer sheet grading powered by an Async Multi-Agent VLM Swarm.")

# Sidebar Configuration
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", value=settings.groq_api_key or "", type="password")

st.header("Upload Documents")
solution_pdf = st.file_uploader("Upload Solution Key (PDF)", type="pdf", key="sol")
student_pdf = st.file_uploader("Upload Student Answer Booklet (PDF)", type="pdf", key="stu")

if st.button("Evaluate Student (Swarm Pipeline)"):
    if not groq_api_key:
        st.error("Please provide the Groq API Key.")
    elif not solution_pdf or not student_pdf:
        st.error("Please upload both the solution key and the student booklet.")
    else:
        # Update environment explicitly for this run
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        try:
            with st.spinner("Executing Swarm Pipeline (Parsing -> Vision -> Math -> Evaluate)..."):
                # Streamlit is synchronous, we use asyncio.run to execute our async pipeline
                final_results = asyncio.run(
                    run_gradecraft_swarm_pipeline(solution_pdf.getvalue(), student_pdf.getvalue())
                )
                
            st.success("Evaluation Complete!")
            
            # Rendering final results
            evaluations = final_results.get("evaluations", [])
            total_score = 0
            total_max = 0
            
            for index, res in enumerate(evaluations):
                q = res.get("question", f"Q{index+1}")
                score = res.get("points_awarded", 0)
                m_score = res.get("max_points", 0)
                total_score += score
                total_max += m_score
                
                with st.expander(f"Question {q} - Score: {score}/{m_score}"):
                    st.write("**Rationale:**", res.get("deduction_rationale"))
                    
            st.subheader(f"Total Score: {total_score} / {total_max}")
            
        except Exception as e:
            st.error(f"Error during evaluation: {e}")
