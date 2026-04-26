import streamlit as st
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

st.set_page_config(page_title="Adaptive Health Insight Agent", page_icon="🩺")

st.title("Adaptive Health Insight Agent")
st.write(
    "Describe how you feel today, or enter your sleep, stress, and activity information. "
    "The agent will identify the key driver and suggest one clear next action."
)

credentials = Credentials(
    api_key=st.secrets["IBM_API_KEY"],
    url=st.secrets["WATSONX_URL"]
)

model = ModelInference(
    model_id="ibm/granite-13b-chat-v2",
    credentials=credentials,
    project_id=st.secrets["PROJECT_ID"],
    params={
        "max_new_tokens": 600,
        "temperature": 0.4
    }
)

user_input = st.text_area(
    "Describe your condition:",
    placeholder="Example: I only slept 4 hours, feel stressed, and didn’t exercise today."
)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please describe your condition first.")
    else:
        prompt = f"""
You are an Adaptive Health Insight Agent.

Your goal is to help users understand how their daily behaviors affect their energy and well-being, and guide them to take the most effective next action.

The user may provide structured input such as sleep hours, stress level, and activity level, or they may describe their feelings in natural language.

User input:
\"\"\"
{user_input}
\"\"\"

Analyze the user's input and respond using this exact structure:

Health Pattern Insight:
- Clearly state what is happening in 1–2 sentences.

Key Driver:
- Primary:
- Secondary, if relevant:

What's happening:
- Explain the situation in simple everyday language. Do not use medical or technical terms.

Action Plan (priority):
1. Highest-impact action:
2. Immediate support action:
3. Supporting action:

What to watch:
- Briefly explain what may happen if this pattern continues.

Next step:
- Give one simple action the user can take right now.

Rules:
- Do not provide medical diagnosis.
- Provide lifestyle and behavioral guidance only.
- Be concise, practical, and action-oriented.
- If information is missing, infer when reasonable and ask only one brief follow-up question.
- Do not write long paragraphs.
"""

        with st.spinner("Analyzing..."):
            result = model.generate_text(prompt=prompt)

        st.subheader("Your Health Insight")
        st.write(result)