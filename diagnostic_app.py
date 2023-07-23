import base64
import json

import streamlit as st
from langchain.agents import AgentExecutor, create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.llms import OpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent


 
MODEL_NAME = "gpt-3.5-turbo-0613"
CSV_FILES = ["data/dataset.csv", "data/symptom_description.csv",
              "data/symptom_precaution.csv", "data/symptom_severity.csv"]

def generate_response(prompt: str, agent: AgentExecutor) -> str:
    try:
        return agent.run(prompt)
    except Exception as e:
        # Handle the exception and display an appropriate message to the user
        st.error(f"Error: {e}")
        return ""


def get_prompt() -> str:
    return st.text_input("You", placeholder="How are you feeling today?",
                         key="user_input", value=st.session_state.user_input)


@st.cache(allow_output_mutation=True)
def initialize_agent(openai_api_key: str) -> AgentExecutor:
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name=MODEL_NAME
    )
    return create_csv_agent(
        ChatOpenAI(temperature=0, model=MODEL_NAME, openai_api_key=openai_api_key),
        CSV_FILES,
        {"encoding": "utf-8", "on_bad_lines": "skip", "index_col": False},
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        memory=ConversationBufferMemory()
    )


@st.cache(allow_output_mutation=True)
def initialize_history() -> ChatMessageHistory:
    return ChatMessageHistory()


def import_history_from_file(history_file, history):
    if not history_file:
        return

    if not st.session_state.get("history_imported", False):
        history_bytes = history_file.getvalue()
        history_messages = messages_from_dict(json.loads(history_bytes))
        for prompt, output in zip(history_messages[::2], history_messages[1::2]):
            st.session_state.past.append(prompt.content)
            st.session_state.generated.append(output.content)
            history.add_user_message(prompt.content)
            history.add_ai_message(output.content)

        st.session_state["history_imported"] = True



def main():
    st.title("Your General Practitioner 👩‍⚕️")
    openai_api_key = st.sidebar.text_input("OpenAI API key")
    
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the right field")
        return

    df_agent = initialize_agent(openai_api_key)

    history = ChatMessageHistory()

    st.session_state.setdefault("generated", [])
    st.session_state.setdefault("past", [])

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    history_file = st.file_uploader("Upload health history")
    import_history_from_file(history_file, history)

    prompt = get_prompt()

    if prompt and len(prompt) > 4:
        output = generate_response(prompt, agent=df_agent)
        if output:
            st.session_state.past.append(prompt)
            st.session_state.generated.append(output)

            history.add_user_message(prompt)
            history.add_ai_message(output)

    if st.session_state["generated"]:
        for i, (gen, past) in enumerate(zip(reversed(st.session_state["generated"]),
                                            reversed(st.session_state["past"]))):
            st.write(past, is_user=True)
            st.write(gen)
            

    dicts = json.dumps(messages_to_dict(history.messages))
    download_icon = '<svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16"><path d="M0 0h16v16H0z" fill="none"/><path d="M9 8V3H7v5H4l4 4 4-4h-3zM4 13v1h8v-1H4z"/></svg>'
    href = f'{download_icon} <a href="data:file/history.json;base64,{base64.b64encode(dicts.encode()).decode()}" download="history.json">Download history</a>'
    st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()