import streamlit as st
import claude

# Page title
st.set_page_config(
    page_title='Bedrock Knowledge Bases with LangChain'
)

# Clear Chat History function
def clear_chat_history():
    claude.history.clear()
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.title('Agentic-RAG')
    st.text('Made with Amazon Bedrock and LangChain 🦜️🔗')
    streaming_on = st.toggle('Streaming')
    st.button('Clear Chat History', on_click=clear_chat_history)
    st.divider()
    st.write("History Logs")
    st.write(claude.history.messages)

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    config = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            for chunk in claude.chain_with_history.stream(
                {"question" : prompt, "history" : claude.history},
                config
            ):
                if 'response' in chunk:
                    full_response += chunk['response']
                    placeholder.markdown(full_response)
                else:
                    full_context = chunk['context']
            placeholder.markdown(full_response)
            # session_state append
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Chain - Invoke
        with st.chat_message("assistant"):
            response = claude.chain_with_history.invoke(
                {"question" : prompt, "history" : claude.history},
                config
            )

            # Handle Tool calling
            if response['tools'].tool_calls:
                st.write(response['response'])
                for tool_call in response['tools'].tool_calls:
                    selected_tool = {
                        "getweather": claude.getWeather,
                        "getaccountname": claude.getAccountName,
                        "getinstances": claude.getInstances,
                        "createinstance": claude.createInstance
                    }[tool_call["name"].lower()]
                    toolResponse = selected_tool.invoke(tool_call)
                    st.markdown(toolResponse.content)
                    st.session_state.messages.append({"role": "assistant", "content": toolResponse.content})

            else:
                st.write(response['response'])
                st.session_state.messages.append({"role": "assistant", "content": response['response']})
