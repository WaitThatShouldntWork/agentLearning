import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from v2.prompts.prompt import agent_prompt


# Load environment variables
load_dotenv(".env")

# Constants
LLM = os.getenv("OPENAI_MODEL")
DOCUMENTS_PATH = "/Users/chrisbooth/Coding/squad/demo_docs/Quality-Management-System-Manual.txt"

# Agent Setup
text = TextLoader(DOCUMENTS_PATH).load()
docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(text)
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name=LLM),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents = True,
)

def run_qa_chain(question):
    """Runs the QA chain and returns results."""
    results = qa_chain({"question": question}, return_only_outputs=False)
    answer = results["answer"]
    sources = results.get("sources")

    # Check if the answer is "I don't know"
    if answer == "I don't know":
        return None, None  # Return None to indicate no valid answer
    
    return answer, sources, results

        
# # Example question Debug from run_qa_chain
# question = "What is the leadership responsbilities of the qms system?"

# # Call the function with the question debug
# answer, sources,results = run_qa_chain(question)

# # Check if an answer was returned debug
# if answer is not None:
#     st.write(f"ANSWER: {answer},\n SOURCES:{sources},\n RESULTS:{results},")
# else:
#     st.write("Sorry, I don't have an answer to that.")

tools = [
    Tool(
        name="QMS QA System",
        func=run_qa_chain,
        description="Answers questions about anything regarding QMS. Input should be a fully formed question.",
        return_direct=True,
    )
]
# Assuming LLM holds a model name like "gpt-3.5-turbo"
llm_instance = ChatOpenAI(model_name=LLM)

# Now, use llm_instance instead of the string LLM
agent = create_react_agent(llm_instance, tools, agent_prompt)

memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Streamlit UI
st.header("Squad AI")

# Initialize Streamlit session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm Athena. Your next-generation AI assistant. How can I help you?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to standardize the response format
# Function to standardize the response format
def standardize_response(response_dict):
    if isinstance(response_dict["output"], str):
        # For simple string responses
        return {
            "text": response_dict["output"],
            "attachments": [],
            "detailed_answer": None
        }
    else:
        # For complex structured responses
        chat_window_answer = response_dict["output"][2]["answer"]
        source_document = response_dict["output"][2]["sources"]
        # Assuming direct attribute access to page_content is possible
        if response_dict["output"][2]["source_documents"]:  # Ensure the list is not empty
            first_doc_content = response_dict["output"][2]["source_documents"][0].page_content
        else:
            first_doc_content = "No document content available."
        
        return {
            "text": chat_window_answer,
            "attachments": [{"type": "document", "url": source_document}],
            "detailed_answer": {
                "sources": [first_doc_content] 
            }
        }

# Accept user input
if prompt := st.chat_input("Enter query here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response using the Conversational agent
    response_dict = agent_executor.invoke({"input": prompt})
    
    # Standardize the response for consistent handling
    standardized_response = standardize_response(response_dict)
    
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": standardized_response["text"]})
    
    # Display assistant response in the main chat window
    with st.chat_message("assistant"):
        st.markdown(standardized_response["text"])
    
    # Handling sidebar information based on the response type
    if standardized_response["detailed_answer"]:
        st.sidebar.markdown(f"**Document:** {standardized_response['attachments'][0]['url']}")
        # Displaying source documents in the sidebar, each document content in markdown format
        for source in standardized_response["detailed_answer"]["sources"]:
            st.sidebar.markdown(f"**Source Document:** {source}")
    else:
        # If no detailed answer is available, display a default message or hide the sidebar content section
        st.sidebar.markdown("**No detailed sources available for this response.**")



# questiondebug1=("Hello")  
# questiondebug2=("What are the leadership responsibilities in regards to the QMS?")
# st.write(agent_executor.invoke({"input": questiondebug1}))
# st.write(agent_executor.invoke({"input": questiondebug2}))