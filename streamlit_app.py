# imports for langchain, streamlit
from langchain_openai import ChatOpenAI
from langchain.schema import(
    HumanMessage,
    AIMessage
)

import streamlit as st
from streamlit_chat import message

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store

# Use vector store and chat history as context to answer user's questions
def ask_and_get_answer(vector_store, query, chat_history, k=3):
    from langchain_openai import ChatOpenAI
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # print(chat_history)

    # Create a history aware retreiver that takes into account the conversation history
    condense_question_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation and a follow up question, \
         rephrase the follow up question to be a standalone question.")
    ])

    history_retriever_chain = create_history_aware_retriever(
        llm, 
        retriever, 
        condense_question_prompt
        )
    
    # Create a document chain that will send prompt to the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following pieces of context to answer the user's question. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    # combine history aware retriever and document chain
    conversational_retrieval_chain = create_retrieval_chain(
        history_retriever_chain, 
        document_chain
        )
    
    # get response from the LLM
    response = conversational_retrieval_chain.invoke({
        "input": query,
        "chat_history": chat_history}
        )
    return response["answer"]


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv(), override=True)

    st.set_page_config(
        page_title='Document Digester',
        page_icon='📚',
    #     initial_sidebar_state='collapsed'
    )
    
    st.image('img.png')
    st.subheader('Document Digester: Talk with your documents 📚')

    # creating the messages (chat history) in the Streamlit session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        # api_key = st.text_input('OpenAI API Key:', type='password')
        # if api_key:
        #     os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number input widget
        chunk_size = st.number_input(
            'Chunk size (value between 100 and 512)', 
            min_value=100, 
            max_value=512, 
            value=256,
            on_change=clear_history
            )
        
        # k number input widget
        k = st.number_input('k (value between 1 and 10)', min_value=1, max_value=10, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        # if the user browsed a file and clicked on Add Data button
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking, and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                # st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Tokens: {tokens}, Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # streamlit chat input widget for the user message
    if user_prompt := st.chat_input("Ask a question about your document"):
        # if the user entered a question
        if 'vs' in st.session_state:
            # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs

            # storing the user question to the session state
            st.session_state.messages.append(
                HumanMessage(content=user_prompt)
            )

            with st.spinner('Working on your request ...'):
                # creating the ChatGPT response
                # response = chat.invoke(st.session_state.messages)
                answer = ask_and_get_answer(
                    vector_store, 
                    user_prompt, 
                    st.session_state.messages,
                    k)
            
            # adding the chatbot's response to the session state
            st.session_state.messages.append(
                AIMessage(content=answer)
            )

    # displaying the messages (chat history)
    # not showing messages[0], as that is the system message 
    for i, msg in enumerate(st.session_state.messages):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=f'{i} + 👽') # user's question
        else:
            message(msg.content, is_user=False, key=f'{i} + 💻') # ChatGPT response

# run the app: streamlit run ./streamlit_app.py
