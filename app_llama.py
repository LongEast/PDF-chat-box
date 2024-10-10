import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.vectorstores import FAISS

from langchain_ollama import OllamaLLM

from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


import getpass

st.set_page_config(page_title='🤗💬 PDF Chat App - GPT')
llm = OllamaLLM(model="llama3.2")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Llama](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
    ''')




def main():
    st.header("Talk to your PDF 💬")
    st.write("This app uses LLM model to answer questions about your PDF file. Upload your PDF file and ask questions about it. The app will return the answer from your PDF file.")

    st.header("1. Pass your Llama 3.2 API KEY here")
    v='demo'
    llama_key=st.text_input("**LLAMA API KEY**", value=v)
    st.write("You can get your llama 3.2 key from [here](https://console.llamaapi.com/)")


    if llama_key ==v:
        llama_key=st.secrets["LLAMA_API_KEY"]
    # if openai_key=='':
    #     load_dotenv()
    os.environ["LLAMA_API_KEY"] = llama_key

    # upload a PDF file

    st.header("2. Upload PDF")
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')

    # st.write(pdf)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            model_path = r"\\wsl.localhost\Ubuntu\mnt\wslg\distro\usr\share\ollama\.ollama\models\manifests\registry.ollama.ai\library\llama3.2"

            embeddings = LlamaCppEmbeddings(model_path=model_path)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    # st.header("or.. Try it with this The Alchaemist PDF book")
    # if st.button('Ask The Alchemist Book Questions'):
    #     with open("The_Alchemist.pkl", "rb") as f:
    #         VectorStore = pickle.load(f)
        # Accept user questions/query
        st.header("3. Ask questions about your PDF file:")
        q="Tell me about the content of the PDF"
        query = st.text_input("Questions",value=q)
        # st.write(query)

        if st.button("Ask"):
            # st.write(openai_key)
            # os.environ["OPENAI_API_KEY"] = openai_key
            if llama_key =='':
                st.write('Warning: Please pass your LLAMA API KEY on Step 1')
            else:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OllamaLLM(model="llama3")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.header("Answer:")
                st.write(response)
                st.write('--')
                st.header("LLAMA API Usage:")
                st.text(cb)

if __name__ == '__main__':
    main()