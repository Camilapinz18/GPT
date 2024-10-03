import os
import streamlit as st

from utils.pdf_analisis import *
from utils.vectorstore import *

VECTORSTORE_DIR = "vectorstore"

def home_page():
    st.title("Análizador de texto")
    st.write("Aquí podrar analiazar y alamcenar sus archivos PDF")

    st.subheader(":gear: Options")
        
    # Step 1: Choose a Large Language Model
    llm_selection = "OpenAI"

    # Step 2: Choose Embeddings Model
    embeddings_selection ="OpenAI"
    # Step 3: Select or Create a Vector Store File
    vectorstore_files = ["Create New"] + os.listdir(VECTORSTORE_DIR)
    st.session_state.vectorstore_selection = st.selectbox(
        "Step 3: Choose a Vector Store File",
        options=vectorstore_files
    )

    custom_vectorstore_name = st.text_input("Enter a name for the new vectorstore (if creating new):", value="")


    # Handle file upload
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if st.button("Process"):
        with st.spinner("Processing"):
            # Get PDF text
            raw_text = get_pdf_text(pdf_docs)

            # Get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # Create or load vector store
            if (
                st.session_state.vectorstore_selection == "Create New"
                or not os.path.exists(
                    os.path.join(VECTORSTORE_DIR, st.session_state.vectorstore_selection)
                )
            ):                   

                vectorstore = get_vectorstore(text_chunks, embeddings_selection)
                vectorstore_filename = f"{custom_vectorstore_name}.pkl"
                save_vectorstore(vectorstore, vectorstore_filename)
                # st.session_state.vectorstore_selection = vectorstore_filename  # Update the current selection to the new file
                st.success(f"El vectorstore '{vectorstore_filename}' ha sido guardado exitosamente.")
            else:
                vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
                vectorstore.update(text_chunks)

            # Get the current vectorstore
            current_vectorstore = vectorstore

            # Create conversation chain
            if current_vectorstore is not None:
                st.session_state.conversation = chain_setup(current_vectorstore, llm_selection)

    if st.button("Clear Chat"):
        st.session_state.user = []
        st.session_state.generated = []
        st.session_state.cost = []

def analysis_page():
    st.title("Página de Análisis")
    st.write("Aquí se realizará el análisis de la imagen.")
    # Agregar el código relacionado con el análisis aquí

def about_page():
    st.title("Acerca de")
    st.write("Esta aplicación analiza planos constructivos para verificar el cumplimiento de normativas.")
