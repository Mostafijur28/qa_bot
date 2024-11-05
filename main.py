import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI  # Ensure this is the correct LLM import for your LangChain version
from typing import List
import json
import fitz  # PyMuPDF for PDF extraction

app = FastAPI()

class QuestionAnswer(BaseModel):
    question: str
    answer: str

@app.post("/answer-questions/", response_model=List[QuestionAnswer])
async def answer_questions(
    questions_file: UploadFile = File(...), content_file: UploadFile = File(...)
):
    # Retrieve OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not set in environment.")

    # Step 1: Load questions from the JSON file
    try:
        if not questions_file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Questions file must be in JSON format.")

        questions = json.load(questions_file.file)
        questions_list = [item["question"] for item in questions]
        if not questions_list:
            raise HTTPException(status_code=400, detail="Questions file is empty or invalid format.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse questions file. Ensure it's a valid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the questions file: {str(e)}")

    # Step 2: Extract content from PDF or JSON document file
    try:
        if content_file.filename.endswith('.pdf'):
            pdf_text = ""
            with fitz.open(stream=content_file.file.read(), filetype="pdf") as doc:
                for page in doc:
                    pdf_text += page.get_text("text")
            document_text = pdf_text
        elif content_file.filename.endswith('.json'):
            content_data = json.load(content_file.file)
            document_text = content_data.get("content", "")
            if not document_text:
                raise HTTPException(status_code=400, detail="Content file is empty or invalid format.")
        else:
            raise HTTPException(status_code=400, detail="Content file must be in PDF or JSON format.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse content file. Ensure it's a valid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the content file: {str(e)}")

    # Step 3: Prepare LangChain components with OpenAIEmbeddings
    try:
        openai_api_key = None # Insert your open api key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(document_text)
        docsearch = Chroma.from_texts(texts, embeddings).as_retriever()

        # Use an appropriate LLM with ConversationalRetrievalChain
        llm = OpenAI(api_key=openai_api_key)
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=docsearch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set up LangChain components: {str(e)}")

    # Step 4: Process each question and get answers
    responses = []
    chat_history = []
    try:
        for question in questions_list:
            # Pass chat_history along with question to maintain context if needed
            answer = qa_chain({"question": question, "chat_history": chat_history})
            responses.append({"question": question, "answer": answer['answer']})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing questions: {str(e)}")

    return responses
