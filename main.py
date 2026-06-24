import pandas as pd
from fastapi import FastAPI
from sqlalchemy import create_engine
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline

app = FastAPI()

# ---- Data ab Supabase (Postgres) se aa raha hai ----
DATABASE_URL = "postgresql://postgres.qttucghmkshukoulcvpx:%23Saeem78912@aws-1-ap-southeast-2.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL)

forecast_df = pd.read_sql("SELECT * FROM crop_forecasts", engine)

forecast_df['next_month_price'] = forecast_df['yhat'].shift(-1)
forecast_df['price_change'] = forecast_df['next_month_price'] - forecast_df['yhat']
forecast_df['recommendation'] = forecast_df['price_change'].apply(
    lambda x: 'Store' if x > 0 else 'Sell'
)

forecast_df['text'] = forecast_df.apply(
    lambda row: f"Month: {row['ds']}\nPredicted Price: {row['yhat']:.2f} PKR\nRecommendation: {row['recommendation']}",
    axis=1
)

loader = DataFrameLoader(forecast_df, page_content_column="text")
documents = loader.load()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="crop_forecast"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm = HuggingFacePipeline(pipeline=generator)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an agricultural advisor.
Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)


@app.post("/get-recommendation")
def get_recommendation(query: str):
    answer = qa_chain.invoke({"query": query})
    raw_answer = answer["result"]

    if "Answer:" in raw_answer:
        clean_answer = raw_answer.split("Answer:")[-1].strip()
    else:
        clean_answer = raw_answer.strip()

    return {"answer": clean_answer}
