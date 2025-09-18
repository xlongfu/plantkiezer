from __future__ import annotations

import pandas as pd
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

import os

api_key = st.secrets["API_KEY"]

# lock_file = "vector_stores/plantkiezer/.lock"
# if os.path.exists(lock_file):
#     os.remove(lock_file)


st.set_page_config(page_title="Plantkiezer.nl", page_icon="🌿", layout="wide")
st.title("🌿 Plantkiezer Plant Recommender")
st.caption("Type your plant needs; I'll recommend and show product cards.")

llm = init_chat_model(
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    model_provider="fireworks",
    api_key=api_key,
    model_kwargs={
        "max_tokens": 512,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.5,
    },
)

# ----- RAG process ----- 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Qdrant.from_existing_collection(
    collection_name="texas_plants",
    embedding=embeddings,
    path="vector_stores/plantkiezer",
)

# -----------------------

# init and show chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# create chat input field 
if prompt := st.chat_input("What kind of plants are you interested in?"):

    # store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # convert session messages to langchain message objects
    session_messages = []

    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            session_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            session_messages.append(AIMessage(content=content))
        else:
            session_messages.append(HumanMessage(content=content))

    query = st.session_state.messages[-1]["content"]

    retrieved_docs = vectorstore.max_marginal_relevance_search(query, k=3, filter=None)

    def format_docs(docs):
        blocks = []
    
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source") or d.metadata.get("id") or f"doc-{i}"
            blocks.append(f"[{i}] {d.page_content}")
        
        return "\n\n".join(blocks)

    context = format_docs(retrieved_docs)

    instruction = (
        "You are an expert botanical assistant and also a sales chatbot. \n"
        "You will be provided with three retrieved plant entries. \n"
        f"Context:\n{context}\n\n"
        "Answer the user query by recommending these three plants, if it is \n"
        "relevant and appropriate to the user query. \n"
        "Use the descriptions of the retrieved data to also provide more \n"
        "information about the plants, if you decide to recommend them. \n"
        "Frame your response concisely, while also sounding \n"
        "like a real salesperson. Here is the user question: \n\n"
    )

    all_context = [SystemMessage(content=instruction)] + session_messages

    # ------ Generation ------
    with st.spinner("Thinking…"):
        response = llm.generate([all_context])

    
    assistant_text = ""

    try:
        assistant_text = response.generations[0][0].text or ""
    except Exception:
        assistant_text = "Sorry, I couldn't generate a response. Try again later."

    # --- Display + store ---
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        
        def show_product_card(row):
            uid = row.metadata['uid']
            price = row.metadata['price']
            delivery = row.metadata['delivery']
            labels = row.metadata['labels']
            labels = labels.split(",") if labels else []

            name = row.metadata['common_name']

            # get image from local folder
            img = None

            if uid:
                images_dir = os.path.join("data", "images")

                if os.path.isdir(images_dir):
                    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
                        candidate = os.path.join(images_dir, f"{uid}{ext}")

                        if os.path.exists(candidate):
                            img = candidate
                            break
                
                else:
                    img = os.path.join(images_dir, "placeholder.png")

            st.markdown('<div class="card">', unsafe_allow_html=True)

            if img:
                try:
                    if os.path.exists(img):
                        st.image(img, use_column_width=True)
                    else:
                        st.markdown(f'<img src="{img}" alt="{name}">', unsafe_allow_html=True)
                except Exception:
                    pass

            st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='price'>{price}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta'>Delivery: {delivery}</div>", unsafe_allow_html=True)
            if labels:
                st.write(" ".join([f"<span class='badge'>{l.strip()}</span>" for l in labels if l.strip()]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        for i, row in enumerate(retrieved_docs):
            show_product_card(row)

    # ------ Finally append the message into history ------
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
