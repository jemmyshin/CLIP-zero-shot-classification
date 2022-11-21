import streamlit as st
from helper import get_label_embedding, classify
import os
import pandas as pd
from docarray import DocumentArray, Document
from clip_client import Client

os.environ['JINA_AUTH_TOKEN'] = '31454a8d0823445012c6de5623aed215'

def embed_tags():
    tags = []
    cur_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(cur_dir, 'tags'), 'r') as f:
        for line in f:
            tags.append(line.split('\n')[0])

    st.session_state.labels = DocumentArray([Document(text=tag) for tag in tags])
    with st.spinner(f"preparing for label..."):
        st.session_state.labels = get_label_embedding(st.session_state.labels, client)

def search():
    with st.spinner(f"Processing..."):
        results = classify(img, st.session_state.labels, int(topn_value), client)
        results = [[results[0][i], results[1][i]] for i in range(len(results[0]))]
        st.text(f"Output label: {results[0][0]}, score: {results[0][1]}")
        st.text("top k results: ")
        df = pd.DataFrame(
                    results,
                    columns=('label', 'score'))
        st.dataframe(df)
    st.success('Done!')

st.set_page_config(page_title='CLIP zero-shot classification', page_icon='🔍')
st.title('CLIP zero-shot classification')

uploaded_file = st.file_uploader('Choose an image')
topn_value = st.text_input('Top N', '5')
cas_url = st.text_input('CLIP-as-service Server', 'grpcs://api.clip.jina.ai:2096')
search_button = st.button('Search')

client = Client(cas_url, credential={'Authorization': os.getenv('JINA_AUTH_TOKEN')})

if uploaded_file:
    img = uploaded_file.getvalue()

if search_button:
    if 'labels' not in st.session_state:
        embed_tags()
    search()
