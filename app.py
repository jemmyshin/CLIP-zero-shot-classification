import streamlit as st
from helper import get_label_embedding, classify
import os
from docarray import DocumentArray, Document
from clip_client import Client


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
        st.text(f"Output: {results}")
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
