import sys
sys.path.insert(0, '.')

import streamlit as st
import utils
from qtrader.rlflow.persistence import SQLitePersistenceProvider

utils.download_db()
ppprovider = SQLitePersistenceProvider(root='.')


keys = list(ppprovider.list(prefix=''))
key = st.selectbox(label='Key', options=keys)

v = ppprovider.load_dict(key)
st.json(v)
