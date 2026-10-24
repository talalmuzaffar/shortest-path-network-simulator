import streamlit as st
import networkx as nx

from shortest_path_network_simulator.graph import create_initial_graph
from shortest_path_network_simulator.ui import main

# Check if graph exists in the session state, if not, initialize it
if 'graph' not in st.session_state:
    st.session_state['graph'] = nx.Graph()

# Initialize the graph with random nodes and edges if it's empty
if not st.session_state['graph'].nodes:
    st.session_state['graph'] = create_initial_graph()

# Run the main function
if __name__ == '__main__':
    main()
