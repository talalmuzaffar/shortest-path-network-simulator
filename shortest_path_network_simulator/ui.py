"""Streamlit page: sidebar controls, graph plots and shortest path."""

import streamlit as st

from shortest_path_network_simulator.dijkstra import dijkstra
from shortest_path_network_simulator.graph import create_initial_graph
from shortest_path_network_simulator.plotting import plot_graph

# Main function to render the Streamlit app
def main():
    st.title('Network Simulation and shortest path using dijkstra')

    # Sidebar inputs to control graph modifications
    with st.sidebar:
        if st.button('Generate Random Graph'):
            st.session_state['graph'] = create_initial_graph()

        node_id = st.number_input('Enter new node ID', min_value=0, value=len(st.session_state['graph'].nodes()), format='%d')
        if st.button('Add Node'):
            st.session_state['graph'].add_node(node_id)

        node1_id = st.selectbox('Select Node 1', st.session_state['graph'].nodes())
        node2_id = st.selectbox('Select Node 2', st.session_state['graph'].nodes())
        edge_weight = st.number_input('Enter edge weight', min_value=1, max_value=100, value=1, step=1)
        if st.button('Add Edge'):
            if node1_id != node2_id:
                st.session_state['graph'].add_edge(node1_id, node2_id, weight=edge_weight)
            else:
                st.warning('Select different nodes to create an edge.')

    # Display the original graph
    st.write('Original Graph')
    fig = plot_graph(st.session_state['graph'])
    st.plotly_chart(fig)

    # Option to find and display the shortest path using the implemented Dijkstra's algorithm
    if st.sidebar.checkbox('Show Shortest Path'):
        start_node = st.sidebar.selectbox('Start Node for Shortest Path', st.session_state['graph'].nodes())
        end_node = st.sidebar.selectbox('End Node for Shortest Path', st.session_state['graph'].nodes())
        if start_node != end_node:
            path = dijkstra(st.session_state['graph'], start_node, end_node)
            if path:
                st.write('Shortest Path: ', path)
                st.write('Graph with Shortest Path Highlighted')
                fig = plot_graph(st.session_state['graph'], path)
                st.plotly_chart(fig)
            else:
                st.error("No path exists between these nodes.")
        else:
            st.sidebar.warning('Select different nodes for start and end.')
