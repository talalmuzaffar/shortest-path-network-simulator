import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import random

# Check if graph exists in the session state, if not, initialize it
if 'graph' not in st.session_state:
    st.session_state['graph'] = nx.Graph()

# Function to generate a random graph
def create_initial_graph(num_nodes=10, num_edges=15):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = random.sample(list(nx.non_edges(G)), min(num_edges, num_nodes * (num_nodes - 1) // 2))
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=random.randint(1, 10))
    return G

# Initialize the graph with random nodes and edges if it's empty
if not st.session_state['graph'].nodes:
    st.session_state['graph'] = create_initial_graph()

# Dijkstra's algorithm implemented from scratch
def dijkstra(graph, start, end):
    Q = set(graph.nodes)
    dist = {node: float('infinity') for node in Q}
    prev = {node: None for node in Q}
    
    dist[start] = 0
    
    while Q:
        u = min(Q, key=lambda node: dist[node])
        Q.remove(u)
        
        if u == end:
            break
        
        for v in graph.neighbors(u):
            alt = dist[u] + graph[u][v]['weight']
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    
    s, u = [], end
    if prev[u] or u == start:
        while u:
            s.insert(0, u)
            u = prev[u]
    return s

# Function to plot the graph using Plotly
def plot_graph(G, path=[]):
    pos = nx.spring_layout(G)
    edge_trace = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        if path and (edge in zip(path, path[1:]) or (edge[1], edge[0]) in zip(path, path[1:])):
            color = 'green'
            width = 4
            dash = 'solid'
        else:
            color = '#888'
            width = 2
            dash = 'dash'

        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color, dash=dash),
            hoverinfo='none',
            mode='lines'))

        # Calculate position for the edge weight label
        middle_hover_trace = go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            text=[str(weight)],
            mode='text',
            hoverinfo='none',
            showlegend=False)
        
        edge_info.append(middle_hover_trace)
    
    # Node trace for Plotly
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Add node positions and labels to the node trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (str(node),)
        node_trace['marker']['color'] += (len(list(G.neighbors(node))),)

    # Create the figure with edge traces, node traces, and edge_info (weights)
    fig = go.Figure(data=edge_trace + [node_trace] + edge_info,
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

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

# Run the main function
if __name__ == '__main__':
    main()
