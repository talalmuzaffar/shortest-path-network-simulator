"""Plotly figure for a graph, with an optional highlighted path."""

import networkx as nx
import plotly.graph_objects as go

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
