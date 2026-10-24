# network-simulator

It is code for streamlit application that uses NetworkX to create a graph and find the shortest path using Dijkstra

## How to run

Install the requirements and start the app from the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the app in your browser (by default at http://localhost:8501). Use the sidebar to generate a random graph, add nodes and edges, then tick "Show Shortest Path" and pick a start and end node to see the path highlighted.

## Project structure

```
shortest-path-network-simulator/
├── app.py                              # Entry point: sets up the graph in session state and runs the UI
├── shortest_path_network_simulator/
│   ├── __init__.py
│   ├── graph.py                        # Random graph generation
│   ├── dijkstra.py                     # Dijkstra's algorithm, written from scratch
│   ├── plotting.py                     # Draws the graph with Plotly and highlights the shortest path
│   └── ui.py                           # Streamlit sidebar controls and page layout
├── requirements.txt                    # Python dependencies
└── README.md
```
