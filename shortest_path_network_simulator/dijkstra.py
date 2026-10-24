"""Dijkstra's shortest path algorithm."""

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
