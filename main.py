import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def network(type, N, p=None, connections=None, save_as=None):
    """
    Generate and visualize a network based on the specified type, optionally saving it to a file.

    Parameters:
    - type (str): Type of the network ('Erdos-Renyi', 'Ring-Lattice', 'Small-World', 'Scale-Free').
    - N (int): Number of nodes.
    - p (float, optional): Probability for edge creation or rewiring (depends on type).
    - connections (int, optional): Number of connections per node (depends on type).
    - save_as (str, optional): File name to save the graph as a PNG image.
    """
    if type == 'Erdos-Renyi':
        if p is None:
            raise ValueError("Parameter 'p' is required for Erdos-Renyi networks.")
        graph = nx.erdos_renyi_graph(n=N, p=p)

    elif type == 'Ring-Lattice':
        if connections is None:
            raise ValueError("Parameter 'connections' is required for Ring-Lattice networks.")
        graph = nx.watts_strogatz_graph(n=N, k=connections, p=0)

    elif type == 'Small-World':
        if p is None or connections is None:
            raise ValueError("Parameters 'p' and 'connections' are required for Small-World networks.")
        graph = nx.watts_strogatz_graph(n=N, k=connections, p=p)

    elif type == 'Scale-Free':
        if connections is None:
            raise ValueError("Parameter 'connections' is required for Scale-Free networks.")
        graph = nx.barabasi_albert_graph(n=N, m=connections)

    else:
        raise ValueError("Unsupported network type. Choose from 'Erdos-Renyi', 'Ring-Lattice', 'Small-World', 'Scale-Free'.")

    # Calculate and plot network statistics
    degrees = [d for n, d in graph.degree()]
    mean_degrees = np.mean(degrees)
    clustering_coeffs = list(nx.clustering(graph).values())
    mean_clustering = sum(clustering_coeffs) / len(clustering_coeffs)
    try:
        avg_path_length = nx.average_shortest_path_length(graph)
    except nx.NetworkXError:
        avg_path_length = float('inf')  # For disconnected graphs

        # Assign positions for visualization (circular layout)
    pos = nx.circular_layout(graph)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    plt.title(f"{type} Network with N={N}\Avg degrees: {mean_degrees}, Avg clustering coeff: {mean_clustering}, Avg path length: {avg_path_length}")
    nx.draw(graph, pos, node_size=50, node_color='blue', edge_color='gray', with_labels=False)

    # Save the graph if a file name is provided
    if save_as:
        plt.savefig(f"{type}/{save_as}", format='png')
        print(f"Graph saved in {type}/{save_as}")
    # plt.show()

    # Plot degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Degree Distribution ({type} Network)")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    savename1 = f"Degree Distribution {type}.png"
    plt.savefig(f"{type}/{savename1}", format='png')
    print(f"Graph saved in {type}/{savename1}")
    # plt.show()

    # Plot clustering coefficient distribution
    plt.figure(figsize=(8, 6))
    plt.hist(clustering_coeffs, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title(f"Clustering Coefficient Distribution ({type} Network)")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    savename2 = f"Clustering Coefficient {type}.png"
    plt.savefig(f"{type}/{savename2}", format='png')
    print(f"Graph saved as {type}/{savename2}")
    # plt.show()

# Example usage
network('Erdos-Renyi', N=250, p=0.01, save_as='erdos_renyi.png')
network('Ring-Lattice', N=250, connections=4, save_as='ring_lattice.png')
network('Small-World', N=250, p=0.1, connections=4, save_as='small_world.png')
network('Scale-Free', N=250, connections=2, save_as='scale_free.png')