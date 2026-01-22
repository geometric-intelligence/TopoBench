"""
Interactive Hypergraph Neighborhood Explorer

A Streamlit app for exploring hypergraph neighborhoods with interactive Plotly visualizations.

Run with: streamlit run analysis/neighborhood_explorer_app.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from omegaconf import OmegaConf

from topobench.data.loaders import MAGAArlequinDatasetLoader

# ============================================================================
# Configuration and Constants
# ============================================================================

st.set_page_config(
    page_title="Hypergraph Neighborhood Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

RANK_LABELS = {
    0: "Posts",
    1: "Users", 
    2: "Interactions",
    3: "Communities",
    4: "Semantic Clusters"
}

RANK_COLORS = {
    0: "#1f77b4",  # Blue - Posts
    1: "#ff7f0e",  # Orange - Users
    2: "#2ca02c",  # Green - Interactions
    3: "#d62728",  # Red - Communities
    4: "#9467bd"   # Purple - Semantic Clusters
}

MAX_RANK = 4

# ============================================================================
# Neighborhood Generation Functions
# ============================================================================

def generate_all_neighborhoods(max_rank):
    """Generate all valid neighborhood strings for a hypergraph."""
    neighborhoods = {
        'up_adjacency': [],
        'down_adjacency': [],
        'up_incidence': [],
        'down_incidence': []
    }
    
    for src_rank in range(max_rank + 1):
        for r in range(1, max_rank + 1 - src_rank + 1):
            if src_rank + r <= max_rank:
                neighborhoods['up_adjacency'].append(f"{r}-up_adjacency-{src_rank}")
    
    for src_rank in range(1, max_rank + 1):
        for r in range(1, src_rank + 1):
            if src_rank - r >= 0:
                neighborhoods['down_adjacency'].append(f"{r}-down_adjacency-{src_rank}")
    
    for src_rank in range(max_rank):
        for r in range(1, max_rank + 1 - src_rank + 1):
            if src_rank + r <= max_rank:
                neighborhoods['up_incidence'].append(f"{r}-up_incidence-{src_rank}")
    
    for src_rank in range(1, max_rank + 1):
        for r in range(1, src_rank + 1):
            if src_rank - r >= 0:
                neighborhoods['down_incidence'].append(f"{r}-down_incidence-{src_rank}")
    
    return neighborhoods


def parse_neighborhood(neighborhood_str):
    """Parse a neighborhood string into its components."""
    parts = neighborhood_str.split("-")
    r = int(parts[0])
    direction_type = parts[1]
    direction, ntype = direction_type.split("_")
    src_rank = int(parts[2])
    return r, direction, ntype, src_rank


def get_target_rank(neighborhood_str):
    """Get the target rank for a neighborhood."""
    r, direction, ntype, src_rank = parse_neighborhood(neighborhood_str)
    if direction == "up":
        return src_rank + r
    else:
        return src_rank - r


def describe_neighborhood(neighborhood_str):
    """Generate a human-readable description of a neighborhood."""
    r, direction, ntype, src_rank = parse_neighborhood(neighborhood_str)
    src_label = RANK_LABELS.get(src_rank, f"Rank {src_rank}")
    
    if ntype == "adjacency":
        via_rank = src_rank + r if direction == "up" else src_rank - r
        via_label = RANK_LABELS.get(via_rank, f"Rank {via_rank}")
        return f"{src_label} ↔ {src_label} (via shared {via_label})"
    else:
        target_rank = src_rank + r if direction == "up" else src_rank - r
        target_label = RANK_LABELS.get(target_rank, f"Rank {target_rank}")
        return f"{src_label} → {target_label}"


# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_resource
def load_dataset(cluster_level_posts, cluster_level_users, max_rank, cluster_seed, 
                 neighborhoods, ho_init_method, max_posts_per_user):
    """Load the MAGA Arlequin dataset with caching."""
    config = OmegaConf.create({
        'data_domain': 'hypergraph',
        'data_type': 'social',
        'data_name': 'MAGAArlequin',
        'data_dir': '/data/gbg141/Arlequin/datasets/hypergraph/social',
        'cluster_level_posts': cluster_level_posts,
        'cluster_level_users': cluster_level_users,
        'max_rank': max_rank,
        'cluster_seed': cluster_seed,
        'neighborhoods': neighborhoods,
        'ho_init_method': ho_init_method,
        'max_posts_per_user': max_posts_per_user
    })
    
    loader = MAGAArlequinDatasetLoader(config)
    data = loader.load()[0]
    return data


def get_neighborhood_matrix(data, neighborhood_str):
    """Extract a neighborhood matrix from the data object."""
    if hasattr(data, neighborhood_str):
        return getattr(data, neighborhood_str)
    elif neighborhood_str in data:
        return data[neighborhood_str]
    else:
        return None


def get_matrix_stats(sparse_tensor):
    """Get statistics for a sparse matrix."""
    if sparse_tensor is None:
        return None
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    num_edges = len(values)
    shape = tuple(sparse_tensor.shape)
    max_possible = shape[0] * shape[1]
    sparsity = 1 - (num_edges / max_possible) if max_possible > 0 else 0
    
    # Compute degree statistics
    row_degrees = torch.zeros(shape[0])
    col_degrees = torch.zeros(shape[1])
    for i in range(len(indices[0])):
        row_degrees[indices[0][i]] += 1
        col_degrees[indices[1][i]] += 1
    
    return {
        'shape': shape,
        'num_edges': num_edges,
        'sparsity': sparsity,
        'density': 1 - sparsity,
        'avg_out_degree': row_degrees[row_degrees > 0].mean().item() if (row_degrees > 0).any() else 0,
        'max_out_degree': row_degrees.max().item(),
        'avg_in_degree': col_degrees[col_degrees > 0].mean().item() if (col_degrees > 0).any() else 0,
        'max_in_degree': col_degrees.max().item()
    }


# ============================================================================
# Graph Conversion Functions
# ============================================================================

def sparse_to_networkx(sparse_tensor, max_nodes=200, min_degree=0):
    """Convert sparse tensor to NetworkX graph with filtering."""
    if sparse_tensor is None:
        return None, {}
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().numpy()
    n_edges = indices.shape[1]
    
    if n_edges == 0:
        return None, {}
    
    # Compute node degrees
    node_degrees = {}
    for i in range(n_edges):
        src, tgt = indices[0, i], indices[1, i]
        node_degrees[src] = node_degrees.get(src, 0) + 1
        node_degrees[tgt] = node_degrees.get(tgt, 0) + 1
    
    # Filter by minimum degree
    if min_degree > 0:
        node_degrees = {k: v for k, v in node_degrees.items() if v >= min_degree}
    
    # Sample top nodes if too many
    if len(node_degrees) > max_nodes:
        top_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes]
        node_degrees = {k: node_degrees[k] for k in top_nodes}
    
    valid_nodes = set(node_degrees.keys())
    
    # Determine if adjacency (square) or incidence (rectangular)
    is_adjacency = sparse_tensor.shape[0] == sparse_tensor.shape[1]
    
    if is_adjacency:
        G = nx.Graph()
        for node in valid_nodes:
            G.add_node(node, degree=node_degrees[node])
        
        for i in range(n_edges):
            src, tgt = indices[0, i], indices[1, i]
            if src in valid_nodes and tgt in valid_nodes and src != tgt:
                G.add_edge(src, tgt)
    else:
        # Bipartite graph
        G = nx.DiGraph()
        src_nodes = set(indices[0])
        tgt_nodes = set(indices[1])
        
        for node in src_nodes:
            if node in valid_nodes or len(valid_nodes) == 0:
                G.add_node(f"src_{node}", bipartite=0, original_id=node)
        for node in tgt_nodes:
            G.add_node(f"tgt_{node}", bipartite=1, original_id=node)
        
        for i in range(n_edges):
            src, tgt = indices[0, i], indices[1, i]
            src_key = f"src_{src}"
            tgt_key = f"tgt_{tgt}"
            if src_key in G.nodes() and tgt_key in G.nodes():
                G.add_edge(src_key, tgt_key)
    
    return G, node_degrees


def get_node_neighbors(sparse_tensor, node_id, direction='both'):
    """Get neighbors of a specific node."""
    if sparse_tensor is None:
        return set()
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().numpy()
    
    neighbors = set()
    for i in range(indices.shape[1]):
        src, tgt = indices[0, i], indices[1, i]
        if direction in ['out', 'both'] and src == node_id:
            neighbors.add(tgt)
        if direction in ['in', 'both'] and tgt == node_id:
            neighbors.add(src)
    
    return neighbors


# ============================================================================
# Plotly Visualization Functions
# ============================================================================

def create_plotly_graph(G, neighborhood_str, node_degrees, height=600, selected_node=None):
    """Create an interactive Plotly graph visualization."""
    if G is None or len(G.nodes()) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    r, direction, ntype, src_rank = parse_neighborhood(neighborhood_str)
    is_adjacency = ntype == 'adjacency'
    target_rank = get_target_rank(neighborhood_str)
    
    # Compute layout
    if is_adjacency:
        pos = nx.spring_layout(G, k=2/np.sqrt(max(len(G.nodes()), 1)), iterations=50, seed=42)
    else:
        # Bipartite layout
        top_nodes = [n for n in G.nodes() if str(n).startswith('src_')]
        bottom_nodes = [n for n in G.nodes() if str(n).startswith('tgt_')]
        pos = {}
        for i, n in enumerate(top_nodes):
            pos[n] = (i / max(1, len(top_nodes)-1) if len(top_nodes) > 1 else 0.5, 1)
        for i, n in enumerate(bottom_nodes):
            pos[n] = (i / max(1, len(bottom_nodes)-1) if len(bottom_nodes) > 1 else 0.5, 0)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_ids = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)
        
        # Determine color based on rank
        if is_adjacency:
            color = RANK_COLORS.get(src_rank, '#888888')
            node_id = node
            rank_label = RANK_LABELS.get(src_rank, f"Rank {src_rank}")
        else:
            if str(node).startswith('src_'):
                color = RANK_COLORS.get(src_rank, '#888888')
                node_id = G.nodes[node].get('original_id', node)
                rank_label = RANK_LABELS.get(src_rank, f"Rank {src_rank}")
            else:
                color = RANK_COLORS.get(target_rank, '#888888')
                node_id = G.nodes[node].get('original_id', node)
                rank_label = RANK_LABELS.get(target_rank, f"Rank {target_rank}")
        
        node_colors.append(color)
        
        # Size based on degree
        degree = node_degrees.get(node if is_adjacency else G.nodes[node].get('original_id', 0), 1)
        node_sizes.append(max(8, min(30, 5 + np.log1p(degree) * 5)))
        
        # Hover text
        hover = f"<b>{rank_label}</b><br>"
        hover += f"ID: {node_id}<br>"
        hover += f"Degree: {degree}"
        node_text.append(hover)
    
    # Highlight selected node and its neighbors
    if selected_node is not None and selected_node in G.nodes():
        neighbors = set(G.neighbors(selected_node))
        for i, node in enumerate(node_ids):
            if node == selected_node:
                node_colors[i] = '#ff0000'  # Red for selected
                node_sizes[i] = node_sizes[i] * 1.5
            elif node in neighbors:
                node_colors[i] = '#ffff00'  # Yellow for neighbors
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=dict(
            text=f"{neighborhood_str}<br><sub>{describe_neighborhood(neighborhood_str)}</sub>",
            x=0.5
        ),
        showlegend=False,
        hovermode='closest',
        height=height,
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig


def create_degree_distribution_plot(sparse_tensor, title="Degree Distribution"):
    """Create a degree distribution histogram."""
    if sparse_tensor is None:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().numpy()
    
    # Compute degrees
    degrees = {}
    for i in range(indices.shape[1]):
        src = indices[0, i]
        degrees[src] = degrees.get(src, 0) + 1
    
    degree_values = list(degrees.values())
    
    fig = go.Figure(data=[go.Histogram(x=degree_values, nbinsx=50)])
    fig.update_layout(
        title=title,
        xaxis_title="Degree",
        yaxis_title="Count",
        height=300
    )
    
    return fig


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.title("🔗 Hypergraph Neighborhood Explorer")
    st.markdown("Interactive visualization of hypergraph neighborhood structures")
    
    # ========================================================================
    # Sidebar - Data Configuration
    # ========================================================================
    
    with st.sidebar:
        st.header("📊 Data Configuration")
        
        with st.expander("Dataset Parameters", expanded=True):
            cluster_level_posts = st.selectbox(
                "Cluster Level (Posts)",
                options=[-2, -1, 0, 1, 2],
                index=0,
                help="FINCH clustering level for posts (-2 = finest)"
            )
            
            cluster_level_users = st.selectbox(
                "Cluster Level (Users)",
                options=[0, 1, 2],
                index=0,
                help="FINCH clustering level for users"
            )
            
            cluster_seed = st.number_input(
                "Cluster Seed",
                value=42,
                min_value=0,
                help="Random seed for clustering"
            )
            
            ho_init_method = st.selectbox(
                "HO Init Method",
                options=["bio", "avg_post"],
                index=0,
                help="Higher-order structure initialization method"
            )
            
            max_posts_per_user = st.number_input(
                "Max Posts per User",
                value=10000,
                min_value=100,
                max_value=50000,
                help="Filter out users with more than this many posts (set high to disable)"
            )
        
        # Generate neighborhoods
        all_neighborhoods = generate_all_neighborhoods(MAX_RANK)
        all_neighborhood_list = []
        for neighs in all_neighborhoods.values():
            all_neighborhood_list.extend(neighs)
        
        # Load button
        if st.button("🔄 Load Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading dataset..."):
                try:
                    data = load_dataset(
                        cluster_level_posts=cluster_level_posts,
                        cluster_level_users=cluster_level_users,
                        max_rank=MAX_RANK,
                        cluster_seed=cluster_seed,
                        neighborhoods=all_neighborhood_list,
                        ho_init_method=ho_init_method,
                        max_posts_per_user=max_posts_per_user if max_posts_per_user < 50000 else None
                    )
                    st.session_state['data'] = data
                    st.session_state['neighborhoods'] = all_neighborhoods
                    st.success("Dataset loaded!")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
        
        st.divider()
        
        # Visualization settings
        st.header("🎨 Visualization Settings")
        
        max_nodes = st.slider(
            "Max Nodes to Display",
            min_value=50,
            max_value=500,
            value=150,
            help="Maximum number of nodes to show in the graph"
        )
        
        min_degree = st.slider(
            "Minimum Degree Filter",
            min_value=0,
            max_value=20,
            value=0,
            help="Only show nodes with at least this many connections"
        )
        
        st.session_state['max_nodes'] = max_nodes
        st.session_state['min_degree'] = min_degree
    
    # ========================================================================
    # Main Content
    # ========================================================================
    
    if 'data' not in st.session_state:
        st.info("👈 Configure and load the dataset using the sidebar to get started.")
        
        # Show neighborhood overview
        st.subheader("Available Neighborhoods")
        
        all_neighborhoods = generate_all_neighborhoods(MAX_RANK)
        
        cols = st.columns(2)
        for idx, (ntype, neighs) in enumerate(all_neighborhoods.items()):
            with cols[idx % 2]:
                st.markdown(f"**{ntype.replace('_', ' ').title()}**")
                for n in neighs:
                    st.markdown(f"- `{n}`: {describe_neighborhood(n)}")
        
        return
    
    data = st.session_state['data']
    neighborhoods = st.session_state['neighborhoods']
    max_nodes = st.session_state.get('max_nodes', 150)
    min_degree = st.session_state.get('min_degree', 0)
    
    # Display shape info
    shape = data.shape if hasattr(data, 'shape') else data['shape']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Posts", f"{shape[0]:,}")
    with col2:
        st.metric("Users", f"{shape[1]:,}")
    with col3:
        st.metric("Interactions", f"{shape[2]:,}")
    with col4:
        st.metric("Communities", f"{shape[3]:,}")
    with col5:
        st.metric("Semantic", f"{shape[4]:,}")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Neighborhood", "⚖️ Compare Neighborhoods", "🎯 Local Explorer"])
    
    # ========================================================================
    # Tab 1: Single Neighborhood View
    # ========================================================================
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Select Neighborhood")
            
            # Category selection
            category = st.selectbox(
                "Neighborhood Type",
                options=list(neighborhoods.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Neighborhood selection
            selected_neighborhood = st.selectbox(
                "Neighborhood",
                options=neighborhoods[category],
                format_func=lambda x: f"{x} - {describe_neighborhood(x)}"
            )
            
            # Get matrix and stats
            matrix = get_neighborhood_matrix(data, selected_neighborhood)
            stats = get_matrix_stats(matrix) if matrix is not None else None
            
            if stats:
                st.markdown("---")
                st.markdown("**Statistics**")
                st.markdown(f"- Shape: {stats['shape']}")
                st.markdown(f"- Edges: {stats['num_edges']:,}")
                st.markdown(f"- Density: {stats['density']:.4%}")
                st.markdown(f"- Avg Out-Degree: {stats['avg_out_degree']:.2f}")
                st.markdown(f"- Max Out-Degree: {stats['max_out_degree']:.0f}")
        
        with col2:
            if matrix is not None:
                # Create graph
                G, node_degrees = sparse_to_networkx(matrix, max_nodes=max_nodes, min_degree=min_degree)
                
                if G and len(G.nodes()) > 0:
                    fig = create_plotly_graph(G, selected_neighborhood, node_degrees, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Showing {len(G.nodes())} nodes and {len(G.edges())} edges")
                else:
                    st.warning("No nodes match the current filter settings.")
            else:
                st.warning(f"Neighborhood '{selected_neighborhood}' not found in data.")
        
        # Degree distribution
        if matrix is not None:
            st.subheader("Degree Distribution")
            fig = create_degree_distribution_plot(matrix, f"Degree Distribution - {selected_neighborhood}")
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # Tab 2: Side-by-Side Comparison
    # ========================================================================
    
    with tab2:
        st.subheader("Compare Two Neighborhoods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat1 = st.selectbox(
                "Type (Left)",
                options=list(neighborhoods.keys()),
                format_func=lambda x: x.replace('_', ' ').title(),
                key="cat1"
            )
            neigh1 = st.selectbox(
                "Neighborhood (Left)",
                options=neighborhoods[cat1],
                format_func=lambda x: f"{x}",
                key="neigh1"
            )
        
        with col2:
            cat2 = st.selectbox(
                "Type (Right)",
                options=list(neighborhoods.keys()),
                format_func=lambda x: x.replace('_', ' ').title(),
                key="cat2",
                index=1 if len(neighborhoods) > 1 else 0
            )
            neigh2 = st.selectbox(
                "Neighborhood (Right)",
                options=neighborhoods[cat2],
                format_func=lambda x: f"{x}",
                key="neigh2"
            )
        
        # Create side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            matrix1 = get_neighborhood_matrix(data, neigh1)
            if matrix1 is not None:
                G1, nd1 = sparse_to_networkx(matrix1, max_nodes=max_nodes//2, min_degree=min_degree)
                if G1 and len(G1.nodes()) > 0:
                    fig1 = create_plotly_graph(G1, neigh1, nd1, height=500)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    stats1 = get_matrix_stats(matrix1)
                    st.markdown(f"**Edges:** {stats1['num_edges']:,} | **Density:** {stats1['density']:.4%}")
        
        with col2:
            matrix2 = get_neighborhood_matrix(data, neigh2)
            if matrix2 is not None:
                G2, nd2 = sparse_to_networkx(matrix2, max_nodes=max_nodes//2, min_degree=min_degree)
                if G2 and len(G2.nodes()) > 0:
                    fig2 = create_plotly_graph(G2, neigh2, nd2, height=500)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    stats2 = get_matrix_stats(matrix2)
                    st.markdown(f"**Edges:** {stats2['num_edges']:,} | **Density:** {stats2['density']:.4%}")
        
        # Comparison table
        if matrix1 is not None and matrix2 is not None:
            st.subheader("Statistics Comparison")
            
            stats1 = get_matrix_stats(matrix1)
            stats2 = get_matrix_stats(matrix2)
            
            comparison_df = pd.DataFrame({
                'Metric': ['Shape', 'Edges', 'Density', 'Avg Out-Degree', 'Max Out-Degree'],
                neigh1: [
                    str(stats1['shape']),
                    f"{stats1['num_edges']:,}",
                    f"{stats1['density']:.4%}",
                    f"{stats1['avg_out_degree']:.2f}",
                    f"{stats1['max_out_degree']:.0f}"
                ],
                neigh2: [
                    str(stats2['shape']),
                    f"{stats2['num_edges']:,}",
                    f"{stats2['density']:.4%}",
                    f"{stats2['avg_out_degree']:.2f}",
                    f"{stats2['max_out_degree']:.0f}"
                ]
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # Tab 3: Local Neighborhood Explorer
    # ========================================================================
    
    with tab3:
        st.subheader("Explore Local Neighborhood")
        st.markdown("Select a node to see its connections across different neighborhood types.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Select rank and node
            selected_rank = st.selectbox(
                "Select Rank",
                options=list(RANK_LABELS.keys()),
                format_func=lambda x: f"{x}: {RANK_LABELS[x]}"
            )
            
            # Get max node ID for this rank
            max_node_id = shape[selected_rank] - 1
            
            selected_node_id = st.number_input(
                f"Node ID (0 - {max_node_id})",
                min_value=0,
                max_value=max_node_id,
                value=0
            )
            
            # Find relevant neighborhoods
            st.markdown("---")
            st.markdown("**Relevant Neighborhoods:**")
            
            relevant_up = [n for n in neighborhoods['up_adjacency'] if int(n.split('-')[2]) == selected_rank]
            relevant_down = [n for n in neighborhoods['down_adjacency'] if int(n.split('-')[2]) == selected_rank]
            
            if relevant_up:
                st.markdown("*Up Adjacency:*")
                for n in relevant_up:
                    st.markdown(f"- {n}")
            
            if relevant_down:
                st.markdown("*Down Adjacency:*")
                for n in relevant_down:
                    st.markdown(f"- {n}")
        
        with col2:
            # Show neighbors in selected neighborhoods
            st.markdown(f"**Neighbors of Node {selected_node_id} ({RANK_LABELS[selected_rank]})**")
            
            all_adjacencies = relevant_up + relevant_down
            
            if all_adjacencies:
                for neigh in all_adjacencies[:4]:  # Show up to 4 neighborhoods
                    matrix = get_neighborhood_matrix(data, neigh)
                    if matrix is not None:
                        neighbors = get_node_neighbors(matrix, selected_node_id)
                        
                        with st.expander(f"{neigh} ({len(neighbors)} neighbors)", expanded=True):
                            if neighbors:
                                # Create a small subgraph
                                matrix_coalesced = matrix.coalesce()
                                indices = matrix_coalesced.indices().numpy()
                                
                                # Build subgraph
                                G_sub = nx.Graph()
                                G_sub.add_node(selected_node_id)
                                for n in list(neighbors)[:50]:  # Limit neighbors
                                    G_sub.add_node(n)
                                    G_sub.add_edge(selected_node_id, n)
                                
                                # Simple visualization
                                node_degrees = {n: 1 for n in G_sub.nodes()}
                                node_degrees[selected_node_id] = len(neighbors)
                                
                                fig = create_plotly_graph(
                                    G_sub, neigh, node_degrees, height=300, 
                                    selected_node=selected_node_id
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No neighbors found in this neighborhood.")
            else:
                st.info("No adjacency neighborhoods available for this rank.")


if __name__ == "__main__":
    main()
