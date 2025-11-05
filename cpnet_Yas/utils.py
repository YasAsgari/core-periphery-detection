from collections import Counter, defaultdict

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy import sparse
import scipy.sparse as sp

def to_adjacency_matrix(net):
    if sp.issparse(net):
        A = net.tocsr().astype(np.float64)
        labels = np.arange(A.shape[0])
    elif isinstance(net, np.ndarray):
        A = sp.csr_matrix(net, dtype=np.float64)
        labels = np.arange(A.shape[0])
    elif isinstance(net, nx.Graph):
        try:
            A = nx.to_scipy_sparse_array(net, format="csr", dtype=np.float64)
        except AttributeError:
            A = nx.adjacency_matrix(net).astype(np.float64)
        labels = np.array(list(net.nodes()))
    else:
        raise TypeError("Unsupported graph type")

    A = A.maximum(A.T)        
    A.data[:] = 1.0            
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr(), labels


def to_nxgraph(net):
    """Convert to an undirected NetworkX graph."""
    if sparse.issparse(net):
        # NetworkX 3.x prefers from_scipy_sparse_array
        return nx.from_scipy_sparse_array(net, create_using=nx.Graph)
    elif isinstance(net, nx.Graph):
        return net
    elif isinstance(net, np.ndarray):
        return nx.from_numpy_array(net)
    else:
        raise TypeError(f"Unsupported type for to_nxgraph: {type(net)}")


def set_node_colors(c, x, cmap, colored_nodes):
    """Return dicts: node -> fill color, node -> edge color."""
    # defaultdict factory must be zero-arg
    node_colors = defaultdict(lambda: "#8d8d8d")
    node_edge_colors = defaultdict(lambda: "#4d4d4d")

    cnt = Counter([c[d] for d in colored_nodes])
    num_groups = len(cnt)

    # palette for groups
    if cmap is None:
        if num_groups <= 10:
            base = sns.color_palette().as_hex()
        elif num_groups <= 20:
            base = sns.color_palette("tab20").as_hex()
        else:
            base = sns.color_palette("hls", num_groups).as_hex()
    else:
        # allow a list-like cmap to be passed
        base = list(cmap)

    group_to_color = dict(
        zip([d[0] for d in cnt.most_common(num_groups)], [base[i] for i in range(num_groups)])
    )

    bounds = np.linspace(0, 1, 11)  # 10 bins between 0..1
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=12, extend="both")

    # make light/dark ramps per group
    cmap_coreness = {k: sns.light_palette(v, n_colors=12).as_hex() for k, v in group_to_color.items()}
    cmap_coreness_dark = {k: sns.dark_palette(v, n_colors=12).as_hex() for k, v in group_to_color.items()}

    for d in colored_nodes:
        gid = c[d]
        bin_idx = int(norm(x[d]))  # 0..11
        bin_idx = max(0, min(bin_idx, 11))
        node_colors[d] = cmap_coreness[gid][bin_idx]
        node_edge_colors[d] = cmap_coreness_dark[gid][11 - bin_idx]
    return node_colors, node_edge_colors


def classify_nodes(G, c, x, max_num=None):
    non_residuals = [d for d in G.nodes() if (c[d] is not None) and (x[d] is not None)]
    residuals = [d for d in G.nodes() if (c[d] is None) or (x[d] is None)]

    cnt = Counter([c[d] for d in non_residuals])
    order_groups = [d[0] for d in cnt.most_common(len(cnt))]
    cset = set(order_groups[:max_num]) if max_num is not None else set(order_groups)

    colored = [d for d in non_residuals if c[d] in cset]
    muted = [d for d in non_residuals if c[d] not in cset]

    # Bring core nodes to front (descending by coreness)
    order = np.argsort([-x[d] for d in colored])
    colored = [colored[i] for i in order]
    return colored, muted, residuals


def calc_node_pos(G, layout_algorithm):
    return nx.spring_layout(G) if layout_algorithm is None else layout_algorithm(G)


def draw(
    G,
    c,
    x,
    ax,
    draw_edge=True,
    font_size=0,
    pos=None,
    cmap=None,
    max_group_num=None,
    draw_nodes_kwd=None,
    draw_edges_kwd=None,
    draw_labels_kwd=None,
    layout_algorithm=None,
):
    """Matplotlib static drawing (colors ramped by coreness within each group)."""
    draw_nodes_kwd = {} if draw_nodes_kwd is None else draw_nodes_kwd
    draw_edges_kwd = {"edge_color": "#adadad"} if draw_edges_kwd is None else draw_edges_kwd
    draw_labels_kwd = {} if draw_labels_kwd is None else draw_labels_kwd

    colored_nodes, muted_nodes, residuals = classify_nodes(G, c, x, max_group_num)
    node_colors, node_edge_colors = set_node_colors(c, x, cmap, colored_nodes)

    if pos is None:
        pos = calc_node_pos(G, layout_algorithm)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[node_colors[d] for d in colored_nodes],
        nodelist=colored_nodes,
        ax=ax,
        **draw_nodes_kwd
    )
    if nodes is not None:
        nodes.set_zorder(3)
        nodes.set_edgecolor([node_edge_colors[d] for d in colored_nodes])

    # residuals (small squares)
    draw_nodes_kwd_residual = dict(draw_nodes_kwd)
    draw_nodes_kwd_residual["node_size"] = 0.1 * draw_nodes_kwd.get("node_size", 100)
    nodes_res = nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#efefef",
        nodelist=residuals,
        node_shape="s",
        ax=ax,
        **draw_nodes_kwd_residual
    )
    if nodes_res is not None:
        nodes_res.set_zorder(1)
        nodes_res.set_edgecolor("#4d4d4d")

    if draw_edge:
        nx.draw_networkx_edges(G.subgraph(colored_nodes + residuals), pos, ax=ax, **draw_edges_kwd)

    if font_size > 0:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, **draw_labels_kwd)

    ax.axis("off")
    return ax, pos


def draw_interactive(G, c, x, hover_text=None, node_size=10.0, pos=None, cmap=None):
    """Plotly interactive scatter; colors ramped by coreness inside groups."""
    # choose which nodes to color (all non-residuals)
    colored_nodes = [d for d in G.nodes() if (c[d] is not None) and (x[d] is not None)]
    node_colors, node_edge_colors = set_node_colors(c, x, cmap, colored_nodes)

    if pos is None:
        pos = nx.spring_layout(G)

    nodelist = list(G.nodes())
    group_ids = [c.get(d, "residual") for d in nodelist]
    coreness = [x.get(d, None) for d in nodelist]
    node_size_list = [(x[d] + 1) if x.get(d) is not None else 0.5 for d in nodelist]

    pos_x = [pos[d][0] for d in nodelist]
    pos_y = [pos[d][1] for d in nodelist]
    df = pd.DataFrame(
        {
            "x": pos_x,
            "y": pos_y,
            "name": nodelist,
            "group_id": group_ids,
            "coreness": coreness,
            "node_size": node_size_list,
        }
    )
    df["marker"] = df["group_id"].apply(lambda s: "circle" if s != "residual" else "square")
    df["hovertext"] = df.apply(
        lambda s: (
            f"Node {s['name']}" if hover_text is None else hover_text.get(s["name"], "")
        ) + f"<br>Group: {s['group_id']}<br>Coreness: {s['coreness']}",
        axis=1,
    )

    # map plotly color arrays
    color_arr = [node_colors[d] for d in nodelist]
    edge_arr = [node_edge_colors[d] for d in nodelist]

    fig = go.Figure(
        data=go.Scatter(
            x=df["x"],
            y=df["y"],
            marker_size=df["node_size"],
            marker_symbol=df["marker"],
            hovertext=df["hovertext"],
            hoverlabel=dict(namelength=0),
            hovertemplate="%{hovertext}",
            mode="markers",
            marker=dict(
                color=color_arr,
                sizeref=1.0 / node_size,
                line=dict(color=edge_arr, width=1),
            ),
        ),
    )
    fig.update_layout(autosize=False, width=800, height=800, template="plotly_white")
    return fig



