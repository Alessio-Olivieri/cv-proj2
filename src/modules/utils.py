import itertools
from pathlib import Path
from typing import Dict, List, Optional, Union

import graphviz
import numpy as np
import pygraphviz as pgv
from IPython.display import display

from modules import model

try:
    import cmapy
except ImportError:
    print("Warning: cmapy not found. 'generate_random_color' will use a basic fallback. Install with: pip install cmapy")
    def generate_random_color_fallback():
        import random
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    cmapy = None

def generate_random_color(colorscheme_str: str) -> str:
    """Generates a random color string using cmapy."""
    if cmapy is None:
        return generate_random_color_fallback()
    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
    return rgb2hex(cmapy.color(colorscheme_str, np.random.randint(0, 256), rgb_order=True))

def abbreviate_node_name(full_name: str) -> str:
    """Abbreviates node names for display in the graph."""
    if full_name == "embedding": return "Emb"
    if full_name == "classifier": return "Cls"
    try:
        parts = full_name.split('.')
        if parts[0] == "encoder" and parts[1] == "blocks":
            block_idx, component_type = parts[2], parts[3]
            if component_type == "mlp": return f"B{block_idx}M"
            elif component_type == "attention" and parts[4] == "heads":
                head_idx = parts[5]
                return f"B{block_idx}H{head_idx}"
    except IndexError: pass
    return full_name

# --- Rewritten ComputationalGraph Class ---

class ComputationalGraph():
    def __init__(self, model: model.ViT):
        self.model = model
        self.num_hidden_layers = model.config["num_hidden_layers"]
        self.num_attention_heads = model.config["num_attention_heads"]
        
        # Store edges as a list of (source, destination) tuples
        self.edges = []
        self.nodes = {}  # Store node_name: module_instance
        self._build_graph()

        # reverse nodes for acdc:
        self.ordered_nodes = sorted(
            list(self.nodes.keys()), 
            key=self._get_node_info, 
            reverse=True
        )        

    def _get_node_info(self, node_name):
        """Helper to get layer index and type for sorting and logic."""
        if node_name == "embedding":
            return (-1, "embedding")
        if node_name == "classifier":
            return (self.num_hidden_layers, "classifier")

        parts = node_name.split('.')
        try:
            block_idx = int(parts[2])
            component_type = parts[3]
            if component_type == "attention":
                return (block_idx, "head")
            elif component_type == "mlp":
                return (block_idx, "mlp")
        except (IndexError, ValueError):
            return (self.num_hidden_layers + 1, node_name)
        raise ValueError(f"Could not parse node name: {node_name}")


    def _build_graph(self):
        # 1. Define and add nodes
        # Note: In a real scenario, use try-except for get_submodule
        self.nodes["embedding"] = self.model.get_submodule("embedding.final_output")
        # self.nodes["classifier"] = self.model.get_submodule("classifier")

        for block_idx in range(self.num_hidden_layers):
            mlp_name = f"encoder.blocks.{block_idx}.mlp.final_output"
            self.nodes[mlp_name] = self.model.get_submodule(mlp_name)

            for head_idx in range(self.num_attention_heads):
                head_name = f"encoder.blocks.{block_idx}.attention.heads.{head_idx}.final_output"
                head = self.model.get_submodule(head_name)
                # Assuming a 'pruned' attribute exists on the module
                if hasattr(head, 'pruned') and head.pruned: continue
                self.nodes[head_name] = head
        
        # 2. Build Edges as (src, dst) tuples
        # Use a set to automatically handle potential duplicates before sorting
        edge_set = set()
        all_potential_source_nodes = [n for n in self.nodes if n != "classifier"]

        for src_name in all_potential_source_nodes:
            src_layer, src_type = self._get_node_info(src_name)

            for dst_name in self.nodes:
                if src_name == dst_name: continue

                dst_layer, dst_type = self._get_node_info(dst_name)
                
                can_connect = False
                # if dst_name == "classifier":
                #     can_connect = True
                if src_type == "embedding" and dst_type in ["head", "mlp"]:
                    can_connect = True
                elif src_layer < dst_layer:
                    can_connect = True
                elif src_layer == dst_layer and src_type == "head" and dst_type == "mlp":
                    can_connect = True
                
                if can_connect:
                    edge_set.add((src_name, dst_name))
        
        # Sort edges for deterministic order: primarily by source, secondarily by destination
        self.edges = sorted(
            list(edge_set), 
            key=lambda edge: (self._get_node_info(edge[0]), self._get_node_info(edge[1]))
        )

    def get_incoming_edges(self, dst_node:str):
        if dst_node not in self.nodes: raise ValueError("Wrong src node name")
        return [(src, dst) for src,dst in self.edges if dst==dst_node]

    def prune_nodes(self, nodes_to_prune: List[str]):
        """
        Removes specified nodes and any edges connected to them.
        """
        nodes_to_prune_set = set(nodes_to_prune)

        # Remove the nodes from the node dictionary
        self.nodes = {name: mod for name, mod in self.nodes.items() if name not in nodes_to_prune_set}
        
        # Rebuild the edge list, filtering out any edge involving a pruned node
        self.edges = [
            (src, dst) for src, dst in self.edges 
            if src not in nodes_to_prune_set and dst not in nodes_to_prune_set
        ]

    def __str__(self):
        output = [
            "Computational Graph Structure:",
            f"Model: {self.model.__class__.__name__}",
            f"Num Layers: {self.num_hidden_layers}, Num Heads/Layer: {self.num_attention_heads}",
            f"Total Nodes: {len(self.nodes)}",
            f"Total Edges: {len(self.edges)}",
            "\nNodes (Name - Module Class):"
        ]
        
        sorted_nodes = sorted(list(self.nodes.keys()), key=self._get_node_info)
        for name in sorted_nodes:
            module = self.nodes[name]
            output.append(f"- {name} ({module.__class__.__name__})")

        output.append("\nEdges (Source → Destinations):")
        # Group sorted edges by source for clean printing
        for src_name, group in itertools.groupby(self.edges, key=lambda edge: edge[0]):
            if src_name not in self.nodes: continue # Skip edges from pruned nodes
            
            # Since group is an iterator, we need to extract destinations
            dst_list = [dst for _, dst in group]
            
            output.append(f"{src_name} →")
            for i, dst_name in enumerate(dst_list):
                prefix = "   ├─" if i < len(dst_list) - 1 else "   └─"
                output.append(f"{prefix} {dst_name}")
        
        # Identify nodes with no outgoing edges
        nodes_with_edges = {edge[0] for edge in self.edges}
        for node_name in sorted_nodes:
            if node_name not in nodes_with_edges and node_name != "classifier":
                 output.append(f"{node_name} → (No outgoing edges)")
                
        return "\n".join(output)
    
    def visualize(self, **kwargs):
        """Convenience method to call the visualization function."""
        agraph = show_computational_graph(self, **kwargs)
        # In a Jupyter/IPython environment, this will display the graph
        try:
            dot_string = agraph.to_string()
            gv_source = graphviz.Source(dot_string, engine="dot")
            display(gv_source)
        except Exception as e:
            print(f"Could not render graph in this environment. Error: {e}")
            print("The pygraphviz.AGraph object was still returned.")


# --- Rewritten Visualization Function ---

def show_computational_graph(
    cg: ComputationalGraph,
    fname: Optional[Union[str, Path]] = None,
    colorscheme_name: str = "Pastel2",
    user_node_colors: Optional[Dict[str, str]] = None,
    penwidth: float = 1.5,
    layout: str = "dot",
    seed: Optional[int] = None,
    node_shape: str = "box",
    node_style: str = "filled,rounded",
    font_name: str = "Helvetica",
    edge_color_from_source: bool = True,
    default_edge_color: str = "#A9A9A9"
) -> pgv.AGraph:
    """
    Generates and optionally saves a Graphviz visualization of the ComputationalGraph.
    This version is adapted to work with an edge list of (src, dst) tuples.
    """
    g = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout=layout)

    if seed is not None:
        np.random.seed(seed)

    # 1. Prepare node names and colors
    abbrev_map = {full: abbreviate_node_name(full) for full in cg.nodes.keys()}
    node_colors = {}
    if user_node_colors:
        node_colors.update(user_node_colors)

    sorted_full_node_names_for_color = sorted(list(cg.nodes.keys()), key=cg._get_node_info)
    for full_name in sorted_full_node_names_for_color:
        abbrev = abbrev_map[full_name]
        if abbrev not in node_colors:
            node_colors[abbrev] = generate_random_color(colorscheme_name)

    # 2. Add all nodes to the graph
    sorted_full_node_names = sorted(list(cg.nodes.keys()), key=cg._get_node_info)
    for full_name in sorted_full_node_names:
        abbrev = abbrev_map[full_name]
        color = node_colors.get(abbrev, generate_random_color(colorscheme_name))
        g.add_node(
            abbrev, fillcolor=color, color="black",
            style=node_style, shape=node_shape, fontname=font_name
        )

    # 3. Add edges by iterating through the (src, dst) tuple list
    for src_full_name, dst_full_name in cg.edges:
        # Skip if a node in the edge tuple was pruned but the edge list wasn't updated
        if src_full_name not in abbrev_map or dst_full_name not in abbrev_map:
            continue
            
        src_abbrev = abbrev_map[src_full_name]
        dst_abbrev = abbrev_map[dst_full_name]
        
        src_color = node_colors.get(src_abbrev, "#000000")
        current_edge_color = src_color if edge_color_from_source else default_edge_color
        
        g.add_edge(
            src_abbrev,
            dst_abbrev,
            penwidth=str(penwidth),
            color=current_edge_color,
        )

    # 4. Save graph if fname is provided
    if fname:
        fname_path = Path(fname)
        fname_path.parent.mkdir(parents=True, exist_ok=True)
        gv_path = fname_path.with_suffix(".gv") if fname_path.suffix != ".gv" else fname_path

        try:
            g.write(str(gv_path))
            print(f"Graphviz DOT file saved to: {gv_path}")
            if fname_path.suffix != ".gv":
                g.draw(str(fname_path), prog=layout)
                print(f"Graph image saved to: {fname_path}")
        except Exception as e:
            print(f"Error saving/drawing graph: {e}\nEnsure Graphviz is installed and in your system's PATH.")

    return g