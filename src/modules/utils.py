import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import graphviz
import numpy as np
import pygraphviz as pgv
import torch
import torch.nn as nn
from contextlib import ContextDecorator
from IPython.display import display
from torch.nn.modules.module import RemovableHandle

class ReplaceActivations(ContextDecorator):
    def __init__(self, module: Tuple[nn.Module, str], new_activation):
        self.module = module
        self.new_activation = new_activation
        self._hook_handle = None
    
    def patch_module(self, module, input, output):
        return self.new_activation

    def __enter__(self):
        self._hook_handle = self.module.register_forward_hook(self.patch_module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handle:
            self._hook_handle.remove()

        return exc_type is None


class SaveActivations(ContextDecorator):
    def __init__(self, modules: List[nn.Module,], verbose=False):
        '''
        Args:
            modules: A list of tuples, where each tuple contains (module, name) pairs
                     - module: The nn.Module whose activations should be saved
                     - name: A string identifier for the module
            verbose: If True, prints debugging information (default: False)
        '''
        self.modules = modules
        self._hook_handles: List[RemovableHandle] = []
        self.activations: List[torch.Tensor] = []
        self.verbose = verbose

    def get_activations(self):
        return {module.name:activation for module, activation in zip(self.modules, self.activations)}

    def save_outputs(self, module, input, output):
        self.activations.append(output)
        if self.verbose:
            print(f"Saving output from {module.__class__.__name__}: {module.name}")
            print(f"ouptut is ot type:", type(output))
            print(f"output lenght:", len(output))
        
    def __enter__(self):
        if self.verbose: print("entering context")
        for module in self.modules:
            self._hook_handles.append(module.register_forward_hook(self.save_outputs))
            if self.verbose: print(f"hooking {module.__class__.__name__}: {module.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handles:
            for hook_handle in self._hook_handles:
                hook_handle.remove()
        
        # Handle exceptions gracefully
        return exc_type is None

class ComputationalGraph():
    def __init__(self, model: nn.Module, num_hidden_layers: int, num_attention_heads: int):
        self.model = model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.edges = defaultdict(list)
        self.nodes = {}  # Store node_name: module_instance
        self._build_graph()

    def _get_node_info(self, node_name):
        """Helper to get layer index and type for sorting and logic."""
        if node_name == "embedding":
            return (-1, "embedding") # Layer before 0
        if node_name == "classifier":
            return (self.num_hidden_layers, "classifier") # Layer after last encoder block

        parts = node_name.split('.')
        try:
            block_idx = int(parts[2])
            component_type = parts[3] # "attention" or "mlp"
            if component_type == "attention":
                # For sorting, we can just use block_idx and a sub-sort for heads vs mlp
                # If head_idx is needed for finer sorting: head_idx = int(parts[5])
                return (block_idx, "head")
            elif component_type == "mlp":
                return (block_idx, "mlp")
        except (IndexError, ValueError) as e:
            # Fallback or raise error if parsing is critical for all node types
            # For nodes not matching the pattern, give them a default high sort order or handle specifically
            # print(f"Warning: Could not parse node name for sorting: {node_name} - {e}")
            return (self.num_hidden_layers + 1, node_name) # Default sort last
        raise ValueError(f"Could not parse node name: {node_name}")


    def _build_graph(self):
        # 1. Define and add nodes
        # Assuming model.get_submodule correctly fetches these specific module instances
        self.nodes["embedding"] = self.model.get_submodule("embedding.final_output")
        self.nodes["classifier"] = self.model.get_submodule("classifier")

        for block_idx in range(self.num_hidden_layers):
            mlp_name = f"encoder.blocks.{block_idx}.mlp.final_output"
            self.nodes[mlp_name] = self.model.get_submodule(mlp_name)

            for head_idx in range(self.num_attention_heads):
                head_name = f"encoder.blocks.{block_idx}.attention.heads.{head_idx}.final_output"
                head = self.model.get_submodule(head_name)
                if head.pruned: continue
                self.nodes[head_name] = head
        
        # 2. Build Edges based on ACDC's "potential influence" principle
        all_potential_source_nodes = [n for n in self.nodes if n != "classifier"]

        for src_name in all_potential_source_nodes:
            src_layer, src_type = self._get_node_info(src_name)

            for dst_name in self.nodes:
                if src_name == dst_name:
                    continue

                dst_layer, dst_type = self._get_node_info(dst_name)

                can_connect = False
                if dst_name == "classifier" and src_name != "classifier":
                    can_connect = True
                elif src_type == "embedding":
                    if dst_type in ["head", "mlp"]:
                        can_connect = True
                elif src_layer < dst_layer:
                    can_connect = True
                elif src_layer == dst_layer:
                    if src_type == "head" and dst_type == "mlp": # Head in block i to MLP in block i
                        can_connect = True
                
                if can_connect:
                    self.edges[src_name].append(dst_name)

        for src_name in self.edges:
            self.edges[src_name] = sorted(list(set(self.edges[src_name])), key=self._get_node_info)
    
    # def prune_nodes(self, nodes_to_be_pruned: List["str"]):
    #     for prune_node in nodes_to_be_pruned:
    #         self.nodes.pop(prune_node)
    #         self.edges.pop(prune_node)
    #         block_id = int(prune_node.split(".")[2]) 
    #         self.edges["embedding"].remove(prune_node)
    #         if block_id <= 0: return
    #         previous_block_id = block_id - 1
    #         while True:
    #             block = "encoder.blocks."+str(previous_block_id)
    #             mlp = block+".mlp.final_output"
    #             if mlp in self.edges:
    #                 self.edges[mlp].remove(prune_node)
    #             for head_id in range(self.num_attention_heads):
    #                 head = block+".attention.heads."+str(head_id)+".final_output"
    #                 if head in self.edges:
    #                     self.edges[head].remove(prune_node)
    #             if previous_block_id == 0:
    #                 break
    #             previous_block_id -= 1




    def __str__(self):
        output = []
        output.append("Computational Graph Structure:")
        output.append(f"Model: {self.model.__class__.__name__}")
        output.append(f"Num Layers: {self.num_hidden_layers}, Num Heads/Layer: {self.num_attention_heads}")
        output.append(f"Total Nodes: {len(self.nodes)}")
        output.append(f"Total Edges: {sum(len(v) for v in self.edges.values())}")

        output.append("\nNodes (Name - Module Class):")
        sorted_nodes = sorted(list(self.nodes.keys()), key=self._get_node_info)
        for name in sorted_nodes:
            module = self.nodes[name]
            output.append(f"- {name} ({module.__class__.__name__})")

        output.append("\nEdges (Source → Destinations):")
        for src in sorted_nodes: 
            if src not in self.edges or not self.edges[src]:
                if src != "classifier":
                    output.append(f"{src} → (No outgoing edges defined in this graph)")
                continue

            output.append(f"{src} →")
            dst_list = self.edges[src]
            for i, dst in enumerate(dst_list):
                prefix = "   ├─" if i < len(dst_list) - 1 else "   └─"
                output.append(f"{prefix} {dst}")
        return "\n".join(output)
    
    def visualize(self):
        agrpah = show_computational_graph(self)
        # Get the DOT language string from the pygraphviz object
        dot_string = agrpah.to_string() 
        
        # Create a graphviz.Source object and display it
        # The engine used here (e.g., 'dot') should match what you expect from GRAPH_LAYOUT
        gv_source = graphviz.Source(dot_string, engine="dot") 
        display(gv_source)


# --- Graphviz Visualization Functions ---

def generate_random_color(colorscheme_str: str) -> str:
    """
    Generates a random color string using cmapy.
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    if cmapy is None: # Fallback if cmapy is not installed
        return generate_random_color_fallback()

    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex(cmapy.color(colorscheme_str, np.random.randint(0, 256), rgb_order=True))

def abbreviate_node_name(full_name: str) -> str:
    """Abbreviates node names for display in the graph."""
    if full_name == "embedding":
        return "Emb"
    if full_name == "classifier":
        return "Cls"

    parts = full_name.split('.')
    # Expected formats:
    # encoder.blocks.{block_idx}.mlp.final_output
    # encoder.blocks.{block_idx}.attention.heads.{head_idx}.final_output
    try:
        if parts[0] == "encoder" and parts[1] == "blocks":
            block_idx = parts[2] # Keep as string for B{block_idx}
            component_type = parts[3]
            if component_type == "mlp" and parts[4] == "final_output":
                return f"B{block_idx}M"
            elif component_type == "attention" and parts[4] == "heads":
                head_idx = parts[5] # Keep as string for H{head_idx}
                if parts[6] == "final_output":
                    return f"B{block_idx}H{head_idx}"
    except IndexError:
        pass # Will return full_name if parsing fails

    return full_name # Fallback for unparseable names


try:
    import cmapy
except ImportError:
    print("Warning: cmapy not found. 'generate_random_color' might fail. Install with: pip install cmapy")
    # Basic fallback for generate_random_color if cmapy is not available
    def generate_random_color_fallback():
        import random
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    cmapy = None # To avoid NameError if cmapy import fails later

def show_computational_graph(
    cg: ComputationalGraph,
    fname: Optional[Union[str, Path]] = None,
    colorscheme_name: str = "Pastel2",
    user_node_colors: Optional[Dict[str, str]] = None, # For user to override colors for abbrev. names
    penwidth: float = 1.5,
    layout: str = "dot",
    seed: Optional[int] = None,
    node_shape: str = "box",
    node_style: str = "filled,rounded",
    font_name: str = "Helvetica",
    edge_color_from_source: bool = True,
    default_edge_color: str = "#A9A9A9" # DarkGray, used if not edge_color_from_source
) -> pgv.AGraph:
    """
    Generates and optionally saves a Graphviz visualization of the ComputationalGraph.

    Args:
        cg: The ComputationalGraph instance.
        fname: Path to save the output image (e.g., "graph.png"). Also saves a ".gv" file.
               If None, graph is not saved to file.
        colorscheme_name: Name of the cmapy colorscheme to use for random node colors.
        user_node_colors: A dictionary mapping abbreviated node names to hex color strings.
        penwidth: Thickness of the edges.
        layout: Graphviz layout algorithm (e.g., "dot", "neato", "fdp").
        seed: Random seed for color generation (for reproducibility).
        node_shape: Shape of the nodes.
        node_style: Style of the nodes.
        font_name: Font for node labels.
        edge_color_from_source: If True, edge color is same as its source node.
        default_edge_color: Color for edges if edge_color_from_source is False.

    Returns:
        A pygraphviz.AGraph object.
    """
    g = pgv.AGraph(directed=True, bgcolor="transparent", overlap="false", splines="true", layout=layout)

    if seed is not None:
        np.random.seed(seed)

    # 1. Prepare node names and colors
    abbrev_map = {full: abbreviate_node_name(full) for full in cg.nodes.keys()}
    
    node_colors = {}
    if user_node_colors:
        node_colors.update(user_node_colors)

    # Assign colors to any abbreviated names not covered by user_node_colors
    # Sort full_names using _get_node_info for consistent color assignment if seed is used
    sorted_full_node_names_for_color = sorted(list(cg.nodes.keys()), key=cg._get_node_info)
    for full_name in sorted_full_node_names_for_color:
        abbrev = abbrev_map[full_name]
        if abbrev not in node_colors:
            node_colors[abbrev] = generate_random_color(colorscheme_name)

    # 2. Add all nodes to the graph
    # Iterate using the graph's sorting preference for potentially better layout stability
    sorted_full_node_names = sorted(list(cg.nodes.keys()), key=cg._get_node_info)
    for full_name in sorted_full_node_names:
        abbrev = abbrev_map[full_name]
        color = node_colors.get(abbrev, generate_random_color(colorscheme_name)) # Fallback
        
        g.add_node(
            abbrev,
            fillcolor=color,
            color="black", # Border color
            style=node_style,
            shape=node_shape,
            fontname=font_name,
        )

    # 3. Add edges
    # Iterate source nodes in sorted order
    sorted_src_full_names = sorted(list(cg.edges.keys()), key=cg._get_node_info)
    for src_full_name in sorted_src_full_names:
        src_abbrev = abbrev_map[src_full_name]
        src_color = node_colors.get(src_abbrev, "#000000") # Fallback color for edge

        # Destinations are already sorted in cg.edges by _get_node_info
        for dst_full_name in cg.edges[src_full_name]:
            dst_abbrev = abbrev_map[dst_full_name]
            
            # Ensure destination node exists (should be covered by step 2)
            if not g.has_node(dst_abbrev):
                # This should ideally not happen if step 2 added all nodes from cg.nodes
                print(f"Warning: Destination node '{dst_abbrev}' (from '{dst_full_name}') was not pre-added. Adding now.")
                dst_color = node_colors.get(dst_abbrev, generate_random_color(colorscheme_name))
                g.add_node(
                    dst_abbrev, fillcolor=dst_color, color="black",
                    style=node_style, shape=node_shape, fontname=font_name
                )
            
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
        
        gv_path = fname_path.with_suffix(fname_path.suffix + ".gv") # e.g., graph.png -> graph.png.gv
        if fname_path.suffix == ".gv": # If fname is already a .gv file
            gv_path = fname_path

        try:
            g.write(str(gv_path))
            print(f"Graphviz DOT file saved to: {gv_path}")
            if fname_path.suffix != ".gv":
                g.draw(str(fname_path), prog=layout)
                print(f"Graph image saved to: {fname_path}")
        except Exception as e:
            print(f"Error saving/drawing graph to {fname_path} (DOT to {gv_path}): {e}")
            print("Ensure Graphviz executables (e.g., 'dot') are in your system PATH.")
            print("You can try: sudo apt-get install graphviz (on Ubuntu/Debian)")
            print("Or: brew install graphviz (on macOS)")

    return g