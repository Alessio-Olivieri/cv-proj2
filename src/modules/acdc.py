import torch
import torch.nn as nn
from contextlib import ContextDecorator
from torch.nn.modules.module import RemovableHandle
from typing import Callable, Literal, List, Tuple
from collections import defaultdict

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
                return (block_idx, "head")
            elif component_type == "mlp":
                return (block_idx, "mlp")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse node name: {node_name} - {e}")
        return (None, None)


    def _build_graph(self):
        # 1. Define and add nodes
        self.nodes["embedding"] = self.model.get_submodule("embedding.final_output")
        self.nodes["classifier"] = self.model.get_submodule("classifier")

        for block_idx in range(self.num_hidden_layers):
            mlp_name = f"encoder.blocks.{block_idx}.mlp.final_output"
            self.nodes[mlp_name] = self.model.get_submodule(mlp_name)

            for head_idx in range(self.num_attention_heads):
                head_name = f"encoder.blocks.{block_idx}.attention.heads.{head_idx}.final_output"
                self.nodes[head_name] = self.model.get_submodule(head_name)

        # 2. Build Edges based on ACDC's "potential influence" principle
        #    (output of an earlier component can influence any later component via residual stream)

        node_names_in_order = ["embedding"]
        for block_idx in range(self.num_hidden_layers):
            for head_idx in range(self.num_attention_heads):
                node_names_in_order.append(f"encoder.blocks.{block_idx}.attention.heads.{head_idx}")
            node_names_in_order.append(f"encoder.blocks.{block_idx}.mlp.{block_idx}") 

        all_potential_source_nodes = [n for n in self.nodes if n != "classifier"]


        for src_name in all_potential_source_nodes:
            src_layer, src_type = self._get_node_info(src_name)

            for dst_name in self.nodes:
                if src_name == dst_name:
                    continue

                dst_layer, dst_type = self._get_node_info(dst_name)

                # Rule: src can connect to dst if dst is conceptually "after" src
                # or if it's a head->MLP connection within the same block.
                can_connect = False
                if dst_name == "classifier" and src_name != "classifier": # Anything can connect to classifier
                    can_connect = True
                elif src_type == "embedding": # Embedding connects to all heads/MLPs
                    if dst_type in ["head", "mlp"]:
                        can_connect = True
                elif src_layer < dst_layer: # Connects to any component in a strictly later layer
                    can_connect = True
                elif src_layer == dst_layer:
                    if src_type == "head" and dst_type == "mlp":
                        can_connect = True

                if can_connect:
                    self.edges[src_name].append(dst_name)

        # Ensure all edges lists are unique (though the logic above should mostly handle it)
        for src_name in self.edges:
            self.edges[src_name] = sorted(list(set(self.edges[src_name])), key=self._get_node_info)


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
                if src != "classifier": # Classifier has no outgoing edges
                    output.append(f"{src} → (No outgoing edges defined in this graph)")
                continue

            output.append(f"{src} →")
            dst_list = self.edges[src]
            for i, dst in enumerate(dst_list):
                prefix = "   ├─" if i < len(dst_list) - 1 else "   └─"
                output.append(f"{prefix} {dst}")
        return "\n".join(output)


