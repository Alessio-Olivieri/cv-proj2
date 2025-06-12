from contextlib import ContextDecorator
from typing import List, Tuple, Dict, AnyStr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle
from tqdm.auto import tqdm

from modules import utils


class PatchInput(ContextDecorator):
    def __init__(self, module: Tuple[nn.Module, str], clean_src_act, corrupted_src_act, verbose=False):
        self.module = module
        self.clean_src_act = clean_src_act
        self.corrupted_src_act = corrupted_src_act
        self._hook_handle = None
        self.verbose = verbose
    
    def patch_module(self, module, input):
        residual_stream = input[0]
        if self.verbose:
            print("patching", module.name, ":")
            print("residual stream(input[0]):", residual_stream.shape)
            print("clean_src_act:", self.clean_src_act.shape)
            print("corrupted_src_act:", self.corrupted_src_act.shape)
        residual_stream = residual_stream - self.clean_src_act + self.corrupted_src_act
        if len(input) > 1:
            return residual_stream, input[1]
        else: return residual_stream

    def __enter__(self):
        self._hook_handle = self.module.register_forward_pre_hook(self.patch_module)

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
        self.activations: Dict[AnyStr | Tensor] = {}
        self.verbose = verbose

    # def get_activations(self):
    #     return {module.name:activation for module, activation in zip(self.modules, self.activations)}

    def save_outputs(self, module, input, output):
        self.activations[module.name] = output.detach().clone()
        if self.verbose:
            print(f"Saving output from {module.__class__.__name__}: {module.name}")
            print(f"ouptut is ot type:", type(output))
            print(f"output shape:", output.shape)
        
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

from contextlib import ContextDecorator
from typing import List, Tuple, Dict, AnyStr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle
from tqdm.auto import tqdm

from modules import utils


class PatchInput(ContextDecorator):
    def __init__(self, module: Tuple[nn.Module, str], clean_src_act, corrupted_src_act, verbose=False):
        self.module = module
        self.clean_src_act = clean_src_act
        self.corrupted_src_act = corrupted_src_act
        self._hook_handle = None
        self.verbose = verbose
    
    def patch_module(self, module, input):
        residual_stream = input[0]
        if self.verbose:
            print("patching", module.name, ":")
            print("residual stream(input[0]):", residual_stream.shape)
            print("clean_src_act:", self.clean_src_act.shape)
            print("corrupted_src_act:", self.corrupted_src_act.shape)
        residual_stream = residual_stream - self.clean_src_act + self.corrupted_src_act
        if len(input) > 1:
            return residual_stream, input[1]
        else: return residual_stream

    def __enter__(self):
        self._hook_handle = self.module.register_forward_pre_hook(self.patch_module)

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
        self.activations: Dict[AnyStr | Tensor] = {}
        self.verbose = verbose

    # def get_activations(self):
    #     return {module.name:activation for module, activation in zip(self.modules, self.activations)}

    def save_outputs(self, module, input, output):
        self.activations[module.name] = output.detach().clone()
        if self.verbose:
            print(f"Saving output from {module.__class__.__name__}: {module.name}")
            print(f"ouptut is ot type:", type(output))
            print(f"output shape:", output.shape)
        
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


def run_ACDC(
    model: nn.Module, 
    tau: float, 
    data_loader, # Should yield (clean_batch, [corrupted_batch_1, corrupted_batch_2, ...])
):
    """
    Runs the ACDC algorithm to find a circuit in the model.

    This implementation is adapted for a coarse-grained classification setting where
    each clean sample is contrasted against multiple "bad class" samples.

    Args:
        model: The model to find the circuit in.
        tau: The threshold for pruning. If the increase in KL divergence from pruning
             an edge is less than tau, the edge is removed.
        data_loader: A data loader that yields a tuple containing one clean batch and
                     a list of corrupted batches (one for each contrastive class).
                     This function only uses the first yielded batch.
        computation_graph: An object representing the model's computational graph,
                           with methods to get nodes and their connectivity.

    Returns:
        A set of tuples representing the edges of the discovered circuit.
    """
    model.eval()
    computation_graph = utils.ComputationalGraph(model)
    # --- 1. Pre-computation: Cache all necessary activations ---
    print("Step 1: Caching clean and corrupted activations...")
    
    clean_batch, corrupted_batches = next(iter(data_loader))

    all_nodes = list(computation_graph.nodes.values())
    
    with SaveActivations(all_nodes) as sa:
        with torch.no_grad():
            clean_samples, clean_labels = clean_batch
            clean_logits, _ = model(clean_samples)
    clean_activations = sa.activations
    clean_logprobs = F.log_softmax(clean_logits, dim=-1)
 
    corrupted_activations_list = []
    for corrupted_batch in tqdm(corrupted_batches, desc="Caching corrupted batches"):
        with SaveActivations(all_nodes) as sa:
            with torch.no_grad():
                corrupted_samples, corrupted_labels = corrupted_batch
                model(corrupted_samples)
            corrupted_activations_list.append(sa.activations)
    
    print("Activation caching complete.")

    print(f"\nStep 2: Running ACDC with tau = {tau}...")

    circuit_edges = set(computation_graph.edges)
    

    current_circuit_kl = 0.0

    nodes_in_order = computation_graph.get_reverse_topological_sort()
    
    pbar_nodes = tqdm(nodes_in_order, desc="Nodes")
    for dest_node in pbar_nodes:
        parents = computation_graph.get_parents(dest_node)
        # print("Dest node: ",dest_node)
        # print(parents)
        
        for src_node in parents:
            edge_to_test = (src_node, dest_node)

            if edge_to_test not in circuit_edges:
                continue

            total_patched_kl = 0.0
            for corrupted_activations in corrupted_activations_list:
                clean_src_act = clean_activations[src_node]
                corrupted_src_act = corrupted_activations[src_node]
                projection_layer = computation_graph.get_projection_for_source(src_node)
                clean_src_contribution = projection_layer(clean_src_act)
                corrupted_src_contribution = projection_layer(corrupted_src_act)
                dst_layer, dst_type = computation_graph._get_node_info(dest_node)
            
                if dst_type == "head":
                    # If the destination is a head, we don't hook its final_output.
                    # We hook the AttentionHead module itself, which takes d_model as input.
                    # e.g., '...heads.0.final_output' -> hook '...heads.0'
                    module_to_hook_name = dest_node.rsplit('.', 1)[0] 
                    module_to_hook = model.get_submodule(module_to_hook_name)
                else:
                    # For MLPs and other components that sit directly on the residual stream,
                    # hooking their input is correct. We hook the `final_output` module.
                    module_to_hook = computation_graph.nodes[dest_node]

                # print("testing:", edge_to_test)
                with PatchInput(module_to_hook, clean_src_contribution, corrupted_src_contribution):
                    with torch.no_grad():
                        patched_logits, _ = model(clean_samples)
                
                patched_logprobs = F.log_softmax(patched_logits, dim=-1)
                kl = F.kl_div(patched_logprobs, clean_logprobs, reduction='batchmean', log_target=True)
                total_patched_kl += kl.item()
            
            avg_patched_kl = total_patched_kl / len(corrupted_batches)

            if avg_patched_kl - current_circuit_kl < tau:
                circuit_edges.remove(edge_to_test)
                current_circuit_kl = avg_patched_kl
    
    print("\nACDC finished.")
    print(f"Discovered circuit with {len(circuit_edges)} edges (out of {len(computation_graph.edges)}).")
    return circuit_edges