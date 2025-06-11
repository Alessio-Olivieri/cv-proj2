from contextlib import ContextDecorator
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle
from tqdm.auto import tqdm

from modules import utils


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


class PatchInput(ContextDecorator):
    def __init__(self, module: Tuple[nn.Module, str], clean_src_act, corrupted_src_act):
        self.module = module
        self.clean_src_act = clean_src_act
        self.corrupted_src_act = corrupted_src_act
        self._hook_handle = None
    
    def patch_module(self, module, input, output):
        residual_stream = input[0]
        residual_stream = residual_stream - self.clean_src_act + self.corrupted_src_act
        return residual_stream, input[1]

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
    

class SaveActivation(ContextDecorator):
    def __init(self, module: nn.Module):
        self.module: nn.Module = module
        self.hook_handle: RemovableHandle = None
        self.activations: torch.Tensor = None

    def save_output(self, module, input, output):
        self.activation = output.detach().copy()

    def __enter__(self):
        self.hook_handle = self.module.register_forward_hook(self.save_output)
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
    
    # Get a single batch of clean and corrupted data
    clean_batch, corrupted_batches = next(iter(data_loader))

    all_nodes = list(computation_graph.nodes.values())
    print(all_nodes)
    
    # Cache clean activations and get the clean model output
    with SaveActivations(all_nodes) as sa:
        with torch.no_grad():
            clean_logits = model(clean_batch)
    clean_activations = sa.get_activations()
    clean_logprobs = F.log_softmax(clean_logits, dim=-1)

    # Cache activations for all corrupted batches
    corrupted_activations_list = []
    for corrupted_batch in tqdm(corrupted_batches, desc="Caching corrupted batches"):
        with SaveActivations(all_nodes) as sa:
            with torch.no_grad():
                model(corrupted_batch)
            corrupted_activations_list.append(sa.get_activations())
    
    print("Activation caching complete.")

    # --- 2. ACDC Algorithm: Iterative Pruning ---
    print(f"\nStep 2: Running ACDC with tau = {tau}...")

    # Start with the full graph
    circuit_edges = set(computation_graph.edges)
    
    # The performance (KL divergence) of the current circuit.
    # We start with the full model, whose KL divergence against itself is 0.
    current_circuit_kl = 0.0

    # Iterate through nodes in reverse topological order (from output to input)
    nodes_in_order = computation_graph.get_reverse_topological_sort()
    
    pbar_nodes = tqdm(nodes_in_order, desc="Nodes")
    for dest_node in pbar_nodes:
        # Iterate through all parents of the current destination node
        parents = computation_graph.get_parents(dest_node.name)
        
        for src_node in parents:
            edge_to_test = (src_node.name, dest_node.name)

            # Skip if edge has already been pruned by an earlier step
            if edge_to_test not in circuit_edges:
                continue

            # --- Test the importance of the edge ---
            # Measure the average KL divergence when this single edge is patched
            # from its clean activation to its corrupted activation.
            total_patched_kl = 0.0
            for corrupted_activations in corrupted_activations_list:
                clean_src_act = clean_activations[src_node.name]
                corrupted_src_act = corrupted_activations[src_node.name]
                
                # The PatchInput hook correctly isolates the effect of the single edge
                # by doing: new_input = old_input - clean_contribution + corrupted_contribution
                with utils.PatchInput(dest_node.module, clean_src_act, corrupted_src_act):
                    with torch.no_grad():
                        # A full forward pass is needed because the patch can affect all subsequent layers
                        patched_logits = model(clean_batch)
                
                patched_logprobs = F.log_softmax(patched_logits, dim=-1)
                # Use 'batchmean' for KL divergence as it's more stable than 'mean'
                kl = F.kl_div(patched_logprobs, clean_logprobs, reduction='batchmean', log_target=True)
                total_patched_kl += kl.item()
            
            avg_patched_kl = total_patched_kl / len(corrupted_batches)

            # --- The ACDC pruning decision ---
            # If the performance drop (increase in KL) is less than tau, prune the edge.
            if avg_patched_kl - current_circuit_kl < tau:
                # This edge is considered unimportant.
                circuit_edges.remove(edge_to_test)
                
                # The new baseline performance is the one with this edge patched.
                # This is the greedy part of the algorithm.
                current_circuit_kl = avg_patched_kl
    
    print("\nACDC finished.")
    print(f"Discovered circuit with {len(circuit_edges)} edges (out of {len(computation_graph.edges)}).")
    return circuit_edges