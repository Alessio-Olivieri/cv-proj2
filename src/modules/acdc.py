from contextlib import ContextDecorator
from typing import List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle
from tqdm.auto import tqdm

from modules import utils

# PatchInput and SaveActivations classes remain the same, but I've included
# the small correction from our previous discussion for completeness.

class PatchInput(ContextDecorator):
    def __init__(self, module: nn.Module, clean_src_act, corrupted_src_act):
        self.module = module
        self.clean_src_act = clean_src_act
        self.corrupted_src_act = corrupted_src_act
        self._hook_handle = None
    
    def patch_module(self, module, input):
        residual_stream = input[0]
        patched_residual_stream = residual_stream - self.clean_src_act + self.corrupted_src_act
        
        if len(input) > 1:
            return (patched_residual_stream,) + input[1:]
        else:
            return (patched_residual_stream,) # Always return a tuple for pre-hooks

    def __enter__(self):
        self._hook_handle = self.module.register_forward_pre_hook(self.patch_module)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handle:
            self._hook_handle.remove()
        return exc_type is None

class SaveActivations(ContextDecorator):
    def __init__(self, modules: List[nn.Module], verbose=False):
        self.modules = modules
        self._hook_handles: List[RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.verbose = verbose
        
    def save_outputs(self, module, input, output):
        self.activations[module.name] = output.detach().clone()
        if self.verbose:
            print(f"Saving output from {module.__class__.__name__}: {module.name}")
        
    def __enter__(self):
        for module in self.modules:
            self._hook_handles.append(module.register_forward_hook(self.save_outputs))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handles:
            for hook_handle in self._hook_handles:
                hook_handle.remove()
        return exc_type is None

def cache_all_activations(
    model: nn.Module, 
    data_loader, 
    nodes_to_cache: List[nn.Module],
    device: torch.device
) -> Tuple[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]], List[torch.Tensor], List[torch.Tensor]]:
    """
    Pre-computes and caches all necessary clean and corrupted activations and logits.
    This is a performance optimization to avoid re-computing during the ACDC loop.
    
    Returns:
        - A list where each element is the clean_samples tensor for a batch.
        - A list where each element is the dictionary of clean activations for a batch.
        - A list of lists for corrupted activations. Outer list for batches, inner for corruptions.
        - A list where each element is the clean_logprobs tensor for a batch.
    """
    model.eval()
    all_clean_samples = []
    all_clean_activations = []
    all_corrupted_activations = []
    all_clean_logprobs = []

    pbar = tqdm(data_loader, desc="Caching all activations", leave=False)
    for clean_batch, corrupted_batches in pbar:
        clean_samples, _ = clean_batch
        clean_samples = clean_samples.to(device, non_blocking=True)
        all_clean_samples.append(clean_samples)

        # Cache clean activations and logits
        with torch.no_grad(), SaveActivations(nodes_to_cache) as sa, torch.amp.autocast(device_type=device.type):
            clean_logits, _ = model(clean_samples)
        
        all_clean_activations.append(sa.activations)
        all_clean_logprobs.append(F.log_softmax(clean_logits, dim=-1))

        # Cache corrupted activations
        corrupted_activations_for_batch = []
        for corrupted_batch in corrupted_batches:
            corrupted_samples, _ = corrupted_batch
            corrupted_samples = corrupted_samples.to(device, non_blocking=True)
            with torch.no_grad(), SaveActivations(nodes_to_cache) as sa_corr, torch.amp.autocast(device_type=device.type):
                model(corrupted_samples)
            corrupted_activations_for_batch.append(sa_corr.activations)
        all_corrupted_activations.append(corrupted_activations_for_batch)
    
    return all_clean_samples, all_clean_activations, all_corrupted_activations, all_clean_logprobs


def run_ACDC(
    model: nn.Module, 
    tau: float, 
    data_loader,
    device: torch.device
):
    """
    Runs the ACDC algorithm to find a circuit in the model, evaluated on a full dataset.
    """
    model.eval()
    computation_graph = utils.ComputationalGraph(model)
    
    # --- 1. Pre-computation: Cache all necessary activations on the correct device ---
    print("Step 1: Caching clean and corrupted activations...")
    all_nodes = list(computation_graph.nodes.values())
    
    clean_samples_list, clean_acts_list, corr_acts_list, clean_logprobs_list = cache_all_activations(
        model, data_loader, all_nodes, device
    )
    
    print("Activation caching complete.")
    print(f"\nStep 2: Running ACDC with tau = {tau}...")

    circuit_edges = set(computation_graph.edges)
    current_circuit_kl = 0.0
    nodes_in_order = computation_graph.get_reverse_topological_sort()
    
    pbar_nodes = tqdm(nodes_in_order, desc="Pruning Edges (Nodes)")
    for dest_node in pbar_nodes:
        parents = computation_graph.get_parents(dest_node)
        
        for src_node in parents:
            edge_to_test = (src_node, dest_node)
            if edge_to_test not in circuit_edges:
                continue

            # --- Calculate KL divergence for this edge averaged over the whole dataset ---
            total_kl_for_edge = 0.0
            for i in range(len(clean_samples_list)):
                clean_samples = clean_samples_list[i]
                clean_activations = clean_acts_list[i]
                corrupted_activations_list = corr_acts_list[i]
                clean_logprobs = clean_logprobs_list[i]

                total_patched_kl_for_batch = 0.0
                for corrupted_activations in corrupted_activations_list:
                    
                    # --- Get projected contributions (this logic is crucial and correct) ---
                    clean_src_act_raw = clean_activations[src_node]
                    corrupted_src_act_raw = corrupted_activations[src_node]
                    projection_layer = computation_graph.get_projection_for_source(src_node).to(device, dtype=clean_src_act_raw.dtype)
                    clean_src_contribution = projection_layer(clean_src_act_raw)
                    corrupted_src_contribution = projection_layer(corrupted_src_act_raw)
                    
                    # --- Determine correct module to hook ---
                    dst_layer, dst_type = computation_graph._get_node_info(dest_node)
                    if dst_type == "head":
                        module_to_hook_name = dest_node.rsplit('.', 1)[0]
                        module_to_hook = model.get_submodule(module_to_hook_name)
                    else:
                        module_to_hook = computation_graph.nodes[dest_node]
                    
                    # --- Run patched forward pass ---
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                        with PatchInput(module_to_hook, clean_src_contribution, corrupted_src_contribution):
                            patched_logits, _ = model(clean_samples)
                    
                    patched_logprobs = F.log_softmax(patched_logits, dim=-1)
                    kl = F.kl_div(patched_logprobs, clean_logprobs, reduction='batchmean', log_target=True)
                    total_patched_kl_for_batch += kl.item()
                
                # Average KL over corruptions for this one clean batch
                avg_kl_for_batch = total_patched_kl_for_batch / len(corrupted_activations_list)
                total_kl_for_edge += avg_kl_for_batch
            
            # Average KL over all clean batches in the dataset
            avg_patched_kl_for_edge = total_kl_for_edge / len(clean_samples_list)

            # --- Pruning Decision ---
            if avg_patched_kl_for_edge - current_circuit_kl < tau:
                circuit_edges.remove(edge_to_test)
                current_circuit_kl = avg_patched_kl_for_edge
                pbar_nodes.set_postfix(pruned=f"{len(computation_graph.edges) - len(circuit_edges)}", kl=f"{current_circuit_kl:.4f}")

    print("\nACDC finished.")
    print(f"Discovered circuit with {len(circuit_edges)} edges (out of {len(computation_graph.edges)}).")
    return circuit_edges