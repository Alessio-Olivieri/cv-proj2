from contextlib import ContextDecorator
from typing import List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle
from tqdm.auto import tqdm

from modules import utils

# ... (PatchInput and SaveActivations classes remain the same) ...
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
            return (patched_residual_stream,)

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


def run_ACDC(
    model: nn.Module, 
    tau: float, 
    data_loader,
    device: torch.device
):
    """
    Runs the ACDC algorithm to find a circuit in the model.
    This version is memory-efficient and processes the dataset on-the-fly
    in batches, avoiding caching all activations at once.
    """
    model.eval()
    computation_graph = utils.ComputationalGraph(model)
    all_nodes_to_hook = list(computation_graph.nodes.values())

    print(f"Step 1: Running ACDC with tau = {tau} in batched mode...")
    
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
            
            # --- We will accumulate KL divergence over all batches for this edge ---
            total_kl_for_edge = 0.0
            num_batches = 0
            
            # Create a new progress bar for iterating through the dataset for each edge
            pbar_batch = tqdm(data_loader, desc=f"Testing edge {src_node[:15]}... -> {dest_node[:15]}...", leave=False)
            
            for clean_batch, corrupted_batches in pbar_batch:
                num_batches += 1
                clean_samples, _ = clean_batch
                clean_samples = clean_samples.to(device, non_blocking=True)

                # --- Step A: Get clean activations and logprobs for the current batch ---
                with torch.no_grad(), SaveActivations(all_nodes_to_hook) as sa, torch.amp.autocast(device_type=device.type):
                    clean_logits, _ = model(clean_samples)
                clean_activations = sa.activations
                clean_logprobs = F.log_softmax(clean_logits, dim=-1)

                # --- Step B: Average patched KL over all corruptions for this batch ---
                total_patched_kl_for_batch = 0.0
                for corrupted_batch in corrupted_batches:
                    corrupted_samples, _ = corrupted_batch
                    corrupted_samples = corrupted_samples.to(device, non_blocking=True)
                    
                    # Get corrupted activations for this specific corruption
                    with torch.no_grad(), SaveActivations(all_nodes_to_hook) as sa_corr, torch.amp.autocast(device_type=device.type):
                        model(corrupted_samples)
                    corrupted_activations = sa_corr.activations

                    # --- Get projected contributions (this logic is crucial and correct) ---
                    clean_src_act_raw = clean_activations[src_node]
                    corrupted_src_act_raw = corrupted_activations[src_node]
                    projection_layer = computation_graph.get_projection_for_source(src_node).to(device, dtype=clean_src_act_raw.dtype)
                    clean_src_contribution = projection_layer(clean_src_act_raw)
                    corrupted_src_contribution = projection_layer(corrupted_src_act_raw)
                    
                    # Determine correct module to hook
                    _, dst_type = computation_graph._get_node_info(dest_node)
                    if dst_type == "head":
                        module_to_hook_name = dest_node.rsplit('.', 1)[0]
                        module_to_hook = model.get_submodule(module_to_hook_name)
                    else:
                        module_to_hook = computation_graph.nodes[dest_node]
                    
                    # Run patched forward pass
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                        with PatchInput(module_to_hook, clean_src_contribution, corrupted_src_contribution):
                            patched_logits, _ = model(clean_samples)
                    
                    patched_logprobs = F.log_softmax(patched_logits, dim=-1)
                    kl = F.kl_div(patched_logprobs, clean_logprobs, reduction='batchmean', log_target=True)
                    total_patched_kl_for_batch += kl.item()
                
                # Average KL over corruptions for this one clean batch
                avg_kl_for_batch = total_patched_kl_for_batch / len(corrupted_batches)
                total_kl_for_edge += avg_kl_for_batch

            # Average KL over all clean batches in the dataset
            avg_patched_kl_for_edge = total_kl_for_edge / num_batches

            # --- Pruning Decision ---
            if avg_patched_kl_for_edge - current_circuit_kl < tau:
                circuit_edges.remove(edge_to_test)
                current_circuit_kl = avg_patched_kl_for_edge
                pbar_nodes.set_postfix(pruned=f"{len(computation_graph.edges) - len(circuit_edges)}", kl=f"{current_circuit_kl:.4f}")
    
    print("\nACDC finished.")
    print(f"Discovered circuit with {len(circuit_edges)} edges (out of {len(computation_graph.edges)}).")
    return circuit_edges