from contextlib import ContextDecorator
from typing import List, Tuple, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle
from tqdm.auto import tqdm

from modules import utils, model, train, paths

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

def run_ACDC_optimized(
    model: nn.Module, 
    tau: float, 
    data_loader,
    device: torch.device
):
    """
    Runs the ACDC algorithm to find a circuit in the model, evaluated on a full dataset.
    This version is optimized by batching patch evaluations and is robust to input tensor dimensionality.
    """
    model.eval()
    computation_graph = utils.ComputationalGraph(model)
    
    # --- 1. Pre-computation: Cache all necessary activations ---
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
        if not parents:
            continue

        total_kl_for_edges = {parent: 0.0 for parent in parents}
        
        dst_layer, dst_type = computation_graph._get_node_info(dest_node)
        if dst_type == "head":
            module_to_hook_name = dest_node.rsplit('.', 1)[0]
            module_to_hook = model.get_submodule(module_to_hook_name)
        else:
            module_to_hook = computation_graph.nodes[dest_node]

        for i in range(len(clean_samples_list)):
            clean_samples = clean_samples_list[i]
            clean_activations = clean_acts_list[i]
            corrupted_activations_list = corr_acts_list[i]
            clean_logprobs = clean_logprobs_list[i]
            
            for corrupted_activations in corrupted_activations_list:
                num_parents = len(parents)
                batch_size = clean_samples.shape[0]
                
                expanded_view = clean_samples.unsqueeze(0).expand(num_parents, *([-1] * clean_samples.dim()))
                
                batched_clean_samples = expanded_view.reshape(-1, *clean_samples.shape[1:])

                clean_src_contributions = []
                corrupted_src_contributions = []
                
                for src_node in parents:
                    clean_src_act_raw = clean_activations[src_node]
                    corrupted_src_act_raw = corrupted_activations[src_node]
                    projection_layer = computation_graph.get_projection_for_source(src_node).to(device, dtype=clean_src_act_raw.dtype)
                    clean_src_contributions.append(projection_layer(clean_src_act_raw))
                    corrupted_src_contributions.append(projection_layer(corrupted_src_act_raw))

                batched_clean_contrib = torch.cat(clean_src_contributions, dim=0)
                batched_corrupted_contrib = torch.cat(corrupted_src_contributions, dim=0)
                
                with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                    with PatchInput(module_to_hook, batched_clean_contrib, batched_corrupted_contrib):
                        patched_logits, _ = model(batched_clean_samples)
                
                patched_logprobs = F.log_softmax(patched_logits, dim=-1)
                
                logprobs_expanded_view = clean_logprobs.unsqueeze(0).expand(num_parents, *([-1] * clean_logprobs.dim()))
                batched_clean_logprobs = logprobs_expanded_view.reshape(-1, *clean_logprobs.shape[1:])
                
                kl_per_sample = F.kl_div(patched_logprobs, batched_clean_logprobs, reduction='none', log_target=True).sum(dim=-1)
                kl_per_parent_and_sample = kl_per_sample.view(num_parents, batch_size)
                avg_kl_for_batch_corruption = kl_per_parent_and_sample.mean(dim=1)
                
                for j, src_node in enumerate(parents):
                    total_kl_for_edges[src_node] += avg_kl_for_batch_corruption[j].item()

        num_corruptions = len(corr_acts_list[0]) if corr_acts_list else 0
        num_batches = len(clean_samples_list)
        
        for src_node in parents:
            edge_to_test = (src_node, dest_node)
            if edge_to_test not in circuit_edges:
                continue

            avg_patched_kl_for_edge = total_kl_for_edges[src_node] / (num_batches * num_corruptions)
            
            if avg_patched_kl_for_edge - current_circuit_kl < tau:
                circuit_edges.remove(edge_to_test)
                current_circuit_kl = avg_patched_kl_for_edge
                pbar_nodes.set_postfix(pruned=f"{len(computation_graph.edges) - len(circuit_edges)}", kl=f"{current_circuit_kl:.4f}")

    print("\nACDC finished.")
    print(f"Discovered circuit with {len(circuit_edges)} edges (out of {len(computation_graph.edges)}).")
    return circuit_edges
    

def get_accuracy_on_coarse_labels(model, dataloader, device, coarse_labels, coarse_model=False) -> float:
    fine_label_to_coarse = {fl: cli for cli, cl in enumerate(coarse_labels.values()) for fl in cl}
    coarse_to_name = {i: cl for i, cl in enumerate(coarse_labels.keys())}

    model.eval()
    correct, total = 0.0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.cpu()  # Move labels to CPU for in-place apply_
            labels.apply_(lambda x: fine_label_to_coarse.get(x, x))
            labels = labels.to(device, non_blocking=True)  # Move back to device

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs, _ = model(images)

            _, predicted = outputs.max(1)
            if not coarse_model:
                predicted = predicted.cpu()
                predicted.apply_(lambda x: fine_label_to_coarse.get(x, x))
                predicted = predicted.to(device)

            batch_correct = predicted.eq(labels).sum().item()
            batch_total = labels.size(0)

            correct += batch_correct
            total += batch_total

    epoch_acc = 100.0 * correct / total
    return epoch_acc

def load_checkpoint(model, path: str, device):
    """Loads training state from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    model_state_dict = checkpoint['state_dict']

    if isinstance(model, torch.nn.DataParallel):
        model.load_state_dict(model_state_dict)
    else:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

def test_taus(circuits: Set[str | Set[Tuple[str, str]]], dataloader, coarse_labels, config, device, base_model="vit1.pth"):


    import time

    # test original model:
    vit = model.ViT(config).to(device)
    load_checkpoint(vit, paths.chekpoints / base_model, device)
    orginal_model_acc = get_accuracy_on_coarse_labels(vit, dataloader, device, coarse_labels, coarse_model=False)
    print(f"Original model accuracy: {orginal_model_acc}")

    warmup_iters = 3
    test_iters = 10

    for tau, circuit in circuits.items():
        vit = model.ViT(config).to(device)
        print("\nTesting model with tau", tau)
        load_checkpoint(vit, paths.chekpoints / base_model, device)
        vit.retain_circuit(circuit)

        # Warm-up
        for _ in range(warmup_iters):
            _ = get_accuracy_on_coarse_labels(vit, dataloader, device, coarse_labels, coarse_model=False)

        # Timing
        total_time = 0.0
        for _ in range(test_iters):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                acc = get_accuracy_on_coarse_labels(vit, dataloader, device, coarse_labels, coarse_model=False)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to sec
            else:
                start = time.time()
                acc = get_accuracy_on_coarse_labels(vit, dataloader, device, coarse_labels, coarse_model=False)
                end = time.time()
                elapsed_time = end - start

            total_time += elapsed_time

        avg_time = total_time / test_iters
        print("Accuracy:", acc, "| Avg Time over", test_iters, "runs:", avg_time, "seconds")
