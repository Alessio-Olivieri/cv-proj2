from contextlib import ContextDecorator
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import RemovableHandle

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

def run_ACDC(model, tau, data_loader):

    computation_graph = utils.ComputationalGraph(model)

    def test_edge(src: Tuple[nn.Module, str], dst: Tuple[nn.Module, str]):
        """
        Tests the edge from the good class with all the bad classes edges.
        The resulting kl is the average of the kl divergences
        """
        clean_src_act = clean_activations[src]
        corrupted_src_act = corrupted_activations[src]
        with PatchInput(dst, clean_src_act, corrupted_src_act):
            logits = model(clean_batch)     
        kl_divergence = F.kl_div(logits, clean_logits)
        return kl_divergence
    
    model.eval()
    # if data_loader is the validation: 2900 total samples * 6 contrastive samples * num of edges to test
    for corrupted_batches, clean_batch in data_loader:
        corrupted_activations = []
        with SaveActivations(computation_graph.nodes) as ctx:
            with torch.no_grad():
                for corrupted_batch in corrupted_batches:
                    model(corrupted_batch)
                    corrupted_activations.append(ctx.activations)
                clean_logits = model(clean_batch)
                clean_activations = ctx.activations
    
                for dest in computation_graph.nodes: # they should be reverse order
                    for src in computation_graph.get_incoming_edges(src): #TODO get_incoming_edges
                        kl_divergence = test_edge(src, dest)
