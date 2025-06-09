from contextlib import ContextDecorator
from typing import List, Tuple

import torch
import torch.nn as nn
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


def test_edge(src, dst, good_input, bad_input):
