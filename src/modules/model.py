import torch
import math
import torch.nn as nn
from torch.nn import Conv2d
from typing import Callable, List, Tuple, Optional, Dict, Set
from jaxtyping import Float, Int
from torch import Tensor

class PatchEmbeddings(nn.Module):
    """Converts input images into patch embeddings.
    
    Args:
        config: Configuration dictionary containing:
            image_size (int): Input image size (assumed square)
            hidden_size (int): Embedding dimension size
            patch_size (int): Size of each image patch (square)
    
    Attributes:
        image_size: Input image dimension
        hidden_size: Embedding dimension size
        patch_size: Patch dimension
        num_patches: Total number of patches per image
        projection: Convolutional layer for patch projection
    """
    def __init__ (self, config) -> None:
        super().__init__()
        self.image_size: int = config["image_size"]
        self.hidden_size: int = config["hidden_size"]
        self.patch_size: int = config["patch_size"]
        self.num_patches: int = int((self.image_size/self.patch_size)**2)
        self.projection: Conv2d = Conv2d(
            in_channels=3, 
            out_channels=self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b n d"]:
        """Converts input images to patch embeddings.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Patch embeddings tensor of shape (batch, num_patches, hidden_size)
        """

        x = self.projection(x)  # (b, hidden_size, grid_size, grid_size)
        x = x.flatten(2).transpose(1, 2)  # (b, num_patches, hidden_size)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.pruned = False
    
    def forward(self, x):
        return x

class Embeddings(nn.Module):
    """Combines patch embeddings with class token and position embeddings.
    
    Args:
        config: Configuration dictionary (see PatchEmbeddings)
    
    Attributes:
        patch_embeddings: Patch embedding module
        cls_token: Learnable classification token
        position_embeddings: Learnable position embeddings
        dropout: Dropout layer
    """
    def __init__ (self, config) -> None:
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]) * 0.02)
        self.position_embeddings = nn.Parameter(
            torch.randn(self.patch_embeddings.num_patches + 1, config["hidden_size"])  * 0.02
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.final_output = Identity()


    def forward(self, x: Int[Tensor, "b c h w"]) -> Float[Tensor, "b n d"]:
        """Generates full input embeddings for transformer.
        
        Args:
            x: Input image tensor (batch, channels, height, width)
        
        Returns:
            Embedding tensor with shape (batch, num_patches+1, hidden_size)
        """
        x = self.patch_embeddings(x)  # (b, num_patches, hidden_size)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (b, 1, hidden_size)
        x = torch.cat([cls_token, x], dim=1)  # (b, num_patches+1, hidden_size)
        x += self.position_embeddings  # Add position embeddings
        x = self.dropout(x)
        x = self.final_output(x)
        return x
    

class AttentionHead(nn.Module):
    """Single attention head implementation.
    
    Args:
        hidden_size: Input dimension size
        attention_head_size: Dimension size for this head
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """
    def __init__(self, hidden_size: int, attention_head_size: int, dropout: float, bias: bool = True):
        super().__init__()
        self.attention_head_size = attention_head_size
        self.key = nn.Linear(hidden_size, attention_head_size, bias)
        self.query = nn.Linear(hidden_size, attention_head_size, bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias)
        self.dropout = nn.Dropout(dropout)
        self.final_output = Identity()
        self.pruned = False

    def forward(self, x: Float[Tensor, "b n d"]) -> Tuple[Float[Tensor, "b n d_head"], Float[Tensor, "b n n"]]:
        """Computes attention output and probabilities.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, hidden_size)
        
        Returns:
            attention_output: Output tensor (batch, seq_len, attention_head_size)
            attention_probs: Attention probabilities (batch, seq_len, seq_len)
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = self.final_output(attention_output)
        return attention_output, attention_probs
    

class MultiHeadAttention(nn.Module):
    """Multi-head attention module.
    
    Args:
        config: Configuration dictionary containing:
            hidden_size: Input dimension size
            num_attention_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            attention_probs_dropout_prob: Attention dropout probability
            hidden_dropout_prob: Output dropout probability
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([AttentionHead(self.hidden_size,
                                    self.attention_head_size,
                                    config["attention_probs_dropout_prob"],
                                    self.qkv_bias) for _ in range(self.num_attention_heads)
                                    ])

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.pruned_heads = set()

    def forward(self, x: Float[Tensor, "b n d"], output_attentions=False) -> Tuple[Float[Tensor, "b n d"], Optional[Float[Tensor, "b head n n"]]]:
        """Computes multi-head attention.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            output_attentions: Whether to return attention tensors
        
        Returns:
            attention_output: Output tensor (batch, seq_len, hidden_size)
            attention_probs: Optional tensor of attention probabilities
                            (batch, num_attention_heads, seq_len, seq_len)
        """
        # If all heads are pruned, return zeros
        if self.num_attention_heads == 0:
            output_shape = (x.shape[0], x.shape[1], self.hidden_size)
            return torch.zeros(output_shape, device=x.device, dtype=x.dtype), None
        
        active_attention_outputs, active_attention_probs = [], [] if output_attentions else None

        for i, head in enumerate(self.heads):
            if i in self.pruned_heads: continue
            
            # Compute only for active heads
            attention_output, attention_probs = head(x)
            active_attention_outputs.append(attention_output)
            if output_attentions:
                active_attention_probs.append(attention_probs)      

        attention_output = torch.cat(active_attention_outputs, dim=2)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output) 

        if not output_attentions:
            return attention_output, None
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in active_attention_outputs], dim=1)
            return attention_output, attention_probs
        
        
    def prune_heads(self, head_to_prune: List[int]):
        head_to_prune = set(head_to_prune) - self.pruned_heads
        if not head_to_prune: return
        for i in head_to_prune:
            self.heads[i].final_output.pruned = True

        self.pruned_heads.update(head_to_prune)
        self.num_attention_heads -= (len(head_to_prune))
        if self.num_attention_heads == 0: return
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        new_output_projection = nn.Linear(self.all_head_size, self.hidden_size)

        # 4. Copy the weights from the old projection layer to the new one
        original_indices = [i for i in range(len(self.heads)) if i not in self.pruned_heads]
        with torch.no_grad():
            # Slice the weight tensor to keep only the weights for active heads
            kept_weight_slices = [
                self.output_projection.weight.data[:, i * self.attention_head_size:(i + 1) * self.attention_head_size]
                for i in original_indices
            ]
            new_output_projection.weight.data = torch.cat(kept_weight_slices, dim=1)
            # Bias is not dependent on the input dimension, so it remains unchanged.
            new_output_projection.bias.data = self.output_projection.bias.data.clone()
        
        self.output_projection = new_output_projection


class MLP(nn.Module):
    """Transformer MLP (feed-forward) block.
    
    Args:
        config: Configuration dictionary containing:
            hidden_size: Input/output dimension size
            intermediate_size: Hidden layer dimension size
            hidden_dropout_prob: Dropout probability
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.dense1 = nn.Linear(self.hidden_size, config["intermediate_size"])
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config["intermediate_size"], self.hidden_size)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.final_output = Identity()
    
    def forward(self, x:Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        """Applies two-layer MLP with GELU activation.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
        
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        if self.final_output.pruned:
            output_shape = (x.shape[0], x.shape[1], self.hidden_size)
            return torch.zeros(output_shape, device=x.device, dtype=x.dtype) 
        else:
            x = self.activation(self.dense1(x))
            x = self.dropout(self.dense2(x))
            x = self.final_output(x)
        return x
        

class EncoderBlock(nn.Module):
    """Single transformer encoder block.
    
    Args:
        config: Configuration dictionary (see MultiHeadAttention and MLP)
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: Float[Tensor, "b n d"], output_attentions=False) -> Tuple[Float[Tensor, "b n d"], Optional[Float[Tensor, "b head n n"]]]:
        """Applies transformer block with residual connections.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            output_attentions: Whether to return attention tensors
        
        Returns:
            Output tensor (batch, seq_len, hidden_size)
            Attention probabilities (optional)
        """
        # Attention with residual
        attention_output, attention_probs = self.attention(self.norm1(x), output_attentions)
        attention_output += x
        
        # MLP with residual
        mlp_output = self.mlp(self.norm2(attention_output))
        output = mlp_output + attention_output
        
        return output, attention_probs
    

class Encoder(nn.Module):
    """Stack of transformer encoder blocks.
    
    Args:
        config: Configuration dictionary containing:
            num_hidden_layers: Number of encoder blocks
            Other parameters for EncoderBlock
    """
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = EncoderBlock(config)
            self.blocks.append(block)

    def forward(self, x:  Float[Tensor, "b n d"], output_attentions=False) -> Tuple[Float[Tensor, "b n d"], Optional[List[Float[Tensor, "b head n n"]]]]:
        """Sequentially applies encoder blocks.
        
        Args:
            x: Input embeddings (batch, seq_len, hidden_size)
            output_attentions: Whether to return attention tensors
        
        Returns:
            Output tensor (batch, seq_len, hidden_size)
            List of attention tensors (optional)
        """
        all_attentions = [] if output_attentions else None
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions)
            if output_attentions: 
                all_attentions.append(attention_probs)
        return x, all_attentions
    
class ViT(nn.Module):
    """Vision Transformer model.
    
    Args:
        config: Configuration dictionary containing:
            image_size: Input image size
            patch_size: Patch size
            hidden_size: Embedding dimension
            num_classes: Number of output classes
            Other parameters for submodules
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])        
        self.apply(self._init_weights)
        self._set_module_names() 

    def forward(self, x: Int[Tensor, "b c h w"], output_attentions=False) -> Tuple[Float[Tensor, "b n_classes"], Optional[List]]:
        """Forward pass for Vision Transformer.
        
        Args:
            x: Input images (batch, channels, height, width)
            output_attentions: Whether to return attention tensors
        
        Returns:
            logits: Classification logits (batch, num_classes)
            attentions: Optional list of attention tensors from all layers
        """
        x = self.embedding(x)  # (b, num_patches+1, hidden_size)
        x, all_attentions = self.encoder(x, output_attentions)  # (b, num_patches+1, hidden_size)
        # Use class token for classification
        x = self.classifier(x[:,0])  # (b, num_classes)
        return x, all_attentions

        
    def prune_heads(self, heads_to_prune: List[str]):
        heads_to_prune = set(heads_to_prune)

        to_be_pruned_per_block = [[] for block_id in range(len(self.encoder.blocks))]
        for head in heads_to_prune:
            block_id = int(head.split(".")[2])
            head_id = int(head.split(".")[5])
            to_be_pruned_per_block[block_id].append(head_id)

        for block_id, block in enumerate(self.encoder.blocks):
            block.attention.prune_heads(to_be_pruned_per_block[block_id])
    

    def prune_mlp(self, mlp_to_prune: List[str]):
        blocks_to_prune = {int(mlp.split(".")[2]) for mlp in mlp_to_prune}
        for block_id, block in enumerate(self.encoder.blocks):
            if block_id in blocks_to_prune:
                block.mlp.final_output.pruned = True


    def retain_circuit(self, circuit: Set[Tuple[str, str]]):
        """
        Prunes the model to keep only the components and edges specified in the circuit.
        Any head or MLP layer not mentioned in the circuit will be pruned.

        Args:
            circuit: A set of (source_name, destination_name) tuples representing the
                     edges of the circuit to keep.
        """
        num_hidden_layers = self.config["num_hidden_layers"]
        num_attention_heads = self.config["num_attention_heads"]

        # 1. Identify all potentially prunable components in the model
        all_mlp_nodes = {
            f"encoder.blocks.{block_idx}.mlp.final_output"
            for block_idx in range(num_hidden_layers)
        }
        all_head_nodes = {
            f"encoder.blocks.{block_idx}.attention.heads.{head_idx}.final_output"
            for block_idx in range(num_hidden_layers)
            for head_idx in range(num_attention_heads)
        }

        # 2. Identify which components are active (i.e., part of the circuit)
        active_nodes = set()
        for src, dst in circuit:
            active_nodes.add(src)
            active_nodes.add(dst)

        # An active component is one that appears in our active_nodes set.
        active_mlp_nodes = {node for node in active_nodes if "mlp" in node}
        active_head_nodes = {node for node in active_nodes if "heads" in node}

        # 3. Find the set of components to prune by taking the set difference
        mlps_to_prune = all_mlp_nodes - active_mlp_nodes
        heads_to_prune = all_head_nodes - active_head_nodes

        # 4. Call the respective pruning methods
        print(f"Pruning {len(mlps_to_prune)} unused MLP layers...")
        self.prune_mlp(list(mlps_to_prune))

        print(f"Pruning {len(heads_to_prune)} unused attention heads...")
        self.prune_heads(list(heads_to_prune))
        
        print("\nModel pruned. Ready for retraining on the circuit.")
            
        
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            nn.init.trunc_normal_(
                module.position_embeddings,
                mean=0.0,
                std=self.config["initializer_range"],
            )

            nn.init.trunc_normal_(
                module.cls_token,
                mean=0.0,
                std=self.config["initializer_range"],
            )

    def _set_module_names(self):
        """
        Recursively sets 'name' attribute for all modules using hierarchical naming
        Format: "parent_name.child_name"
        """
        for name, module in self.named_modules():
            module.name = name

    def print_module_names(self):
        """
        Recursively sets 'name' attribute for all modules using hierarchical naming
        Format: "parent_name.child_name"
        """
        for name, module in self.named_modules():
            print(name)