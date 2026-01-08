import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from models.pretrain_models import PretrainModel, get_hidden_dim
from config.configs import PretrainConfig

try:
    from einops import repeat, rearrange
except ImportError:
    print("Warning: einops not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'einops'])
    from einops import repeat, rearrange


class H2SCANConfig(PretrainConfig):
    def __init__(self):
        super(H2SCANConfig, self).__init__()
        # Hypergraph specific parameters
        self.n_hypergraph_layers = 3
        self.n_attention_heads = 4
        self.node_types = ['time', 'freq', 'stat']  # Multi-domain nodes
        self.hyperedge_types = ['temporal', 'freq_coherence', 'cross_domain']
        
        # Frequency decomposition
        self.n_freq_bands = 8  # Number of frequency bands to extract
        
        # Statistical features
        self.n_stat_features = 5  # mean, std, trend, min, max
        
        # Contrastive learning
        self.temperature = 0.07
        self.similarity_threshold = 0.5
        
        # Meta-learning for adaptation
        self.use_meta_learning = True
        self.meta_hidden_dim = 64


class MultiDomainNodeEncoder(nn.Module):
    """Encodes time series into multiple domain representations"""
    
    def __init__(self, config: H2SCANConfig):
        super(MultiDomainNodeEncoder, self).__init__()
        self.config = config
        self.hidden_dim = get_hidden_dim(config)
        
        # Time domain encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Frequency domain encoder
        self.freq_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),  # Real + Imaginary
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Statistical features encoder
        self.stat_encoder = nn.Sequential(
            nn.Linear(config.n_stat_features, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
    def extract_frequency_features(self, x):
        """Extract frequency domain features using FFT"""
        # x shape: (B, L, 1)
        fft_x = torch.fft.rfft(x.squeeze(-1), dim=-1)  # (B, F)
        
        # Split into frequency bands
        F = fft_x.shape[-1]
        band_size = F // self.config.n_freq_bands
        
        freq_nodes = []
        for i in range(self.config.n_freq_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < self.config.n_freq_bands - 1 else F
            
            band = fft_x[:, start_idx:end_idx]
            # Use magnitude and phase as features
            magnitude = torch.abs(band).mean(dim=-1, keepdim=True)
            phase = torch.angle(band).mean(dim=-1, keepdim=True)
            
            freq_feature = torch.cat([magnitude, phase], dim=-1)
            freq_nodes.append(freq_feature)
        
        return torch.stack(freq_nodes, dim=1)  # (B, n_freq_bands, 2)
    
    def extract_statistical_features(self, x):
        """Extract statistical features from time series"""
        # x shape: (B, L, 1)
        x = x.squeeze(-1)  # (B, L)
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        
        # Simple trend estimation (slope)
        time_idx = torch.arange(x.shape[-1], device=x.device).float()
        time_idx = time_idx / x.shape[-1]
        time_idx = repeat(time_idx, 'L -> B L', B=x.shape[0])
        
        # Linear regression for trend
        x_mean = x.mean(dim=-1, keepdim=True)
        t_mean = time_idx.mean(dim=-1, keepdim=True)
        
        numerator = ((x - x_mean) * (time_idx - t_mean)).sum(dim=-1, keepdim=True)
        denominator = ((time_idx - t_mean) ** 2).sum(dim=-1, keepdim=True)
        trend = numerator / (denominator + 1e-8)
        
        stat_features = torch.cat([mean, std, trend, min_val, max_val], dim=-1)
        return stat_features.unsqueeze(1)  # (B, 1, n_stat_features)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, 1) time series
        Returns:
            time_nodes: (B, L, D) time domain node embeddings
            freq_nodes: (B, n_freq_bands, D) frequency domain node embeddings
            stat_nodes: (B, 1, D) statistical node embeddings
        """
        B, L, _ = x.shape
        
        # Time domain nodes (each timestamp is a node)
        time_nodes = self.time_encoder(x)  # (B, L, D)
        
        # Frequency domain nodes
        freq_features = self.extract_frequency_features(x)  # (B, n_freq_bands, 2)
        freq_nodes = self.freq_encoder(freq_features)  # (B, n_freq_bands, D)
        
        # Statistical nodes
        stat_features = self.extract_statistical_features(x)  # (B, 1, n_stat_features)
        stat_nodes = self.stat_encoder(stat_features)  # (B, 1, D)
        
        return time_nodes, freq_nodes, stat_nodes


class HeterogeneousHyperedgeBuilder(nn.Module):
    """Builds different types of hyperedges connecting heterogeneous nodes"""
    
    def __init__(self, config: H2SCANConfig):
        super(HeterogeneousHyperedgeBuilder, self).__init__()
        self.config = config
        self.hidden_dim = get_hidden_dim(config)
        
        # Learnable hyperedge embeddings
        self.temporal_hyperedge_encoder = nn.Parameter(
            torch.randn(config.pretrain_sample_length, self.hidden_dim)
        )
        self.freq_coherence_hyperedge_encoder = nn.Parameter(
            torch.randn(config.n_freq_bands, self.hidden_dim)
        )
        self.cross_domain_hyperedge_encoder = nn.Parameter(
            torch.randn(config.pretrain_sample_length // 4, self.hidden_dim)
        )
        
    def build_temporal_hyperedges(self, time_nodes):
        """Connect time nodes that are temporally adjacent"""
        B, L, D = time_nodes.shape
        
        # Each temporal hyperedge connects adjacent timestamps
        # Return hyperedge embeddings and incidence matrix
        hyperedge_emb = repeat(
            self.temporal_hyperedge_encoder[:L],
            'L D -> B L D',
            B=B
        )
        
        # Incidence matrix: each timestamp connects to its hyperedge
        incidence = torch.eye(L, device=time_nodes.device)
        incidence = repeat(incidence, 'L1 L2 -> B L1 L2', B=B)
        
        return hyperedge_emb, incidence
    
    def build_freq_coherence_hyperedges(self, freq_nodes):
        """Connect frequency nodes with similar patterns"""
        B, F, D = freq_nodes.shape
        
        hyperedge_emb = repeat(
            self.freq_coherence_hyperedge_encoder[:F],
            'F D -> B F D',
            B=B
        )
        
        # Compute pairwise similarity
        freq_norm = F.normalize(freq_nodes, dim=-1)
        similarity = torch.bmm(freq_norm, freq_norm.transpose(1, 2))  # (B, F, F)
        
        # Incidence: connect nodes with high similarity
        incidence = (similarity > self.config.similarity_threshold).float()
        
        return hyperedge_emb, incidence
    
    def build_cross_domain_hyperedges(self, time_nodes, freq_nodes, stat_nodes):
        """Connect nodes across different domains"""
        B = time_nodes.shape[0]
        L = time_nodes.shape[1]
        n_hyperedges = L // 4
        
        hyperedge_emb = repeat(
            self.cross_domain_hyperedge_encoder[:n_hyperedges],
            'H D -> B H D',
            B=B
        )
        
        # Each cross-domain hyperedge connects:
        # - A segment of time nodes
        # - Related frequency nodes
        # - The statistical node
        
        # For simplicity, create segments and connect them
        # This is a placeholder for more sophisticated connection logic
        incidence = torch.zeros(B, n_hyperedges, L, device=time_nodes.device)
        segment_size = L // n_hyperedges
        
        for i in range(n_hyperedges):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_hyperedges - 1 else L
            incidence[:, i, start_idx:end_idx] = 1.0
        
        return hyperedge_emb, incidence
    
    def forward(self, time_nodes, freq_nodes, stat_nodes):
        """
        Returns:
            hyperedges: dict of hyperedge embeddings for each type
            incidences: dict of incidence matrices for each type
        """
        temporal_he, temporal_inc = self.build_temporal_hyperedges(time_nodes)
        freq_he, freq_inc = self.build_freq_coherence_hyperedges(freq_nodes)
        cross_he, cross_inc = self.build_cross_domain_hyperedges(
            time_nodes, freq_nodes, stat_nodes
        )
        
        hyperedges = {
            'temporal': temporal_he,
            'freq_coherence': freq_he,
            'cross_domain': cross_he
        }
        
        incidences = {
            'temporal': temporal_inc,
            'freq_coherence': freq_inc,
            'cross_domain': cross_inc
        }
        
        return hyperedges, incidences


class HypergraphMessagePassing(nn.Module):
    """Message passing on heterogeneous hypergraph"""
    
    def __init__(self, config: H2SCANConfig):
        super(HypergraphMessagePassing, self).__init__()
        self.config = config
        self.hidden_dim = get_hidden_dim(config)
        
        # Node-to-hyperedge attention
        self.node2hyperedge_attn = nn.MultiheadAttention(
            self.hidden_dim,
            config.n_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Hyperedge-to-node aggregation
        self.hyperedge2node_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Node update
        self.node_update = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
    def forward(self, nodes, hyperedges, incidence):
        """
        Args:
            nodes: (B, N, D) node embeddings
            hyperedges: (B, H, D) hyperedge embeddings
            incidence: (B, N, H) or (B, H, N) incidence matrix
        Returns:
            updated_nodes: (B, N, D)
        """
        B, N, D = nodes.shape
        
        # Node-to-hyperedge: aggregate node features to hyperedges
        # Use incidence matrix as attention mask
        if incidence.shape[1] == N:  # (B, N, H)
            # Reshape for attention
            attn_mask = incidence.transpose(1, 2)  # (B, H, N)
        else:  # (B, H, N)
            attn_mask = incidence
        
        # Create attention mask (True = masked position)
        attn_mask_bool = (attn_mask == 0)
        
        # Attend from hyperedges to nodes
        updated_hyperedges, _ = self.node2hyperedge_attn(
            hyperedges,
            nodes,
            nodes,
            key_padding_mask=None
        )  # (B, H, D)
        
        # Hyperedge-to-node: aggregate hyperedge features back to nodes
        updated_hyperedges = self.hyperedge2node_transform(updated_hyperedges)
        
        # Weighted aggregation based on incidence
        if incidence.shape[1] == N:
            # (B, N, H) @ (B, H, D) -> (B, N, D)
            aggregated = torch.bmm(incidence, updated_hyperedges)
        else:
            # (B, H, N) -> (B, N, H)
            aggregated = torch.bmm(incidence.transpose(1, 2), updated_hyperedges)
        
        # Normalize by degree
        degree = incidence.sum(dim=-1, keepdim=True).clamp(min=1)
        aggregated = aggregated / degree
        
        # Update nodes
        updated_nodes = self.node_update(
            torch.cat([nodes, aggregated], dim=-1)
        )
        
        return updated_nodes


class StructuralContrastiveLoss(nn.Module):
    """Non-augmentation contrastive loss based on hypergraph structure"""
    
    def __init__(self, config: H2SCANConfig):
        super(StructuralContrastiveLoss, self).__init__()
        self.config = config
        self.temperature = config.temperature
        
    def compute_hypergraph_similarity(self, nodes, incidence):
        """
        Compute similarity based on hypergraph connectivity
        Nodes connected by many common hyperedges are more similar
        """
        B, N, D = nodes.shape
        
        # Compute structural similarity via shared hyperedges
        # incidence: (B, N, H)
        if incidence.dim() == 3 and incidence.shape[1] != N:
            incidence = incidence.transpose(1, 2)
        
        # Structural similarity: nodes sharing hyperedges
        struct_sim = torch.bmm(incidence, incidence.transpose(1, 2))  # (B, N, N)
        
        # Normalize
        degree = incidence.sum(dim=-1, keepdim=True).clamp(min=1)
        struct_sim = struct_sim / (degree + degree.transpose(1, 2))
        
        return struct_sim
    
    def forward(self, nodes, incidence):
        """
        Args:
            nodes: (B, N, D) node embeddings
            incidence: (B, N, H) incidence matrix
        """
        B, N, D = nodes.shape
        
        # Normalize node embeddings
        nodes_norm = F.normalize(nodes, dim=-1)
        
        # Feature similarity
        feat_sim = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))  # (B, N, N)
        
        # Structural similarity from hypergraph
        struct_sim = self.compute_hypergraph_similarity(nodes, incidence)
        
        # Use structural similarity as soft labels
        # High structural similarity -> should have high feature similarity
        logits = feat_sim / self.temperature
        
        # Remove self-similarity
        mask = torch.eye(N, device=nodes.device).bool()
        mask = repeat(mask, 'N1 N2 -> B N1 N2', B=B)
        
        logits = logits.masked_fill(mask, float('-inf'))
        struct_sim = struct_sim.masked_fill(mask, 0)
        
        # Normalize structural similarity as target distribution
        struct_sim_norm = F.softmax(struct_sim / self.temperature, dim=-1)
        
        # KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, struct_sim_norm, reduction='batchmean')
        
        return loss


class MetaAdaptationNetwork(nn.Module):
    """Meta-network for dynamic hypergraph adaptation"""
    
    def __init__(self, config: H2SCANConfig):
        super(MetaAdaptationNetwork, self).__init__()
        self.config = config
        
        # Input: statistical properties of current segment
        self.meta_net = nn.Sequential(
            nn.Linear(config.n_stat_features, config.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.meta_hidden_dim, len(config.hyperedge_types))
        )
        
    def forward(self, stat_features):
        """
        Args:
            stat_features: (B, n_stat_features) statistical properties
        Returns:
            weights: (B, n_hyperedge_types) weights for each hyperedge type
        """
        weights = torch.softmax(self.meta_net(stat_features), dim=-1)
        return weights


class H2SCAN(PretrainModel):
    """Heterogeneous Hypergraph Representation Learning for Time Series"""
    
    def __init__(self, config: H2SCANConfig):
        super(H2SCAN, self).__init__(config)
        
        self.config = config
        self.hidden_dim = get_hidden_dim(config)
        
        # Multi-domain encoder
        self.domain_encoder = MultiDomainNodeEncoder(config)
        
        # Hyperedge builder
        self.hyperedge_builder = HeterogeneousHyperedgeBuilder(config)
        
        # Message passing layers
        self.message_passing_layers = nn.ModuleList([
            HypergraphMessagePassing(config)
            for _ in range(config.n_hypergraph_layers)
        ])
        
        # Contrastive loss
        self.contrastive_loss = StructuralContrastiveLoss(config)
        
        # Meta-adaptation network
        if config.use_meta_learning:
            self.meta_adapter = MetaAdaptationNetwork(config)
        
        # Projection head for final representation
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        self.to(config.device)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, 1) input time series
        Returns:
            representation: (B, D) learned representation
        """
        B, L, _ = x.shape
        
        # Encode to multi-domain nodes
        time_nodes, freq_nodes, stat_nodes = self.domain_encoder(x)
        # time_nodes: (B, L, D)
        # freq_nodes: (B, F, D)
        # stat_nodes: (B, 1, D)
        
        # Build heterogeneous hyperedges
        hyperedges, incidences = self.hyperedge_builder(
            time_nodes, freq_nodes, stat_nodes
        )
        
        # Meta-adaptation: compute weights for different hyperedge types
        if self.config.use_meta_learning:
            stat_features = self.domain_encoder.extract_statistical_features(x)
            stat_features = stat_features.squeeze(1)  # (B, n_stat_features)
            hyperedge_weights = self.meta_adapter(stat_features)  # (B, n_types)
        else:
            hyperedge_weights = torch.ones(
                B, len(self.config.hyperedge_types),
                device=x.device
            ) / len(self.config.hyperedge_types)
        
        # Message passing on hypergraph
        all_losses = []
        
        for layer in self.message_passing_layers:
            # Update time nodes
            for i, edge_type in enumerate(self.config.hyperedge_types):
                weight = hyperedge_weights[:, i:i+1, None]  # (B, 1, 1)
                
                if edge_type == 'temporal':
                    time_nodes_updated = layer(
                        time_nodes,
                        hyperedges[edge_type],
                        incidences[edge_type]
                    )
                    time_nodes = time_nodes + weight * time_nodes_updated
                    
                    # Compute contrastive loss for this layer
                    loss = self.contrastive_loss(time_nodes, incidences[edge_type])
                    all_losses.append(weight.squeeze() * loss)
        
        # Aggregate representations from all domains
        time_repr = time_nodes.mean(dim=1)  # (B, D)
        freq_repr = freq_nodes.mean(dim=1)  # (B, D)
        stat_repr = stat_nodes.squeeze(1)   # (B, D)
        
        # Combine multi-domain representations
        combined = torch.cat([time_repr, freq_repr, stat_repr], dim=-1)
        representation = self.projection(combined)  # (B, D)
        
        # Total contrastive loss
        total_loss = torch.stack(all_losses).sum()
        
        return representation, total_loss
    
    def compute_loss(self, x, y, criterion):
        """
        Args:
            x: (B, L, 1) input time series
            y: labels (not used in self-supervised learning)
            criterion: loss function
        Returns:
            loss: scalar loss value
        """
        representation, contrastive_loss = self(x)
        
        # For pre-training, we only use contrastive loss
        return contrastive_loss


# Export config for easy access
__all__ = ['H2SCAN', 'H2SCANConfig']
