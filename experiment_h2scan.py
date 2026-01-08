import sys
import torch
import numpy as np
import argparse
from util.trainer import *
from config.configs import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    return seed


def run_h2scan_experiment():
    """Main experiment runner for H2SCAN"""
    
    parser = argparse.ArgumentParser(description='H2SCAN Time Series Representation Learning')
    parser.add_argument('--task_type', default='p', type=str,
                        help='p (pre-train) or l (linear eval) or f (fine-tune)')
    parser.add_argument('--task', default='c', type=str,
                        help='c (classification) or r (regression)')
    parser.add_argument('--model', default=None, type=str,
                        help='Path to pre-trained model')
    parser.add_argument('--dataset', default='SLE', type=str,
                        help='Dataset name')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device to use')
    parser.add_argument('--seed', default=2024, type=int,
                        help='Random seed')
    
    # H2SCAN specific arguments
    parser.add_argument('--n_layers', default=3, type=int,
                        help='Number of hypergraph layers')
    parser.add_argument('--n_heads', default=4, type=int,
                        help='Number of attention heads')
    parser.add_argument('--n_freq_bands', default=8, type=int,
                        help='Number of frequency bands')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature for contrastive learning')
    parser.add_argument('--use_meta', action='store_true',
                        help='Use meta-learning for adaptation')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create config
    config = H2SCANConfig()
    config.device = args.device
    config.n_hypergraph_layers = args.n_layers
    config.n_attention_heads = args.n_heads
    config.n_freq_bands = args.n_freq_bands
    config.temperature = args.temperature
    config.use_meta_learning = args.use_meta
    
    if args.task_type == 'p':
        # Pre-training
        config.pretrain_dataset = args.dataset
        print(f"\n{'='*60}")
        print(f"Starting H²SCAN Pre-training on {args.dataset}")
        print(f"{'='*60}\n")
        print(f"Configuration:")
        print(f"  - Hypergraph layers: {config.n_hypergraph_layers}")
        print(f"  - Attention heads: {config.n_attention_heads}")
        print(f"  - Frequency bands: {config.n_freq_bands}")
        print(f"  - Temperature: {config.temperature}")
        print(f"  - Meta-learning: {config.use_meta_learning}")
        print(f"  - Device: {config.device}")
        print(f"\n{'='*60}\n")
        
        model = pre_train_h2scan(config)
        
        print(f"\n{'='*60}")
        print("Pre-training completed successfully!")
        print(f"Model saved to: {model.model_path}")
        print(f"{'='*60}\n")
        
    elif args.task_type in ['l', 'f']:
        # Fine-tuning or linear evaluation
        if args.task == 'c':
            finetune_config = DownstreamConfig_cls()
            finetune_config.finetune_dataset = args.dataset
            finetune_config.finetune_encoder = (args.task_type == 'f')
            finetune_config.device = args.device
            
            if args.dataset == "UCR":
                results = fine_tune_UCR(config, finetune_config, args.model, "H2SCAN")
                print(f"\nUCR Results: {results}")
            else:
                results = fine_tune_cls(config, finetune_config, args.model, "H2SCAN")
                print(f"\nClassification Results: {results}")
                
        elif args.task == 'r':
            finetune_config = DownstreamConfig_reg()
            finetune_config.finetune_dataset = args.dataset
            finetune_config.finetune_encoder = (args.task_type == 'f')
            finetune_config.device = args.device
            
            results = fine_tune_reg(config, finetune_config, args.model, "H2SCAN")
            print(f"\nRegression Results: {results}")
    else:
        raise ValueError(f"Unknown task type: {args.task_type}")


if __name__ == '__main__':
    run_h2scan_experiment()
