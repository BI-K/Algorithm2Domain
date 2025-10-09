import os
import torch
import subprocess
from Algorithm2Domain_AdaTime.trainers.sweep import Trainer
from Algorithm2Domain_AdaTime.configs.data_model_configs import get_dataset_class
from Algorithm2Domain_AdaTime.configs.hparams import get_hparams_class
from Algorithm2Domain_AdaTime.configs.sweep_params import get_sweep_train_hparams, get_combined_sweep_hparams
import argparse

def setup_gpu_environment(device):
    """Setup GPU environment properly"""
    if device.startswith('cuda'):
        # Extract GPU ID
        if ':' in device:
            gpu_id = device.split(':')[1]
        else:
            gpu_id = '0'
        
        # Set CUDA_VISIBLE_DEVICES to limit to specific GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        
        # Set TensorFlow GPU memory growth (if TensorFlow is used)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Verify GPU availability
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use device 0
            print(f"Using GPU {gpu_id}, PyTorch device: {torch.cuda.current_device()}")
        else:
            print("CUDA not available, falling back to CPU")
            return 'cpu'
        
        return 'cuda:0'  # After CUDA_VISIBLE_DEVICES, always use cuda:0
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='Deep_Coral', type=str,
                        help='DANN, Deep_Coral, WDGRL, MMDA, VADA, DIRT, CDAN, ADDA, HoMM, CoDATS')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'./ADATIME_data', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA - PHD)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN - CfCN - GRUHinrichs - TransformerHinrichs - SWIFT)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive runs with different seeds')
    parser.add_argument('--device', default="cuda:0", type=str, help='Device: cpu, cuda:0, cuda:1, cuda:2, cuda:3')
    parser.add_argument('--exp_name', default='sweep_EXP1', type=str, help='experiment name')

    # ======== sweep settings =====================
    parser.add_argument('--num_sweeps', default=1, type=int, help='Number of sweep runs')
    parser.add_argument('--sweep_id', default='', type=str, help='Wandb sweep ID')
    parser.add_argument('--sweep_project_wandb', default='ADATIME_refactor', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', type=str, help='Entity name in Wandb')
    parser.add_argument('--hp_search_strategy', default="random", type=str, help='Hyperparameter search strategy')
    parser.add_argument('--metric_to_minimize', default="src_risk", type=str, help='Metric to minimize')
    parser.add_argument('--save_dir', default='ADATime/experiments_logs/sweep_logs', type=str, help='Save directory')

    args = parser.parse_args()
    
    # Setup GPU environment
    args.device = setup_gpu_environment(args.device)
    print(f"Running sweep with device: {args.device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    trainer = Trainer(args)
    dataset_class = get_dataset_class(args.dataset)
    dataset_config = dataset_class()
    sweep_hparams = get_sweep_train_hparams()
    hparams = None

    trainer.sweep(dataset_configs=dataset_config, sweep_hparams=sweep_hparams, hparams=hparams)
