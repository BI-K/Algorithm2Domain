import numpy as np

sweep_train_hparams = {
    'num_epochs':   {'values': [30, 50, 70, 90, 100, 120, 150]},
    'batch_size':   {'values': [32, 64, 128, 256, 512, 1024, 2048]},
    'learning_rate':{'values': list(np.logspace(-5, -1, num=10))},
    'disc_lr':      {'values': [1]},
    'weight_decay': {'values': list(np.logspace(-6, -4, num=6))},
    'step_size':    {'values': [5, 10, 30]},
    'gamma':        {'values': [1]},
    'optimizer':    {'values': ['adam']},
}
sweep_alg_hparams = {
        'TARGET_ONLY': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'values': [1]},
            'domain_loss_wt':   {'values': [1]}
        },
        'NO_ADAPT': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'values': [1]},
            'domain_loss_wt':   {'values': [1]}
        },
        'DANN': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'AdvSKM': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoDATS': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CDAN': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'Deep_Coral': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
        },

        'DIRT': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'vat_loss_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'HoMM': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'hommd_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'MMDA': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'DSAN': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
        },

        'DDC': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },
        
        'SASA': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoTMix': {
            'learning_rate':    {'values': list(np.logspace(-5, -1, num=10))},
            'temporal_shift':           {'values': [5, 10, 15, 20, 30, 50]},
            'src_cls_weight':           {'distribution': 'uniform', 'min': 1e-1, 'max': 1},
            'mix_ratio':                {'distribution': 'uniform', 'min': 0.5, 'max': 0.99},
            'src_supCon_weight':        {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_cont_weight':          {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_entropy_weight':       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
        },
}

sweep_model_params = {
    'CNN': {
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'stride': {'values': [1, 2]},
        'kernel_size': {'values': [3, 5, 7, 11]},
        'mid_channels': {'values': [8, 16, 32, 64, 128]}
    },
    'TCN': {
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'tcn_layers': {'values': [[16], [32], [32, 16], [32, 16, 8], [64], [64, 32], [64, 32, 16]]},
        'tcn_kernel_size': {'values': [3, 5, 7, 11]},
        'tcn_final_out_channels': {'values': [2, 4, 8, 16, 32, 64]}
    },
    'RESNET18': {
    },
    'LTCN': {
        'ltcn_layers': {'values': [[64], [64, 32], [64, 32, 16]]},
        'ode_unfolds': {'values': [2, 4, 6, 8]},
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]}
    },
    'CfCN': {
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'backbone_units': {'values': [8, 16, 32, 64]},
        'backbone_layers': {'values': [1, 2, 3]},
        'backbone_dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'cfcn_layers': {'values': [[64], [64, 32], [64, 32, 16]]}
    },
    'GRUHinrichs': {
        'hidden_size_gru': {'values': [8, 16, 32, 64, 128]},
        'num_layers': {'values': [1, 2, 3]},
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]}
    },
    'TransformerHinrichs': {
        'n_heads': {'values': [1, 2, 4]},
        'num_layers': {'values': [1, 2, 3]},
        'activation': {'values': ['relu', 'gelu']},
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'dim_feedforward': {'values': [16, 32, 64, 128, 256]},
        'pos_encoding': {'values': ['fixed']},
        'freeze': {'values': [True, False]},
        'norm': {'values': ['BatchNorm']}
    },
    'LSTM': {
        'bidirectional_channels': {'values': [True, False]},
        'lstm_layers': {'values': [[16], [32], [32, 16], [32, 16, 8], [64], [64, 32], [64, 32, 16]]},
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4, 0.5]}
    }
}

def get_sweep_model_hparams(model_name):
    """
    Get hyperparameters for the specified model for sweep.
    If the model is not found, return an empty dictionary.
    """
    if model_name in sweep_model_params:
        return sweep_model_params[model_name]
    else:
        print(f"Model {model_name} not found in sweep_model_params.")
        return {}

def get_sweep_train_hparams(ui_hparams=None):
    """
    Get sweep training hyperparameters.
    If ui_hparams is provided (from hyperparameter tuning UI), use those values.
    Otherwise, use the default sweep_train_hparams.
    """
    if ui_hparams is None:
        return sweep_alg_hparams
    
    # Start with default sweep_train_hparams
    merged_hparams = sweep_train_hparams.copy()
    
    # Update with UI values if provided
    for param_name, param_config in ui_hparams.items():
        if param_name in merged_hparams:
            # Ensure the format matches what sweep expects
            if isinstance(param_config, dict) and "values" in param_config:
                merged_hparams[param_name] = param_config
            else:
                # Convert to proper format if needed
                merged_hparams[param_name] = {"values": param_config}
    
    return merged_hparams

def get_combined_sweep_hparams(ui_hparams=None, algorithm=None):
    """
    Get combined sweep hyperparameters including both training and algorithm-specific parameters.
    If ui_hparams is provided (from hyperparameter tuning UI), use those values for training params.
    """
    # Get training hyperparameters (with UI overrides if provided)
    training_hparams = get_sweep_train_hparams(ui_hparams)
    
    # Get algorithm-specific hyperparameters
    alg_hparams = {}
    if algorithm and algorithm in sweep_alg_hparams:
        alg_hparams = sweep_alg_hparams[algorithm].copy()

    # Get all model-specific hyperparameters
    #model_hparams = {}
    #if model and model in sweep_model_params:
    #    model_hparams = get_sweep_model_hparams(model)
    
    # Combine both sets of hyperparameters
    # Training parameters (especially from UI) take precedence over algorithm-specific ones
    combined_hparams = {**alg_hparams, **training_hparams}#, **model_hparams}

    return combined_hparams