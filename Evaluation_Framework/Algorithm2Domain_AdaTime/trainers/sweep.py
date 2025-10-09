import sys

sys.path.append('../')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
#import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions
import json

from ..configs.sweep_params import sweep_alg_hparams, sweep_model_params, get_sweep_model_hparams, sweep_train_hparams
from ..utils import fix_randomness, starting_logs, DictAsObject
from ..algorithms.algorithms import get_algorithm_class
from ..models.models import get_backbone_class
from ..utils import AverageMeter

from ..trainers.abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()

sweep_results = []

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_id = args.sweep_id
        self.sweep_project_wandb =   self.dataset + '_' + self.scenario_string + '_' + self.backbone + '_' + self.da_method
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)


    def load_json_config(self, file_path):
        """Load a JSON configuration file."""
        if not os.path.exists(file_path):
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}

    def load_dataset_creation_details(self):

        # read json file with sweep parameters
        scenario_creation_configs = {}
        # please flatten list of lists to a list with singular elements
        scenarios = []
        for scen_1, scen_2 in self.dataset_configs.scenarios:
            scenarios.append(scen_1)
            scenarios.append(scen_2)

        scenarios = list(set(scenarios))
        for scenario in scenarios:
            # base_config 
            base_config_path = os.path.join(self.home_path, "ADATime_data", self.dataset, "details", scenario,  "base_config.json")
            base_config = self.load_json_config(base_config_path)

            # create_config
            create_config_path = os.path.join(self.home_path, "ADATime_data", self.dataset, "details", scenario,  "create_config.json")
            create_config = self.load_json_config(create_config_path)

            # split_config
            split_config_path = os.path.join(self.home_path, "ADATime_data", self.dataset, "details", scenario, "split_config.json")
            split_config = self.load_json_config(split_config_path)

            scenario_creation_configs[scenario] = {
                "base_config": base_config,
                "create_config": create_config,
                "split_config": split_config
            }


        return scenario_creation_configs

    def sweep(self, dataset_configs=None, sweep_hparams=None, hparams=None):
        if dataset_configs is not None:
            self.dataset_configs = dataset_configs
        
        if hparams is not None:
            # Merge the provided hparams with existing ones to preserve default values
            self.hparams.update(hparams)
             

        if sweep_hparams is not None:
             sweep_alg_hparams = sweep_hparams

        model_sweep_hparams = {}
        if self.backbone in sweep_model_params:
            model_sweep_hparams = sweep_model_params[self.backbone]

        scenario_creation_configs = self.load_dataset_creation_details()
        data_augmentation_path = os.path.join(self.home_path, "Algorithm2Domain_AdaTime", "configs", "data_augmentation_configs.json")
        data_augmentation_configs = self.load_json_config(data_augmentation_path)
        #postprocessing_configs = self.load_json_config(os.path.join(self.home_path, "configs", "postprocessing_configs.json"))

        # log info not loggable otherwise as freetext in description
        # TODO add postprocessing
        self.description = f"scenario_configs: {scenario_creation_configs}, data_augmentation: {data_augmentation_configs}"
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone + '_' + self.dataset,
            'parameters': {**sweep_train_hparams, **sweep_alg_hparams[self.da_method], **model_sweep_hparams,
                           }
        }

        if self.sweep_id == '':
            sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)
        else:
            sweep_id = self.sweep_id

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)

        return self.all_results


    def train(self):

        temp_config = {**self.hparams, **self.dataset_configs.to_dict()}
        run = wandb.init(config=temp_config, notes=self.description)
        self.hparams = wandb.config

        # merge self.dataset_configs with wandb.config
        self.dataset_configs.update_configs(wandb.config)

        # create tables for results and risks
        columns = self.results_columns
        table_results = wandb.Table(columns=columns, allow_mixed_types=True)
        columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]
        table_risks = wandb.Table(columns=columns, allow_mixed_types=True)

        for src_id, trg_id in self.dataset_configs.scenarios:
                for run_id in range(self.num_runs):
                    # set random seed and create logger
                    fix_randomness(run_id)
                    self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)

                    # average meters
                    self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                    # load data and train model
                    self.load_data(src_id, trg_id)

                    # initiate the domain adaptation algorithm
                    self.initialize_algorithm()
                    # Train the domain adaptation algorithm
                    self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

                    results_entry_as_list, risks = self.create_results_table(src_id, trg_id, run_id)

                    # append results to tables
                    scenario = f"{src_id}_to_{trg_id}"
                    table_results.add_data(*results_entry_as_list)
                    table_risks.add_data(scenario, run_id, *risks)


        # calculate overall metrics and risks
        total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # log results to WandB
        self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)

        # update hparams with the best results
        best_hparams = {key: wandb.config[key] for key in wandb.config.keys()}
        self.hparams = best_hparams
        self.dataset_configs.update_configs(wandb.config)

        # finish the run
        run.finish()

        return total_results, summary_metrics, total_risks, summary_risks

