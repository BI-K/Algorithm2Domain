import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score, Recall, Specificity
from ..dataloader.dataloader import data_generator, few_shot_data_generator
from ..configs.data_model_configs import get_dataset_class
from ..configs.hparams import get_hparams_class
from ..configs.sweep_params import sweep_alg_hparams, get_sweep_model_hparams
from ..utils import fix_randomness, starting_logs, DictAsObject,AverageMeter
from ..algorithms.algorithms import get_algorithm_class
from ..models.models import get_backbone_class
from ..postprocessing.postprocesser import postprocess
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):

        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset

        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset 
        self.run_description = f"{args.da_method}_{args.exp_name}"
        
        # paths
        self.home_path =  os.getcwd() #os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)


        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        self.scenario_string = ""
        for (source, target) in self.dataset_configs.scenarios:
            self.scenario_string += f"{source}_to_{target},"
        self.scenario_string = self.scenario_string[:-1]  # Remove trailing comma and space

        # Perform validity checks on arguments - e.g. do not apply da methods for classification tasks to continuous prediction tasks
        # DSAN, 
        if self.dataset_configs.num_classes == 0 and self.da_method in ["DSAN"]:
            raise ValueError(f"{self.da_method} not applicable for continuous prediction tasks. Please select a different DA method.")

        self.post_processing_classification = self.dataset_configs.post_processing_classification

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channels if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        # to fix dimension of first linear layer in TransformerHinrichs 
        self.dataset_configs.batch_size = self.hparams_class.train_params["batch_size"]

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}
    

        # metrics
        self.num_cont_output_channels = self.dataset_configs.num_cont_output_channels
        self.num_classes = self.dataset_configs.num_classes
        if self.num_cont_output_channels == 0:
            self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
            self.Sensitivity = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
            self.Specificity = Specificity(task="multiclass", num_classes=self.num_classes, average="macro")
            self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes) 
            self.MSE = 0   
            self.RMSE = 0
            self.MAPE = 0
        else:
            self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
            self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes) 
            self.Sensitivity = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
            self.Specificity = Specificity(task="multiclass", num_classes=self.num_classes, average="macro")
            self.MSE = torch.nn.MSELoss()
            self.RMSE = lambda x,y : torch.sqrt(self.MSE(x,y))
            self.MAPE = lambda x, y: torch.mean(torch.abs((x - y) / torch.where(y == 0, torch.ones_like(y), y))) * 100

        self.all_results = []

        self.results_columns = ["scenario", "run", "source_acc", "source_f1_score", "source_auroc", "source_sensitvity", "source_specificity",
                                "target_acc", "target_f1_score", "target_auroc", "target_sensitvity", "target_specificity"] 
        self.results_columns += ["source_mse_" + str(i) for i in range(self.num_cont_output_channels)] 
        self.results_columns += ["source_rmse_" + str(i) for i in range(self.num_cont_output_channels)]
        self.results_columns += ["source_mape_" + str(i) for i in range(self.num_cont_output_channels)] 
        self.results_columns +=  ["target_mse_" + str(i) for i in range(self.num_cont_output_channels)] 
        self.results_columns +=  ["target_rmse_" + str(i) for i in range(self.num_cont_output_channels)] 
        self.results_columns +=  ["target_mape_" + str(i) for i in range(self.num_cont_output_channels)] 
        self.results_columns +=  ["src_risk", "few_shot_risk", "trg_risk"]


    def sweep(self):
        # sweep configurations
        pass
    
    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        if best_model is None:
            best_model = last_model
        return last_model, best_model

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
        return self.last_model, self.best_model
    

    def create_results_table(self, src_id, trg_id, run_id, on_all_target=False):
        # calculate metrics and risks
        metrics_source = self.calculate_metrics(for_target=False)
        if on_all_target:
            metrics_target = self.calculate_metrics(for_target=False, for_all_target=True)
        else:
            metrics_target = self.calculate_metrics(for_target=True)
        risks = self.calculate_risks()


        # Table.add_data
        result_entry = {
            "scenario" : f"{src_id}_to_{trg_id}",
            "run": run_id,
            "target_acc": metrics_target[0],
            "target_f1_score": metrics_target[1],
            "target_auroc": metrics_target[2],
            "target_sensitvity": metrics_target[3],
            "target_specificity": metrics_target[4],
            "source_acc": metrics_source[0],
            "source_f1_score": metrics_source[1],
            "source_auroc": metrics_source[2],
            "source_sensitvity": metrics_source[3],
            "source_specificity": metrics_source[4],
            "src_risk": risks[0],
            "few_shot_risk": risks[1],
            "trg_risk": risks[2]
        }
        for i in range(self.num_cont_output_channels):
            # target
            result_entry[f"target_mse_{i}"] = metrics_target[5 + i]
            result_entry[f"target_rmse_{i}"] = metrics_target[5 + self.num_cont_output_channels + i]
            result_entry[f"target_mape_{i}"] = metrics_target[5 + 2 * self.num_cont_output_channels + i]

            # source
            result_entry[f"source_mse_{i}"] = metrics_source[5 + i]
            result_entry[f"source_rmse_{i}"] = metrics_source[5 + self.num_cont_output_channels + i]
            result_entry[f"source_mape_{i}"] = metrics_source[5 + 2 * self.num_cont_output_channels + i]

        # order results correctly
        results_entry_as_list = []
        for column in self.results_columns:
            results_entry_as_list.append(result_entry[column])

        self.all_results.append(results_entry_as_list)
        return results_entry_as_list, risks

    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        # optionally 

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                if self.num_cont_output_channels == 0:
                    labels = labels.long().to(self.device)
                else:
                    labels = labels.float().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)
                # postprocessing on ground truth and predictions
                # predictions = postprocess(predictions)
                # labels = postprocess(labels)


                # compute loss
                if self.num_cont_output_channels == 0:
                    loss = F.cross_entropy(predictions, labels)
                else:
                    labels = labels.view(labels.size(0), -1)  # flatten labels to match predictions
                    loss = F.mse_loss(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train", is_source=True)
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test", is_source=True)

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train", is_source=True)     
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test", is_source=True)
        self.trg_all = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "all", is_source=False)

        self.few_shot_dl_5 = few_shot_data_generator(self.trg_test_dl, self.dataset_configs,
                                                     5)  # set 5 to other value if you want other k-shot FST

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        mse = self.MSE(self.full_preds.cpu(), self.full_labels.cpu()).item()
        rmse = self.RMSE(self.full_preds.cpu(), self.full_labels.cpu()).item()
        mape = self.MAPE(self.full_preds.cpu(), self.full_labels.cpu()).item()

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc, mse, rmse, mape

        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):
       
        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'total_results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self, for_target=True, for_all_target=False):

        if for_target:
            self.evaluate(self.trg_test_dl)
        elif for_all_target:
            self.evaluate(self.trg_all)
        else:
            self.evaluate(self.src_test_dl)

        if self.num_cont_output_channels == 0:
            # accuracy  
            acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
            # f1
            f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
            # auroc 
            auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
            sensitvity = self.Sensitivity(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
            specificity = self.Specificity(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
             # mse, rmse, mape

            mses = [0 for _ in range(self.num_cont_output_channels)]
            rmses = [0 for _ in range(self.num_cont_output_channels)]
            mapes = [0 for _ in range(self.num_cont_output_channels)]
            
        else:

            if self.num_cont_output_channels > 1:
                self.full_labels = self.full_labels.squeeze(1)  # remove the second dimension

            if self.post_processing_classification:
                # apply postprocessing to self.full_preds and self.full_labels

                full_preds_classified = postprocess(self.full_preds)
                full_labels_classified = postprocess(self.full_labels)
                # add dimension to full_labels_classified
                acc = self.ACC(full_preds_classified.argmax(dim=1).cpu(), full_labels_classified.argmax(dim=1).cpu()).item()
                f1 = self.F1(full_preds_classified.argmax(dim=1).cpu(), full_labels_classified.argmax(dim=1).cpu()).item()
                sensitvity = self.Sensitivity(full_preds_classified.argmax(dim=1).cpu(), full_labels_classified.argmax(dim=1).cpu()).item()
                specificity = self.Specificity(full_preds_classified.argmax(dim=1).cpu(), full_labels_classified.argmax(dim=1).cpu()).item()
                full_preds_classified = full_preds_classified.long()
                full_labels_classified = full_labels_classified.long()
                auroc = self.AUROC(full_preds_classified.float().cpu(), full_labels_classified.argmax(dim=1).cpu()).item()
            else:
                acc = 0
                f1 = 0
                auroc = 0
                sensitvity = 0
                specificity = 0

            mses = [self.MSE(self.full_preds[:, i].cpu(), self.full_labels[:,  i].cpu()).item() for i in range(self.num_cont_output_channels)]
            rmses = [self.RMSE(self.full_preds[:, i].cpu(), self.full_labels[:,  i].cpu()).item() for i in range(self.num_cont_output_channels)]
            mapes = [self.MAPE(self.full_preds[:, i].cpu(), self.full_labels[:,  i].cpu()).item() for i in range(self.num_cont_output_channels)]

        return acc, f1, auroc, sensitvity, specificity, *mses, *rmses, *mapes

    def calculate_risks(self):
         # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, fst_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        if scenario is None or run_id is None or metrics is None:
            results_row = [*metrics]
        else:
            results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        # Check if table is empty to avoid FutureWarning
        if table.empty:
            table = results_df.copy()
        else:
            table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        if table.empty:
            table = pd.concat([mean_metrics_df, std_metrics_df], ignore_index=True)
        else:
            table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.map(format_func)
        #table = table.applymap(format_func)

        return table 