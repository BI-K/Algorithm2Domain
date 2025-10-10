import os
import pandas as pd
import numpy as np
import torch
import argparse
import sys

def create_dataset_pt(path:str, is_train: bool):
    samples = []
    labels = []

    if is_train:
        path += '/train'
    else:
        path += '/test'
    
    # read all folders in the path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    i = 0
    for folder in folders:
        observation_files = [f for f in os.listdir(os.path.join(path, folder, "observation")) if f.endswith('.csv')]
        for file in observation_files:
            file_path = os.path.join(path, folder, "observation", file)
            samples_df = pd.read_csv(file_path)
            labels_df = pd.read_csv(os.path.join(path, folder, "prediction", file))

            # convert samples_df to a list of rows
            samples.extend([samples_df.values.tolist()])
            labels.extend([labels_df.values.tolist()])

    
        i += 1
        print(f"Processed {i}/{len(folders)} folders.")

    dataset_dict = {
        'samples': np.array(samples),
        'labels': np.array(labels)
    }

    print("number of samples: ", len(dataset_dict["samples"]))

    # save as .pt file
    dataset_name = path.split('/')[-3]
    adatime_data_path = "./Algorithm2Domain/Evaluation_Framework/ADATime_data/PHD/"
    torch.save(dataset_dict, os.path.join(adatime_data_path, f'{"train" if is_train else "test"}_{dataset_name}.pt'))


def main():
    parser = argparse.ArgumentParser(description='Convert observation/prediction CSV files to PyTorch dataset format')

    parser.add_argument('--path', 
                       type=str, 
                       help='Path to the split_data directory containing train/test folders')
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist!")
        sys.exit(1)
    
    # Validate that it's a directory
    if not os.path.isdir(args.path):
        print(f"Error: '{args.path}' is not a directory!")
        sys.exit(1)

    # create_dataset_pt(args.path, True)
    # create only test file
    create_dataset_pt(args.path, False)

if __name__ == "__main__":
    main()
