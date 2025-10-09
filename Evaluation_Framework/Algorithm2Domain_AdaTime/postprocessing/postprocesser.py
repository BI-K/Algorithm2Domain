import json 
import os
import torch
import torch.nn.functional as F

def postprocess(input_data):
    # read json
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "..", "configs", "postprocessing_configs.json")
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as file:
        data = json.load(file)

    # apply postprocessing steps
    for step in data.get("postprocessing_steps", []):
        output_channel = step.get("output_channel", 0)
        step_type = step.get("steps", {}).get("type", "")
        step_params = step.get("steps", {}).get("params", {})

        # input data is an array of shape [batch_size, num_channels, predictions]
        channel_data = input_data[:, output_channel]

        preds = []

        if step_type == "interval_to_label":
            intervals = step_params.get("intervals", [])
            for pred in channel_data:
                label_found = False
                for interval in intervals:
                    if interval["start"] <= pred < interval["end"]:
                        preds.append(interval["label"])
                        label_found = True
                        break
                
                # Handle case where prediction doesn't fall in any interval
                if not label_found:
                    # Assign to default class (e.g., 0) or closest interval
                    preds.append(0)

    num_classes = data.get("num_classes", 2)
    
    # Convert to tensor first, then apply one-hot encoding
    preds = torch.tensor(preds, dtype=torch.long)
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).float()

    return preds_one_hot