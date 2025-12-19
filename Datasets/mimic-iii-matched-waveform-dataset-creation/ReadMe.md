# MIMIC-III Waveform Database Matched Subset

The dataset consists of 10,282 Patients and 22,247 numeric records. Only 10,269 patients have at least one numeric record. You can create input files for this data transformer via [WavePrep](https://github.com/BI-K/WavePrep)


## Transform data to required format that is accespted by Algorithm2Domain_ADATime

Call 
```python ./scripts/transform_input_data_to_expected_format_ADATime.py --path ./outputs/split/default/split_data```
with the --path argument pointing towards your desired path. 
The script will output a single .pt file, that conforms to the requirements of Algorithm2Domain_ADATime to  `adatime_data_path = "./Algorithm2Domain/Evaluation_Framework/ADATime_data/PHD/"` please change the path to your liking. 