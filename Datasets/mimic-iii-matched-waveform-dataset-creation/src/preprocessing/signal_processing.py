import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from preprocessing.downsampling import create_downsampler
from preprocessing.imputing import create_imputer


class DatasetCreationError(Exception):
    """Custom exception for dataset creation errors."""
    pass


def get_channel_signal_from_array(data, channel_name, channel_names):
    # get channel signal
    if channel_name not in channel_names:
            raise DatasetCreationError(f"Channel {channel_name} not found in filtered names: {channel_names}")
    
    channel_index = channel_names.index(channel_name)
    channel = data[:, channel_index]
    # turn into np.array
    channel = np.array(channel)
    return channel


def clean_data(channel, lower_threshold, upper_threshold):
    """
    Clean data by removing values outside specified thresholds.
    
    Args:
        channel: Signal channel data as np.ndarray
        lower_threshold: Lower threshold for cleaning
        upper_threshold: Upper threshold for cleaning
        
    Returns:
        Cleaned channel data
    """
    if lower_threshold is not None:
        channel[channel < lower_threshold] = np.nan
    if upper_threshold is not None:
        channel[channel > upper_threshold] = np.nan
    return channel


def downsample_record(channel, to_downsample, current_fs):

    downsampling_strategy = to_downsample.get('downsampling_strategy', 'decimate')
    desired_resolution = to_downsample.get('desired_resolution', 1.0)
    downsampler = create_downsampler(downsampling_strategy)
    channel = downsampler.downsample(channel, desired_resolution, current_fs)

    return channel, desired_resolution


def perform_signal_processing(
        filtered_data: np.ndarray, 
        filtered_names: List[str], 
        signal_processing: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
    """
    Perform signal processing on the filtered data.
    
    Args:
        filtered_data: Filtered signal data (samples x channels)
        filtered_names: Names of the channels in the filtered data
        signal_processing: Configuration for signal processing
        metadata: Metadata containing sampling rate and other info
        
    Returns:
        Processed signal data and updated channel names
    """
    
    processed_data = []
    processed_names = []

    # process channels individually
    for channel_config in signal_processing:
        channel_name = channel_config.get('channel')
        steps = channel_config.get('steps', [])
        steps = sorted(steps, key=lambda x: x['step'])
        channel = get_channel_signal_from_array(filtered_data, channel_name, filtered_names)
        # other channels are needed for hinrichs imputer
        other_channels = np.delete(filtered_data, filtered_names.index(channel_name), axis=1)
        # for donwsampling we need the current fs
        current_fs = metadata['sampling_rate']
        for step in steps:

            to_downsample = step.get("downsampling", {})
            to_data_cleaning = step.get("data_cleaning", {})
            to_imputation = step.get("imputation", {})


            if to_downsample != {}:
                channel, current_fs = downsample_record(channel, to_downsample, current_fs)

            if to_data_cleaning != {}:
                lower_threshold = to_data_cleaning.get('lower_threshold')
                upper_threshold = to_data_cleaning.get('upper_threshold')
                channel = clean_data(channel, lower_threshold, upper_threshold)

            if to_imputation != {}:
                imputation_stategy = to_imputation.get('imputation_strategy', 'mean')
                imputer = create_imputer(imputation_stategy)
                channel = imputer.impute(channel, other_channels)
                
        processed_data.append(channel)
        processed_names.append(channel_name)

    ignored_channels = set(filtered_names) - set(processed_names)
    for ignored_channel in ignored_channels:
        processed_names.append(ignored_channel)
        channel = get_channel_signal_from_array(filtered_data, ignored_channel, filtered_names)
        processed_data.append(channel[0])

    # turn processed_data into np.array with the shape (channels x samples )
    processed_data = np.array(processed_data)

    return processed_data, processed_names
