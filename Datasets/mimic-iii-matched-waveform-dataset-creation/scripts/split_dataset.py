#!/usr/bin/env python3
"""
MIMIC III Matched Waveform Dataset Splitting Script

This script splits structured datasets into training, validation, and test sets.
"""

import sys
from pathlib import Path
from typing import Dict, Any


# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from split.mimic_splitter import run_dataset_splitting

from common import (
    create_standard_parser,
    load_script_configuration,
    setup_script_environment,
    validate_required_config,
    ConfigError
)


def main() -> int:
    """Main entry point for the dataset splitting script."""
    logger = None
    
    try:
        # Parse arguments
        parser = create_standard_parser("Split dataset into train/validation/test sets")
        args = parser.parse_args()
        
        # Load configuration
        config = load_script_configuration(args.config)
        
        # Set up environment
        logger, output_manager = setup_script_environment("split", config, args.config)
        
        # Log script start
        logger.info("Starting dataset splitting")
        logger.debug(f"Configuration loaded from: {args.config}")
        
        # Validate required configuration
        required_keys = [
            "input_path",
            "splitting_method",
            "train_ratio",
            "validation_ratio",
            "test_ratio"
        ]
        validate_required_config(config, required_keys, logger)
        
        # Log splitting parameters
        input_path = config.get("input_path")
        splitting_method = config.get("splitting_method")
        train_ratio = config.get("train_ratio")
        validation_ratio = config.get("validation_ratio")
        test_ratio = config.get("test_ratio")
        random_seed = config.get("random_seed", 42)
        
        logger.info(f"Input path: {input_path}")
        logger.info(f"Splitting method: {splitting_method}")
        logger.info(f"Random seed: {random_seed}")
        logger.info(f"Split ratios - Train: {train_ratio}, Val: {validation_ratio}, Test: {test_ratio}")
        
        # Validate split ratios sum to 1.0
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:  # Allow for small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Log subject filtering if configured
        exclude_subjects = config.get("exclude_subjects", [])
        include_only_subjects = config.get("include_only_subjects", [])
        
        if exclude_subjects:
            logger.info(f"Excluding {len(exclude_subjects)} subjects")
        if include_only_subjects:
            logger.info(f"Including only {len(include_only_subjects)} subjects")
        
        # Run the splitting process
        results = run_dataset_splitting(config, output_manager, logger)
        
        # Log final results
        logger.info(f"Dataset splitting completed successfully")
        logger.info(f"Results saved to: {results['output_dir']}")
        
        return 0
        
    except ConfigError as e:
        if logger:
            logger.error(f"Configuration error: {str(e)}")
        else:
            print(f"Configuration error: {str(e)}", file=sys.stderr)
        return 1
        
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {str(e)}")
        else:
            print(f"Unexpected error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
