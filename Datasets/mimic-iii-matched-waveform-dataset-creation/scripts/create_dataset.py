#!/usr/bin/env python3
"""
MIMIC III Matched Waveform Dataset Creation Script

This script creates structured datasets from the MIMIC III matched waveform database
for machine learning and analysis purposes.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import warnings
warnings.filterwarnings("ignore")

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common import (
    create_standard_parser,
    load_script_configuration,
    setup_script_environment,
    validate_required_config,
    ConfigError
)


def main() -> int:
    """Main entry point for the dataset creation script."""
    logger = None
    
    try:
        # Parse arguments
        parser = create_standard_parser("Create structured dataset from MIMIC III waveform data")
        args = parser.parse_args()
        
        # Load configuration
        config = load_script_configuration(args.config)
        
        # Set up environment
        logger, output_manager = setup_script_environment("create", config, args.config)
        
        # Log script start
        logger.info("Starting dataset creation")
        logger.debug(f"Configuration loaded from: {args.config}")
        
        # Validate required configuration
        required_keys = [
            "database_name",
            "input_channels",
            "output_channels",
            "signal_processing",
            "windowing"
        ]
        validate_required_config(config, required_keys, logger)
        
        # Import the dataset creation functions
        from create.dataset_creator import run_dataset_creation
        
        # Run dataset creation
        run_dataset_creation(config, output_manager, logger)
        
        # Log completion
        logger.info("Dataset creation completed")
        
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
