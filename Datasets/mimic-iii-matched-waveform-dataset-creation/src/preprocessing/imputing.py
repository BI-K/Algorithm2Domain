from preprocessing import AbstractImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

class HinrichsPaperImputer(AbstractImputer):
    """Imputer based on the method described in Hinrichs et al. (2020)."""

    def impute(self, channel: np.ndarray, other_channels) -> np.ndarray:
        """
        Impute missing values in a signal channel using the specified method.
        
        Args:
            channel: Signal channel data as a numpy array
            method: Imputation method to use (e.g., 'mean', 'median')
        Returns:
            Channel with imputed values
        """
        channel = channel.copy()
        isnan = np.isnan(channel)
        n = len(channel)

        # Forward-fill up to 3 consecutive missing values
        i = 0
        while i < n:
            if isnan[i]:
                # Start of missing block
                start = i
                while i < n and isnan[i]:
                    i += 1
                end = i
                gap = end - start
                # If previous value exists and gap <= 3, forward fill
                if start > 0 and gap <= 3 and not isnan[start - 1]:
                    channel[start:end] = channel[start - 1]
                # Otherwise, leave as np.nan for iterative imputation
            else:
                i += 1

        # This clearly does not work 
        # If any np.nan remain, use IterativeImputer
        if np.any(np.isnan(channel)):
            if other_channels is None:
                raise ValueError("other_channels must be provided for iterative imputation.")
            # Stack channel with other_channels for imputation
            # Shape: (n_timepoints, n_features)
            data = np.column_stack([channel, other_channels])
            imputer = IterativeImputer(estimator=None, max_iter=10, random_state=0, sample_posterior=False)
            imputed = imputer.fit_transform(data)
            channel = imputed[:, 0]

        return channel


def create_imputer(method: str = 'hinrichs') -> AbstractImputer:
    """
    Factory function to create an imputer instance based on the specified method.
    
    Args:
        method: Imputation method to use (e.g., 'hinrichs')
        
    Returns:
        Instance of AbstractImputer subclass
    """
    if method == 'hinrichs_paper':
        return HinrichsPaperImputer()
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    

            