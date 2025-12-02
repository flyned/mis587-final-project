"""
Inference Script for Trained Models (Random Forest or Neural Network)

This script provides a simple interface for making predictions with
either a Random Forest or Neural Network model.

Supports:
- Random Forest models (.pkl files)
- Neural Network models (.keras files)

Usage:
    # Load model and make predictions
    from predict import CoStarPredictor

    # For Random Forest
    predictor = CoStarPredictor(
        model_path='./models/rf_model_20250128_143022.pkl',
        metadata_path='./models/rf_metadata_20250128_143022.json'
    )

    # For Neural Network
    predictor = CoStarPredictor(
        model_path='./models/nn_model_20250128_143022.keras',
        metadata_path='./models/nn_metadata_20250128_143022.json'
    )

    # Make predictions on new data
    new_data = pd.read_csv('./data/test.csv')
    predictions = predictor.predict(new_data)
    print(predictions)

    # Or use from command line
    python predict.py --model ./models/rf_model_20250128_143022.pkl --data ./data/test.csv
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path

from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling, evaluate_model, print_metrics


class CoStarPredictor:
    """
    Wrapper class for making predictions with trained models.

    Supports both:
    - Random Forest models (.pkl files)
    - Neural Network models (.keras files)

    This class handles:
    - Auto-detecting model type from file extension
    - Loading the trained model
    - Applying feature engineering pipeline
    - Ensuring feature alignment between training and inference
    - Making predictions on new data
    - Formatting output with predictions and errors
    """

    def __init__(self, model_path, metadata_path=None):
        """
        Initialize predictor with saved model.

        Args:
            model_path: Path to model file (.pkl for RF, .keras for NN)
            metadata_path: Optional path to metadata JSON file
        """
        print("="*70)
        print("COSTAR PREDICTOR - LOADING MODEL")
        print("="*70)

        model_path = str(model_path)

        # Detect model type from extension
        if model_path.endswith('.keras') or model_path.endswith('.h5'):
            self.model_type = 'neural_network'
            self._load_neural_network(model_path, metadata_path)
        else:
            self.model_type = 'random_forest'
            self._load_random_forest(model_path, metadata_path)

        print("\n" + "="*70)
        print("Model ready for predictions!")
        print("="*70)

    def _load_random_forest(self, model_path, metadata_path):
        """Load Random Forest model from pickle file."""
        print(f"\nLoading Random Forest model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model_dict = pickle.load(f)

        self.model = self.model_dict['model']
        self.scaler = self.model_dict.get('scaler')
        self.nn_scaler = None  # Not used for RF

        print(f"Model loaded: {type(self.model).__name__}")
        if hasattr(self.model, 'n_estimators'):
            print(f"  n_estimators: {self.model.n_estimators}")
        if hasattr(self.model, 'max_depth'):
            print(f"  max_depth: {self.model.max_depth}")

        # Load metadata
        self._load_metadata(metadata_path, is_nn=False)

    def _load_neural_network(self, model_path, metadata_path):
        """Load Neural Network model from Keras file."""
        print(f"\nLoading Neural Network model from: {model_path}")

        try:
            from tensorflow import keras
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("TensorFlow is required for neural network models. "
                            "Install with: pip install tensorflow>=2.13.0")

        # Load Keras model
        self.model = keras.models.load_model(model_path)
        self.scaler = None  # RF scaler not used

        print(f"Model loaded: Neural Network (Keras)")
        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Output shape: {self.model.output_shape}")

        # Load metadata and reconstruct scaler
        self._load_metadata(metadata_path, is_nn=True)

    def _load_metadata(self, metadata_path, is_nn=False):
        """Load metadata and scaler parameters."""
        self.metadata = None
        self.feature_names = None
        self.target_column = 'Rent/SF/Yr'
        self.nn_scaler = None

        if metadata_path:
            print(f"\nLoading metadata from: {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            self.feature_names = self.metadata.get('feature_names', [])
            self.target_column = self.metadata.get('target_column', 'Rent/SF/Yr')

            print(f"Metadata loaded")
            print(f"  Features: {self.metadata.get('feature_count', 'Unknown')}")
            print(f"  Target: {self.target_column}")

            # Get metrics - handle both old and new key formats
            val_metrics = self.metadata.get('validation_metrics', self.metadata.get('val_metrics', {}))
            test_metrics = self.metadata.get('test_metrics', {})

            # Handle both uppercase and lowercase metric keys
            val_r2 = val_metrics.get('r2', val_metrics.get('R2', val_metrics.get('R²', 'N/A')))
            test_r2 = test_metrics.get('r2', test_metrics.get('R2', test_metrics.get('R²', 'N/A')))

            if isinstance(val_r2, (int, float)):
                print(f"  Validation R2: {val_r2:.3f}")
            if isinstance(test_r2, (int, float)):
                print(f"  Test R2: {test_r2:.3f}")

            # Reconstruct scaler for neural network
            if is_nn and 'scaler_mean' in self.metadata:
                from sklearn.preprocessing import StandardScaler
                self.nn_scaler = StandardScaler()
                self.nn_scaler.mean_ = np.array(self.metadata['scaler_mean'])
                self.nn_scaler.scale_ = np.array(self.metadata['scaler_scale'])
                self.nn_scaler.var_ = self.nn_scaler.scale_ ** 2
                self.nn_scaler.n_features_in_ = len(self.nn_scaler.mean_)
                print(f"  Scaler reconstructed for NN")

    def predict(self, data, include_actual=True):
        """
        Make predictions on new data.

        This method:
        1. Applies feature engineering pipeline
        2. Prepares data for modeling (alignment, encoding)
        3. Makes predictions using trained model
        4. Formats output with predictions and errors

        Args:
            data: DataFrame with raw property data (same format as training data)
            include_actual: If True and target column exists, include actual values and errors

        Returns:
            DataFrame with columns:
                - Property Address
                - Predicted_Rent_SF_Yr
                - Actual_Rent_SF_Yr (if target column exists)
                - Error (if target column exists)
                - Percent_Error (if target column exists)
        """
        print("\n" + "="*70)
        print("MAKING PREDICTIONS")
        print("="*70)

        print(f"\nInput data: {data.shape}")

        # Step 1: Feature engineering
        print("\n1. Applying feature engineering...")
        data_eng = engineer_features(data.copy(), target_col=self.target_column)
        print(f"   ✓ Engineered features: {data_eng.shape}")

        # Step 2: Prepare for modeling
        print("\n2. Preparing data for modeling...")

        # Check if target column exists
        has_target = self.target_column in data_eng.columns

        if has_target:
            X, y, _, _ = prepare_data_for_modeling(
                data_eng,
                target_column=self.target_column,
                ref_columns=self.feature_names
            )
        else:
            # If no target, create a dummy one
            data_eng[self.target_column] = 0
            X, _, _, _ = prepare_data_for_modeling(
                data_eng,
                target_column=self.target_column,
                ref_columns=self.feature_names
            )
            y = None

        print(f"   ✓ Prepared features: {X.shape}")

        # Step 3: Apply scaling if needed
        if self.model_type == 'neural_network' and self.nn_scaler:
            print("\n3. Applying feature scaling (Neural Network)...")
            X = self.nn_scaler.transform(X)
        elif self.scaler:
            print("\n3. Applying feature scaling (Random Forest)...")
            X = self.scaler.transform(X)

        # Step 4: Make predictions
        print("\n4. Making predictions...")
        if self.model_type == 'neural_network':
            predictions = self.model.predict(X, verbose=0).flatten()
        else:
            predictions = self.model.predict(X)
        print(f"   ✓ Predictions made: {len(predictions)}")

        # Step 5: Format results
        print("\n5. Formatting results...")

        # Start with property address
        result = pd.DataFrame()
        if 'Property Address' in data.columns:
            result['Property Address'] = data['Property Address'].values

        result['Predicted_Rent_SF_Yr'] = predictions

        # Add actual values and errors if available
        if has_target and include_actual:
            result['Actual_Rent_SF_Yr'] = y
            result['Error'] = predictions - y
            result['Percent_Error'] = (result['Error'] / y) * 100

            # Calculate metrics
            print("\n" + "="*70)
            print("PREDICTION METRICS")
            print("="*70)
            metrics = evaluate_model(y, predictions, name="Predictions")
            print_metrics(metrics)

        print("\n" + "="*70)
        print("PREDICTIONS COMPLETE!")
        print("="*70)
        print(f"\nPredicted {len(predictions)} properties")
        print(f"Mean predicted rent: ${predictions.mean():.2f}/SF/Yr")
        print(f"Median predicted rent: ${pd.Series(predictions).median():.2f}/SF/Yr")
        print(f"Range: ${predictions.min():.2f} - ${predictions.max():.2f}/SF/Yr")

        return result

    def predict_single(self, property_data):
        """
        Make prediction for a single property.

        Args:
            property_data: Dictionary or Series with property features

        Returns:
            Float prediction
        """
        if isinstance(property_data, dict):
            df = pd.DataFrame([property_data])
        else:
            df = pd.DataFrame([property_data.to_dict()])

        result = self.predict(df, include_actual=False)
        return result['Predicted_Rent_SF_Yr'].iloc[0]

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': self.model_type,
            'model_class': type(self.model).__name__ if self.model_type == 'random_forest' else 'NeuralNetwork',
            'target_column': self.target_column
        }

        if self.metadata:
            # Get metrics - handle both old and new key formats
            val_metrics = self.metadata.get('validation_metrics', self.metadata.get('val_metrics', {}))
            test_metrics = self.metadata.get('test_metrics', {})

            # Handle both uppercase and lowercase metric keys
            val_r2 = val_metrics.get('r2', val_metrics.get('R2', val_metrics.get('R²', None)))
            test_r2 = test_metrics.get('r2', test_metrics.get('R2', test_metrics.get('R²', None)))
            val_mae = val_metrics.get('mae', val_metrics.get('MAE', None))
            test_mae = test_metrics.get('mae', test_metrics.get('MAE', None))

            info.update({
                'training_date': self.metadata.get('training_date'),
                'feature_count': self.metadata.get('feature_count'),
                'training_samples': self.metadata.get('training_samples'),
                'val_r2': val_r2,
                'test_r2': test_r2,
                'val_mae': val_mae,
                'test_mae': test_mae
            })

        return info


def main():
    """Command-line interface for making predictions."""
    parser = argparse.ArgumentParser(
        description='Make predictions with trained Random Forest model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to pickled model file (.pkl)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Path to metadata JSON file (optional)'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV file with property data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions CSV (optional)'
    )

    args = parser.parse_args()

    # Load predictor
    predictor = CoStarPredictor(
        model_path=args.model,
        metadata_path=args.metadata
    )

    # Load data
    print(f"\nLoading data from: {args.data}")
    data = pd.read_csv(args.data)

    # Make predictions
    predictions = predictor.predict(data)

    # Display first few predictions
    print("\nFirst 10 predictions:")
    print(predictions.head(10).to_string(index=False))

    # Save if output path provided
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\n✓ Predictions saved to: {args.output}")


if __name__ == "__main__":
    # Example usage when run as script
    import sys

    if len(sys.argv) == 1:
        # No arguments - run example
        print("\nNo arguments provided. Example usage:\n")
        print("python predict.py \\")
        print("  --model ./models/rf_model_20250128_143022.pkl \\")
        print("  --metadata ./models/rf_metadata_20250128_143022.json \\")
        print("  --data ./data/test.csv \\")
        print("  --output ./predictions.csv")
        print("\nOr import and use programmatically:")
        print("from predict import CoStarPredictor")
        print("predictor = CoStarPredictor(model_path='...', metadata_path='...')")
        print("predictions = predictor.predict(data)")
    else:
        main()
