"""
Quick Test - Memory-Optimized Training (No Hyperparameter Tuning)

This runs a quick test (5 minutes) to verify:
1. Memory usage stays under 2GB
2. Training completes successfully
3. Model performs reasonably well

After this succeeds, run train_local_memory_optimized.py for full training.
"""

from train_local_memory_optimized import train_and_save_model_local

if __name__ == "__main__":
    print("="*70)
    print("QUICK TEST - NO HYPERPARAMETER TUNING")
    print("="*70)
    print("\nThis will:")
    print("  - Train Random Forest with default parameters")
    print("  - Expected time: ~5 minutes")
    print("  - Expected memory: 1-2 GB")
    print("  - No GridSearchCV (just to verify pipeline works)")
    print("\n" + "="*70 + "\n")

    results = train_and_save_model_local(
        tune_hyperparameters=False,  # No hyperparameter tuning
        cv_folds=3,
        monitor_memory=True
    )

    print("\n" + "="*70)
    print("QUICK TEST COMPLETE!")
    print("="*70)
    print(f"\nValidation R²: {results['val_metrics']['R²']:.3f}")
    print(f"Test R²: {results['test_metrics']['R²']:.3f}")
    print(f"\nModel saved to: {results['paths']['model']}")
    print("\nIf this worked well, you can now run:")
    print("  python train_local_memory_optimized.py")
    print("\nfor full hyperparameter tuning (30-45 minutes)")
