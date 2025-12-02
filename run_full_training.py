"""
Run full training directly without input prompt
"""

from train_local_memory_optimized import train_and_save_model_local

if __name__ == "__main__":
    print("\nStarting full training with hyperparameter tuning...")
    print("This will take approximately 30-45 minutes")
    print("Memory usage will be monitored\n")

    results = train_and_save_model_local(
        tune_hyperparameters=True,   # Full hyperparameter tuning
        cv_folds=3,                  # 3-fold CV
        monitor_memory=True
    )

    print("\n" + "="*70)
    print("FULL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest model saved to: {results['paths']['model']}")
