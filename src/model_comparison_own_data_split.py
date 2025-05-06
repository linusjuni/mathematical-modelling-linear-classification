import numpy as np
import pandas as pd
from evaluation_utils import correlated_t_test
from part_1_own_data_split import run_part1_own_data_split
from part_2_own_data_split import run_part2_own_data_split
from part_3_own_data_split import run_part3_own_data_split

def compare_models():
    print("Running Part 1 model...")
    part1_results = run_part1_own_data_split()
    
    print("\nRunning Part 2 model (Sobel features)...")
    part2_results = run_part2_own_data_split()
    
    print("\nRunning Part 3 model (Histogram of Orientation features)...")
    part3_results = run_part3_own_data_split()
    
    part1_cv_df = part1_results['cv_df']
    part2_cv_df = part2_results['cv_df']
    part3_cv_df = part3_results['cv_df']
    
    assert len(part1_cv_df) == len(part2_cv_df) == len(part3_cv_df), "Different number of folds!"
    
    K_fold = len(part1_cv_df)
    
    models = {
        "Part 1 (Raw)": part1_cv_df,
        "Part 2 (Sobel)": part2_cv_df,
        "Part 3 (Histogram)": part3_cv_df
    }
    
    metrics = ["accuracy", "auc"]
    
    print("\n===== MODEL COMPARISON RESULTS =====")
    
    # Perform all pairwise comparisons
    for i, (model1_name, model1_df) in enumerate(models.items()):
        for j, (model2_name, model2_df) in enumerate(models.items()):
            if i >= j:  # Skip self-comparisons and repeats
                continue
                
            print(f"\n----- {model2_name} vs {model1_name} -----")
            
            for metric in metrics:
                diffs = model2_df[metric].values - model1_df[metric].values
                
                test_result = correlated_t_test(diffs, K_fold=K_fold)
                
                print(f"\n{metric.upper()} Comparison:")
                print(f"Mean difference ({model2_name} - {model1_name}): {test_result['mean_diff']:.4f}")
                print(f"95% Confidence Interval: [{test_result['ci'][0]:.4f}, {test_result['ci'][1]:.4f}]")
                print(f"t-statistic: {test_result['t_stat']:.4f}")
                print(f"p-value: {test_result['p_value']:.4f}")
                
                if test_result['p_value'] < 0.05:
                    better = model2_name if test_result['mean_diff'] > 0 else model1_name
                    print(f"Significant difference detected! {better} is better in terms of {metric}.")
                else:
                    print(f"No significant difference in {metric} between the models.")
    
    print("\n===== OVERALL MODEL PERFORMANCE =====")
    for model_name, model_df in models.items():
        print(f"\n{model_name}:")
        for metric in metrics:
            mean_val = model_df[metric].mean()
            std_val = model_df[metric].std()
            print(f"Mean {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    return {
        "part1_results": part1_results,
        "part2_results": part2_results,
        "part3_results": part3_results
    }

if __name__ == "__main__":
    compare_models()