import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

def sequestration_rhs(x, t, a1, b1, a2, b2, g1, g2, d, ct):
    """
    Defines the ODE system tracking competitive sequestration and resource allocation.
    """
    S1, A1, S2, A2, C1, C2 = x
    C = max(0.0, ct - C1 - C2)
    
    dS1 = a1 - d*S1 - g1*A1*S1 - g2*S1*C
    dA1 = b1 - d*A1 - g1*A1*S1
    dS2 = a2 - d*S2 - g1*A2*S2 - g2*S2*C
    dA2 = b2 - d*A2 - g1*A2*S2
    dC1 = g2*S1*C - d*C1
    dC2 = g2*S2*C - d*C2
    
    return np.array([dS1, dA1, dS2, dA2, dC1, dC2])

def compute_model_prediction(params, conditions, t_max=10.0):
    """
    Generates steady-state predictions for the partitioned complex fraction.
    """
    g1, g2, ct = params
    d_fixed = 1.0
    predictions = []
    t_span = np.linspace(0, t_max, 100)
    x0 = np.zeros(6)
    
    for cond in conditions:
        a1, b1, a2, b2 = cond
        sol = scipy.integrate.odeint(
            sequestration_rhs, x0, t_span, 
            args=(a1, b1, a2, b2, g1, g2, d_fixed, ct)
        )
        predictions.append(sol[-1, 4] / ct)
        
    return np.array(predictions)

def cross_validation_loss(params, conditions, targets):
    """Calculates residual sum of squares minimization criteria."""
    predictions = compute_model_prediction(params, conditions)
    return np.sum((targets - predictions) ** 2)

def run_leave_one_condition_out_cv(conditions, targets, bounds):
    """
    Executes a Leave-One-Condition-Out Cross-Validation routine.
    """
    num_points = len(conditions)
    cv_predictions = np.zeros(num_points)
    initial_guess = [10.0, 50.0, 0.5]
    
    print("Running Leave-One-Condition-Out Cross-Validation on Panel c data...")
    for left_out_idx in range(num_points):
        train_conds = [conditions[m] for m in range(num_points) if m != left_out_idx]
        train_targets = np.delete(targets, left_out_idx)
        val_cond = [conditions[left_out_idx]]
        
        optimization_result = scipy.optimize.minimize(
            cross_validation_loss, initial_guess, 
            args=(train_conds, train_targets), 
            bounds=bounds, method='L-BFGS-B'
        )
        
        predicted_val = compute_model_prediction(optimization_result.x, val_cond)
        cv_predictions[left_out_idx] = predicted_val[0]
        
    return cv_predictions

def load_panel_c_data(excel_path, sheet_name="Panel c"):
    """
    The exact Excel loader method matching your group member's script.
    Parses column structures and handles missing rows/tabs dynamically.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    renamed_columns = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "cdil" in col_lower:
            renamed_columns[col] = "Cdil"
        elif "dgtr" in col_lower or "delta" in col_lower or "δg" in col_lower:
            renamed_columns[col] = "DGtr"

    df = df.rename(columns=renamed_columns)
    
    df["Cdil"] = pd.to_numeric(df["Cdil"], errors="coerce")
    df["DGtr"] = pd.to_numeric(df["DGtr"], errors="coerce")
    df = df.dropna(subset=["Cdil", "DGtr"]).copy()
    
    df = df.sort_values("Cdil").reset_index(drop=True)
    return df

def generate_and_save_cv_metrics(targets, predictions, save_dir):
    """Plots and validates observed values versus out-of-fold model predictions."""
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(targets, predictions, color='#2ca02c', edgecolor='k', zorder=3, s=45)
    
    identity_line = np.linspace(0, 1, 50)
    plt.plot(identity_line, identity_line, color='black', linestyle='--', linewidth=1.2, zorder=2)
    
    residual_sum_squares = np.sum((targets - predictions) ** 2)
    total_sum_squares = np.sum((targets - np.mean(targets)) ** 2)
    q2y_score = 1.0 - (residual_sum_squares / total_sum_squares)
    
    plt.title(f"LOCO Cross-Validation Metrics ($Q^2_Y$ = {q2y_score:.3f})", fontsize=11, color='black')
    plt.xlabel("True Experimental Fraction (from $\Delta G_{tr}$)", fontsize=10, color='black')
    plt.ylabel("Predicted Model Fraction", fontsize=10, color='black')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    output_filename = save_dir / "cross_validation_real_data.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

def main():
    # Setup CLI parser to match your group's run scheme
    parser = argparse.ArgumentParser(description="Cross-validation fitting pipeline.")
    parser.add_argument("--data", required=True, help="Path to the source-data Excel file.")
    parser.add_argument("--sheet", default="Panel c", help="Excel sheet tab to parse.")
    args = parser.parse_args()

    # Automatically resolves the Figures path relative to where this script lives
    script_dir = Path(__file__).resolve().parent
    figures_directory = script_dir.parent / "Figures"
    
    # Ingest data matching the exact group method
    df = load_panel_c_data(args.data, sheet_name=args.sheet)
    
    # Thermodynamic translation equations
    RT = 1.9872e-3 * 298.15 
    k_partition = np.exp(-df['DGtr'] / RT)
    measured_targets = k_partition / (1 + k_partition)
    measured_targets = (measured_targets - measured_targets.min()) / (measured_targets.max() - measured_targets.min())
    
    experimental_conditions = []
    for c_dil in df['Cdil']:
        experimental_conditions.append((c_dil, 0.0, 0.0, 0.0))
        
    parameter_search_bounds = [(0.1, 500.0), (0.1, 500.0), (0.01, 2.0)]
    
    # Run cross-validation analysis loop
    cv_predictions = run_leave_one_condition_out_cv(
        experimental_conditions, measured_targets, parameter_search_bounds
    )
    
    # Save the cross-validation plot directly into the resolved Figures directory
    generate_and_save_cv_metrics(measured_targets, cv_predictions, figures_directory)
    print(f"\nWorkflow complete! Validation metrics chart exported to:\n{figures_directory}")

if __name__ == "__main__":
    main()