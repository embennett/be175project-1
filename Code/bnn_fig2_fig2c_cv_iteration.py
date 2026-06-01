"""
bnn_fig2_fig2c_cv_iteration.py

This script applies a Stratified 5-Fold Cross-Validation pipeline to your 
original time-series ODE framework, utilizing the scaling and energy-mapping 
parameters established by your group mate.

Original model idea:
    molecular input concentration -> nonlinear transcriptional output

Condensate extension:
    dilute-phase NPM1 concentration -> transfer free energy of condensate partitioning

For Riback Figure 2C:
    Cdil = NPM1 concentration in the dilute phase outside the nucleolus
    DGtr = transfer free energy for moving NPM1 into the nucleolus

Run from the repository root with:

& "H:\My Drive\2025-2026 School Year\Spring26\BE175\be175project-1\be175final.venv\Scripts\python.exe" "Code/bnn_fig2_fig2c_cv_iteration.py" \
    --data "Data/41586_2020_2256_MOESM5_ESM.xlsx" \
    --sheet "Panel c"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

def sequestration_rhs(x, t, a1, b, g1, g2, d, ct):
    """
    Your core ODE system tracking competitive sequestration and resource allocation.
    Uses 'b' as a constant input driving the competitor branch A1.
    """
    S1, A1, S2, A2, C1, C2 = x
    C = max(0.0, ct - C1 - C2)
    
    # a1 is the scaled Cdil input, b is the fixed group mate baseline competitor influx
    dS1 = a1 - d*S1 - g1*A1*S1 - g2*S1*C
    dA1 = b - d*A1 - g1*A1*S1
    dS2 = 0.0 - d*S2 - g1*A2*S2 - g2*S2*C
    dA2 = 0.0 - d*A2 - g1*A2*S2
    dC1 = g2*S1*C - d*C1
    dC2 = g2*S2*C - d*C2
    
    return np.array([dS1, dA1, dS2, dA2, dC1, dC2])

"""
def compute_ode_predictions(Cdil, params, t_max=15.0):

    # Generates steady-state predictions using your integration loop, 
    # structuring inputs and outputs according to your group mate's parameters.
    
    b, C_scale, DG_min, DG_range = params
    
    # Hardcoded biological constants from your group mate's framework
    g1_fixed = 100.0
    g2_fixed = 10.0
    d_fixed = 1.0
    ct_fixed = 0.2  # 1/5
    
    predictions = []
    t_span = np.linspace(0, t_max, 100)
    x0 = np.zeros(6)
    
    for c_dil_val in Cdil:
        # Scale the raw experimental concentration into dimensionless model input space
        a1_scaled = c_dil_val / C_scale
        
        sol = scipy.integrate.odeint(
            sequestration_rhs, x0, t_span, 
            args=(a1_scaled, b, g1_fixed, g2_fixed, d_fixed, ct_fixed)
        )
        
        # Calculate the bound resource activation fraction
        activation_fraction = sol[-1, 4] / ct_fixed
        
        # Map the fractional activation directly to raw energy units (kcal/mol)
        predicted_dgtr = DG_min + DG_range * activation_fraction
        predictions.append(predicted_dgtr)
        
    return np.array(predictions)
"""
def compute_ode_predictions(Cdil, params, t_max=15.0):
    # Added gamma as a 5th parameter to control high-concentration curvature
    b, C_scale, DG_min, DG_range, gamma = params
    
    g1_fixed, g2_fixed, d_fixed, ct_fixed = 100.0, 10.0, 1.0, 0.2
    predictions = []
    t_span = np.linspace(0, t_max, 100)
    x0 = np.zeros(6)
    
    for c_dil_val in Cdil:
        a1_scaled = c_dil_val / C_scale
        sol = scipy.integrate.odeint(
            sequestration_rhs, x0, t_span, 
            args=(a1_scaled, b, g1_fixed, g2_fixed, d_fixed, ct_fixed)
        )
        
        activation_fraction = sol[-1, 4] / ct_fixed
        
        # Power-law transformation breaks the hard linear saturation ceiling
        transformed_fraction = np.power(activation_fraction, gamma)
        
        predicted_dgtr = DG_min + DG_range * transformed_fraction
        predictions.append(predicted_dgtr)
        
    return np.array(predictions)

def cross_validation_loss(params, Cdil_train, DGtr_train):
    """Calculates residual sum of squares on raw energy metrics."""
    predictions = compute_ode_predictions(Cdil_train, params)
    return np.sum((DGtr_train - predictions) ** 2)

def run_stratified_gradient_cv(Cdil, DGtr, bounds, n_folds=5):
    """
    Distributes points into 5 stratified folds along the concentration gradient.
    Optimizes the 4 mapping parameters within your time-series ODE engine.
    """
    num_points = len(Cdil)
    cv_predictions = np.zeros(num_points)
    
    # Baseline guesses generated from raw dataset structure
   # Added 1.0 to the end of the list as the baseline guess for gamma
    initial_guess = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr), 1.0]
    
    # Interleave index assignments along the pre-sorted Cdil gradient
    fold_assignments = np.array([i % n_folds for i in range(num_points)])
    
    print(f"Starting {n_folds}-Fold CV")
    
    for fold_idx in range(n_folds):
        print(f"Fold {fold_idx + 1}/{n_folds}... ", end="", flush=True)
        
        val_indices = np.where(fold_assignments == fold_idx)[0]
        train_indices = np.where(fold_assignments != fold_idx)[0]
        
        Cdil_train, DGtr_train = Cdil[train_indices], DGtr[train_indices]
        Cdil_val = Cdil[val_indices]
        
        opt_res = scipy.optimize.minimize(
            cross_validation_loss, initial_guess,
            args=(Cdil_train, DGtr_train),
            bounds=bounds, method='L-BFGS-B'
        )
        
        # Evaluate out-of-fold performance using the optimized parameters
        predicted_vals = compute_ode_predictions(Cdil_val, opt_res.x)
        cv_predictions[val_indices] = predicted_vals
        print("Done.")
        
    return cv_predictions

def load_panel_c_data(excel_path, sheet_name="Panel c"):
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
    
    # Sort by concentration to guarantee proper stratified sampling splits
    df = df.sort_values("Cdil").reset_index(drop=True)
    return df

def generate_and_save_cv_metrics(y_true, y_pred, save_dir):
    """Generates the true vs. predicted 1:1 cross-validation scatter plot."""
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(y_true, y_pred, color='#2ca02c', edgecolor='k', zorder=3, s=45)
    
    min_val = min(y_true.min(), y_pred.min()) - 0.2
    max_val = max(y_true.max(), y_pred.max()) + 0.2
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.2, zorder=2)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    q2y_score = 1.0 - (ss_res / ss_tot)
    
    plt.title(f"5-Fold Cross-Validation ($Q^2_Y$ = {q2y_score:.3f})", fontsize=16, color='black')
    plt.xlabel("True Experimental $\Delta G_{tr}$ (kcal/mol)", fontsize=14, color='black')
    plt.ylabel("Predicted Model $\Delta G_{tr}$ (kcal/mol)", fontsize=14, color='black')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    output_filename = save_dir / "cross_validation_real_data.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_continuous_model_fit(Cdil, DGtr, cv_predictions, bounds, save_dir):
    """
    Generates a continuous line plot showing the true data points,
    the out-of-fold cross-validation values, and the smooth model prediction trace.
    """
    plt.figure(figsize=(6.0, 4.5))
    
    # 1. Plot the raw experimental data points
    plt.scatter(Cdil, DGtr, color='#1f77b4', edgecolor='k', alpha=0.8, s=50, label="Experimental Data", zorder=3)
    
    # 2. Plot the out-of-fold cross-validation predictions for comparison
    plt.scatter(Cdil, cv_predictions, color='#2ca02c', marker='x', s=45, alpha=0.9, label="Out-of-Fold CV Predictions", zorder=4)
    
    # 3. Generate a smooth, high-density grid across the Cdil range to trace the true model line
    Cdil_grid = np.linspace(np.min(Cdil), np.max(Cdil), 300)
    
    # Run a global fit over all data to determine parameters for the clean trace line
    # Added 1.0 to the end of the list as the baseline guess for gamma
    initial_guess = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr), 1.0]
    opt_res = scipy.optimize.minimize(
        cross_validation_loss, initial_guess,
        args=(Cdil, DGtr), bounds=bounds, method='L-BFGS-B'
    )
    grid_predictions = compute_ode_predictions(Cdil_grid, opt_res.x)
    
    plt.plot(Cdil_grid, grid_predictions, color="#ff0e62", linewidth=2.5, label="Optimized ODE Model Fit", zorder=2)
    
    # Visual Polish
    plt.title("Competitive ODE Model Fit vs. Partitioning Data", fontsize=16, color='black')
    plt.xlabel("Dilute-Phase Concentration ($C_{dil}$)", fontsize=14, color='black')
    plt.ylabel("Transfer Free Energy $\Delta G_{tr}$ (kcal/mol)", fontsize=14, color='black')
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    output_filename = save_dir / "model_fit_vs_experimental.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ODE Pipeline Aligned to Groupmate Parameters.")
    parser.add_argument("--data", required=True, help="Path to the source-data Excel file.")
    parser.add_argument("--sheet", default="Panel c", help="Excel sheet tab to parse.")
    args = parser.parse_args()

    # Hardcoded absolute directory location to bypass virtual/stream path sync tracking
    figures_directory = Path(r"H:\My Drive\2025-2026 School Year\Spring26\BE175\be175project-1\Figures")
    figures_directory.mkdir(parents=True, exist_ok=True)
    
    df = load_panel_c_data(args.data, sheet_name=args.sheet)
    
    Cdil = df["Cdil"].to_numpy(dtype=float)
    DGtr = df["DGtr"].to_numpy(dtype=float)
    
    # Paired sequential tuples precisely matching SciPy parameter unpacking specifications
    # Added the 5th tuple to constrain gamma
    bnn_bounds = (
        (0.0, 5.0),                       # Bounds for competitor influx 'b'
        (1e-6, np.max(Cdil) * 10),        # Bounds for 'C_scale'
        (-20.0, 20.0),                    # Bounds for 'DG_min'
        (-20.0, 20.0),                    # Bounds for 'DG_range'
        (0.1, 8.0)                        # Bounds for the power-law exponent 'gamma'
    )
    
    # Run cross-validation loop
    cv_predictions = run_stratified_gradient_cv(Cdil, DGtr, bnn_bounds, n_folds=5)
    
    # Output 1: Generate cross-validation metrics correlation scatter plot
    generate_and_save_cv_metrics(DGtr, cv_predictions, figures_directory)
    
    # Output 2: Generate the physical continuous model-fit curve trace plot
    plot_continuous_model_fit(Cdil, DGtr, cv_predictions, bnn_bounds, figures_directory)
    
    print(f"\nDone! Both plots saved to:\n{figures_directory}")

if __name__ == "__main__":
    main()