"""
bnn_fig2_fig2c_cv_iteration.py

This script applies a Stratified 5-Fold Cross-Validation pipeline to your 
original time-series ODE framework, utilizing the scaling and energy-mapping 
parameters established by your group mate. Includes ODE stability safeguards.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

def sequestration_rhs(t, x, a1, b, g1, g2, d, ct):
    """
    Core ODE system tracking competitive sequestration and resource allocation.
    NOTE: solve_ivp requires the time variable 't' as the FIRST argument.
    """
    S1, A1, S2, A2, C1, C2 = x
    C = max(0.0, ct - C1 - C2)
    
    dS1 = a1 - d*S1 - g1*A1*S1 - g2*S1*C
    dA1 = b - d*A1 - g1*A1*S1
    dS2 = 0.0 - d*S2 - g1*A2*S2 - g2*S2*C
    dA2 = 0.0 - d*A2 - g1*A2*S2
    dC1 = g2*S1*C - d*C1
    dC2 = g2*S2*C - d*C2
    
    return [dS1, dA1, dS2, dA2, dC1, dC2]

def compute_ode_predictions(Cdil, params, t_max=15.0):
    """
    Generates steady-state predictions using solve_ivp with safety boundaries
    to handle numerical convergence failures automatically.
    """
    b, C_scale, DG_min, DG_range, gamma = params
    
    g1_fixed = 100.0
    g2_fixed = 10.0
    d_fixed = 1.0
    ct_fixed = 0.2
    
    predictions = []
    x0 = np.zeros(6)
    
    for c_dil_val in Cdil:
        a1_scaled = c_dil_val / C_scale
        
        # solve_ivp safely manages integration tolerances to prevent infinite loops
        sol = scipy.integrate.solve_ivp(
            sequestration_rhs, 
            t_span=(0.0, t_max), 
            y0=x0, 
            method='LSODA', 
            args=(a1_scaled, b, g1_fixed, g2_fixed, d_fixed, ct_fixed),
            max_step=1.0  # Prevents infinite sub-stepping loops
        )
        
        # Safe fallback: if integration failed or blew up, penalize with an unfeasible energy value
        if not sol.success or np.any(np.isnan(sol.y)):
            predictions.append(99.0)
            continue
            
        activation_fraction = sol.y[4, -1] / ct_fixed
        
        # Ensure fraction stays in physical bounds before calculating power law
        activation_fraction = np.clip(activation_fraction, 0.0, 1.0)
        transformed_fraction = np.power(activation_fraction, gamma)
        
        predicted_dgtr = DG_min + DG_range * transformed_fraction
        predictions.append(predicted_dgtr)
        
    return np.array(predictions)

def cross_validation_loss(params, Cdil_train, DGtr_train):
    """Calculates residual sum of squares with protection against unstable regions."""
    predictions = compute_ode_predictions(Cdil_train, params)
    
    # Catch any bad fallbacks and penalize heavily to push optimizer away
    if np.any(predictions == 99.0):
        return 1e10
        
    return np.sum((DGtr_train - predictions) ** 2)

def run_stratified_gradient_cv(Cdil, DGtr, bounds, n_folds=5):
    num_points = len(Cdil)
    cv_predictions = np.zeros(num_points)
    
    initial_guess = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr), 1.0]
    fold_assignments = np.array([i % n_folds for i in range(num_points)])
    
    print(f"Starting Aligned ODE Stratified {n_folds}-Fold CV")
    print("-" * 65)
    
    for fold_idx in range(n_folds):
        print(f"Processing Fold {fold_idx + 1}/{n_folds}... ", end="", flush=True)
        
        val_indices = np.where(fold_assignments == fold_idx)[0]
        train_indices = np.where(fold_assignments != fold_idx)[0]
        
        Cdil_train, DGtr_train = Cdil[train_indices], DGtr[train_indices]
        Cdil_val = Cdil[val_indices]
        
        opt_res = scipy.optimize.minimize(
            cross_validation_loss, initial_guess,
            args=(Cdil_train, DGtr_train),
            bounds=bounds, method='L-BFGS-B'
        )
        
        predicted_vals = compute_ode_predictions(Cdil_val, opt_res.x)
        cv_predictions[val_indices] = predicted_vals
        print("Done.")
        
    print("-" * 65)
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
    
    df = df.sort_values("Cdil").reset_index(drop=True)
    return df

def generate_and_save_cv_metrics(y_true, y_pred, save_dir):
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(y_true, y_pred, color='#2ca02c', edgecolor='k', zorder=3, s=45)
    
    min_val = min(y_true.min(), y_pred.min()) - 0.2
    max_val = max(y_true.max(), y_pred.max()) + 0.2
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.2, zorder=2)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    q2y_score = 1.0 - (ss_res / ss_tot)
    
    plt.title(f"Stratified 5-Fold Cross-Validation ($Q^2_Y$ = {q2y_score:.3f})", fontsize=16, color='black')
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
    plt.figure(figsize=(6.0, 4.5))
    plt.scatter(Cdil, DGtr, color='#1f77b4', edgecolor='k', alpha=0.8, s=50, label="Experimental Data", zorder=3)
    plt.scatter(Cdil, cv_predictions, color='#2ca02c', marker='x', s=45, alpha=0.9, label="Out-of-Fold CV Predictions", zorder=4)
    
    Cdil_grid = np.linspace(np.min(Cdil), np.max(Cdil), 300)
    initial_guess = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr), 1.0]
    
    opt_res = scipy.optimize.minimize(
        cross_validation_loss, initial_guess,
        args=(Cdil, DGtr), bounds=bounds, method='L-BFGS-B'
    )
    grid_predictions = compute_ode_predictions(Cdil_grid, opt_res.x)
    
    plt.plot(Cdil_grid, grid_predictions, color='#ff7f0e', linewidth=2.5, label="Optimized ODE Model Fit", zorder=2)
    
    plt.title("Competitive ODE Model Fit vs. Partitioning Data", fontsize=16, color='black')
    plt.xlabel("Dilute-Phase Concentration $[NPM1]_{dil}$", fontsize=14, color='black')
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

    figures_directory = Path(r"H:\My Drive\2025-2026 School Year\Spring26\BE175\be175project-1\Figures")
    figures_directory.mkdir(parents=True, exist_ok=True)
    
    df = load_panel_c_data(args.data, sheet_name=args.sheet)
    Cdil = df["Cdil"].to_numpy(dtype=float)
    DGtr = df["DGtr"].to_numpy(dtype=float)
    
    # Expanded boundaries to lift the -1.5 ceiling while safely bounded for stability
    bnn_bounds = (
        (0.0, 5.0),                       # Competitor influx 'b'
        (1e-6, np.max(Cdil) * 10),        # Scale factor 'C_scale'
        (-30.0, 30.0),                    # Lowered minimum baseline to allow wider sweep
        (-30.0, 30.0),                    # Expanded range parameter
        (0.01, 10.0)                      # Flexible power-law exponent
    )
    
    cv_predictions = run_stratified_gradient_cv(Cdil, DGtr, bnn_bounds, n_folds=5)
    generate_and_save_cv_metrics(DGtr, cv_predictions, figures_directory)
    plot_continuous_model_fit(Cdil, DGtr, cv_predictions, bnn_bounds, figures_directory)
    
    print(f"\nPipeline successfully aligned! Both visualization plots saved to:\n{figures_directory}")

if __name__ == "__main__":
    main()