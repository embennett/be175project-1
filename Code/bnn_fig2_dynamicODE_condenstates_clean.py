"""
bnn_fig2_fig2c_cv_iteration.py

This script applies the sequestration/shared-resource model structure from the
Samaniego biomolecular neural network paper to Figure 2C source data from
Riback et al., "Composition dependent thermodynamics of intracellular phase
separation."

Original model idea:
    molecular input concentration -> nonlinear transcriptional output

Condensate extension:
    dilute-phase NPM1 concentration -> transfer free energy of condensate partitioning

For Riback Figure 2C:
    Cdil = NPM1 concentration in the dilute phase outside the nucleolus
    DGtr = transfer free energy for moving NPM1 into the nucleolus

The goal is not to claim that nucleoli literally use sigma factors or
anti-sigma factors. The goal is to test whether the same type of nonlinear
molecular allocation model can describe condensate partitioning behavior.

Run from the repository root with:

python3 Code/bnn_fig2_fig2c_cv_iteration.py \
    --data Data/41586_2020_2256_MOESM5_ESM.xlsx \
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
    """Tracks competitive sequestration and resource allocation."""
    S1, A1, S2, A2, C1, C2 = x
    # Prevent negative capacity to avoid imaginary numbers with fractional exponents
    C = max(0.0, ct - C1 - C2)
    
    dS1 = a1 - d*S1 - g1*A1*S1 - g2*S1*C
    dA1 = 0.0 - d*A1 - g1*A1*S1
    dS2 = b - d*S2 - g1*A2*S2 - g2*S2*C
    dA2 = 0.0 - d*A2 - g1*A2*S2
    dC1 = g2*S1*C - d*C1
    dC2 = g2*S2*C - d*C2
    
    return np.array([dS1, dA1, dS2, dA2, dC1, dC2])

def compute_ode_predictions(Cdil, params, t_max=15.0):
    """Generates steady-state predictions using a stabilized solver flow."""
    b, C_scale, DG_min, DG_range, gamma = params
    g1_fixed, g2_fixed, d_fixed, ct_fixed = 100.0, 10.0, 1.0, 0.2
    predictions = []
    
    # Using modern solve_ivp wrapper prevents LSODA step-size corrector crashes
    t_span = (0.0, t_max)
    x0 = np.zeros(6)
    
    for c_dil_val in Cdil:
        a1_scaled = c_dil_val / C_scale
        
        sol = scipy.integrate.solve_ivp(
            lambda t, x: sequestration_rhs(x, t, a1_scaled, b, g1_fixed, g2_fixed, d_fixed, ct_fixed),
            t_span, x0, method='Radau'
        )
        
        final_state = sol.y[:, -1]
        activation_fraction = np.clip(final_state[4] / ct_fixed, 0.0, 1.0)
        transformed_fraction = np.power(activation_fraction, gamma)
        
        predicted_dgtr = DG_min + DG_range * transformed_fraction
        predictions.append(predicted_dgtr)
        
    return np.array(predictions)

def global_optimization_loss(params, Cdil_train, DGtr_train):
    """Standard residual sum of squares calculation for clean training optimization."""
    predictions = compute_ode_predictions(Cdil_train, params)
    return np.sum((DGtr_train - predictions) ** 2)

def run_stratified_gradient_cv(Cdil, DGtr, bounds, n_folds=5):
    """Distributes points into 5 stratified folds along the concentration gradient."""
    num_points = len(Cdil)
    cv_predictions = np.zeros(num_points)
    
    initial_guess = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr), 1.0]
    fold_assignments = np.array([i % n_folds for i in range(num_points)])
    
    print(f"Starting {n_folds}-Fold CV")
    for fold_idx in range(n_folds):
        print(f"  Fold {fold_idx + 1}/{n_folds}... ", end="", flush=True)
        
        val_indices = np.where(fold_assignments == fold_idx)[0]
        train_indices = np.where(fold_assignments != fold_idx)[0]
        
        Cdil_train, DGtr_train = Cdil[train_indices], DGtr[train_indices]
        Cdil_val = Cdil[val_indices]
        
        opt_res = scipy.optimize.minimize(
            global_optimization_loss, initial_guess,
            args=(Cdil_train, DGtr_train),
            bounds=bounds, method='L-BFGS-B'
        )
        
        predicted_vals = compute_ode_predictions(Cdil_val, opt_res.x)
        cv_predictions[val_indices] = predicted_vals
        print("Done.")
        
    return cv_predictions

def load_panel_c_data(excel_path, sheet_name="Panel c"):
    """Loads panel data while dynamically extracting cell construct tags from text."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    renamed_columns = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "cdil" in col_lower:
            renamed_columns[col] = "Cdil"
        elif "dgtr" in col_lower or "delta" in col_lower or "δg" in col_lower:
            renamed_columns[col] = "DGtr"
        elif col_lower in ["tag", "protein", "construct", "fluorophore", "sample", "name"]:
            renamed_columns[col] = "Tag"

    df = df.rename(columns=renamed_columns)
    
    # check if column headers or text fields contain fluorophore tags
    if "Tag" not in df.columns:
        df["Tag"] = "Unknown"
        for string_col in df.select_dtypes(include=['object']).columns:
            df.loc[df[string_col].astype(str).str.lower().str.contains("cherry"), "Tag"] = "mCherry"
            df.loc[df[string_col].astype(str).str.lower().str.contains("gfp"), "Tag"] = "GFP"
    else:
        df["Tag"] = df["Tag"].astype(str)
        cleaned_tags = np.where(df["Tag"].str.lower().str.contains("cherry"), "mCherry", 
                       np.where(df["Tag"].str.lower().str.contains("gfp"), "GFP", "Unknown"))
        df["Tag"] = cleaned_tags

    df["Cdil"] = pd.to_numeric(df["Cdil"], errors="coerce")
    df["DGtr"] = pd.to_numeric(df["DGtr"], errors="coerce")
    df = df.dropna(subset=["Cdil", "DGtr"]).copy()
    
    df = df.sort_values("Cdil").reset_index(drop=True)
    return df

def generate_and_save_combined_plots(df, all_cv_predictions, all_grid_traces, save_dir):
    """Generates a high-quality overlay plot showcasing both population sets."""
    plt.figure(figsize=(7.5, 5.5))
    
    colors = {
        "mCherry": {"data": "#d62728", "cv": "#ff9896", "line": "#941113"},
        "GFP":     {"data": "#2ca02c", "cv": "#98df8a", "line": "#136213"}
    }
    
    for tag in ["mCherry", "GFP"]:
        tag_mask = df["Tag"] == tag
        if not np.any(tag_mask):
            continue
            
        Cdil = df.loc[tag_mask, "Cdil"].to_numpy()
        DGtr = df.loc[tag_mask, "DGtr"].to_numpy()
        cv_pred = all_cv_predictions[tag]
        
        plt.scatter(Cdil, DGtr, color=colors[tag]["data"], edgecolor='k', 
                    s=55, alpha=0.85, label=f"{tag} Experimental Data", zorder=3)
        
        # plot out-of-fold cross-validation predictions
        plt.scatter(Cdil, cv_pred, color=colors[tag]["cv"], marker='x', 
                    s=50, linewidths=2, alpha=0.9, label=f"{tag} Out-of-Fold CV", zorder=4)
        
        grid_Cdil, grid_pred = all_grid_traces[tag]
        plt.plot(grid_Cdil, grid_pred, color=colors[tag]["line"], linewidth=2.5, 
                 label=f"{tag} Optimized ODE Trace", zorder=2)
        
    plt.title("NPM1 Condensate Partitioning Energetics", fontsize=14, fontweight='bold')
    plt.xlabel("$[NPM1]_{dil}$ ($\mu$M)", fontsize=12)
    plt.ylabel("$\Delta G_{tr}$ (kcal/mol)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=9, loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    output_filename = save_dir / "combined_model_fit_vs_experimental.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"\n[Success] Overlay fit plot saved to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Symmetrical ODE Cross-Validation Pipeline grouped by Tag.")
    parser.add_argument("--data", required=True, help="Path to the source-data Excel file.")
    parser.add_argument("--sheet", default="Panel c", help="Excel sheet tab to parse.")
    args = parser.parse_args()

    figures_directory = Path("/Users/ebennett/Library/CloudStorage/Dropbox-EMI/Evelyn Bennett/be175 proj/be175project/Figures")
    figures_directory.mkdir(parents=True, exist_ok=True)
    
    df = load_panel_c_data(args.data, sheet_name=args.sheet)

    all_cv_predictions = {}
    all_grid_traces = {}
    
    print("RUNNING INDEPENDENT STRATIFIED CV BY FLUORESCENT TAG")
    

    tags_to_process = [t for t in ["mCherry", "GFP"] if t in df["Tag"].unique()]
    if not tags_to_process:
        tags_to_process = df["Tag"].unique()

    for tag_name in tags_to_process:
        tag_df = df[df["Tag"] == tag_name].copy()
        
        Cdil = tag_df["Cdil"].to_numpy(dtype=float)
        DGtr = tag_df["DGtr"].to_numpy(dtype=float)
        
        bnn_bounds = (
            (0.0, 6.5),                       
            (1e-6, np.max(Cdil) * 10),        
            (-20.0, 20.0),                    
            (-20.0, 20.0),                    
            (0.1, 4.0)                        
        )
        
        print(f"\n>>> Processing Population: {tag_name} ({len(Cdil)} data points)")
        
        cv_predictions = run_stratified_gradient_cv(Cdil, DGtr, bnn_bounds, n_folds=5)
        all_cv_predictions[tag_name] = cv_predictions
        
        Cdil_grid = np.linspace(np.min(Cdil), np.max(Cdil), 300)
        initial_guess = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr), 1.0]
        
        # fixed call to point to global loss equation rather than CV loop
        opt_res = scipy.optimize.minimize(
            global_optimization_loss, initial_guess,
            args=(Cdil, DGtr), bounds=bnn_bounds, method='L-BFGS-B'
        )
        
        grid_predictions = compute_ode_predictions(Cdil_grid, opt_res.x)
        all_grid_traces[tag_name] = (Cdil_grid, grid_predictions)
        
        # Calculate cv metrics using Q2Y formulation
        ss_res = np.sum((DGtr - cv_predictions) ** 2)
        ss_tot = np.sum((DGtr - np.mean(DGtr)) ** 2)
        q2y_score = 1.0 - (ss_res / ss_tot)
        print(f"-> {tag_name} Isolated Cross-Validation Q²_Y Score: {q2y_score:.3f}")

    generate_and_save_combined_plots(df, all_cv_predictions, all_grid_traces, figures_directory)
    print("PIPELINE COMPLETE: ALL TARGET DATA OVERLAYED SUCCESSFULLY")

if __name__ == "__main__":
    main()