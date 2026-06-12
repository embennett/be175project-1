"""
BNN_fig2C_condensate_extension_clean.py

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

python3 Code/BNN_fig2C_condensate_extension_clean.py \
    --data Data/41586_2020_2256_MOESM5_ESM.xlsx \
    --out Results/Fig2C_condensate_extension
"""


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize


# This function mirrors the shared-resource steady-state function used in the
# original Samaniego-style code. In the original model, a sigma factor is
# sequestered by an anti-sigma factor and also competes for a limited shared
# resource such as core RNA polymerase.
def fun_SS_s1(a, b, g1, g2, d, ct):
    # The original steady-state system reduces to a cubic equation.
    # Solving the cubic gives the free input species remaining after
    # sequestration and shared-resource competition.
    A = -a / d + b / d + ct + d / g1 + d / g2
    B = -a / g1 - a / g2 + b / g2 + d * ct / g1 + d**2 / g1 / g2
    C = -d * a / g1 / g2

    roots = np.roots(np.array([1, A, B, C]))

    # Keep the root that is biologically meaningful: real and non-negative.
    real_positive_roots = roots[
        (np.isclose(roots.imag, 0, atol=1e-8)) & (roots.real >= 0)
    ].real

    if len(real_positive_roots) == 0:
        return np.nan

    return np.min(real_positive_roots)


# This function converts the Riback Figure 2C concentration input into a
# dimensionless activation-like output from the Samaniego model.
def bnn_activation_from_cdil(Cdil, b, C_scale, g1=100.0, g2=10.0, d=1.0, ct=1 / 5):
    # Cdil is measured in the paper's concentration units. The BNN equations are
    # nondimensional, so C_scale maps the experimental concentration into the
    # model input variable a.
    Cdil = np.asarray(Cdil, dtype=float)
    a_values = Cdil / C_scale

    free_species = np.empty_like(a_values, dtype=float)
    for i, a in enumerate(a_values):
        free_species[i] = fun_SS_s1(a, b, g1, g2, d, ct)

    # In the original model, this expression represents fractional activation
    # through binding to a shared resource. Here, we reinterpret it as a generic
    # nonlinear allocation term that can influence condensate partitioning.
    activation = free_species / (free_species + d / g2)
    return activation


# This is the main condensate-extension model. It uses the BNN-style nonlinear
# activation term to predict transfer free energy, DGtr.
def dgtr_model(Cdil, b, C_scale, DG_min, DG_range):
    activation = bnn_activation_from_cdil(Cdil, b, C_scale)
    return DG_min + DG_range * activation


# A constant model asks whether Cdil explains anything at all. If this performs
# poorly, it means the concentration dependence matters.
def constant_model(Cdil, c0):
    Cdil = np.asarray(Cdil, dtype=float)
    return np.full_like(Cdil, c0, dtype=float)


# A linear model asks whether a straight-line relationship is enough to describe
# the data.
def linear_model(Cdil, m, c):
    Cdil = np.asarray(Cdil, dtype=float)
    return m * Cdil + c


# A Hill/saturation model is a simple nonlinear baseline. It is not the main BNN
# model, but it is useful for checking whether the BNN-inspired model performs
# similarly to a standard saturating biological response model.
def hill_saturation_model(Cdil, DG_min, DG_range, K_half, n):
    Cdil = np.asarray(Cdil, dtype=float)
    return DG_min + DG_range * (Cdil**n / (K_half**n + Cdil**n))


# Load the Figure 2C source data from the Riback Excel supplement.
def load_panel_c_data(excel_path, sheet_name="Panel c"):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Remove blank rows and columns that may appear in supplemental Excel files.
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Rename columns to simple names so the rest of the code is easier to read.
    renamed_columns = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "cdil" in col_lower:
            renamed_columns[col] = "Cdil"
        elif "dgtr" in col_lower or "delta" in col_lower or "δg" in col_lower:
            renamed_columns[col] = "DGtr"
        elif "tag" in col_lower:
            renamed_columns[col] = "Tag"

    df = df.rename(columns=renamed_columns)

    required_columns = ["Cdil", "DGtr"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns {missing_columns}. Columns found: {list(df.columns)}"
        )

    # Keep only rows with actual numeric concentration and energy values.
    df["Cdil"] = pd.to_numeric(df["Cdil"], errors="coerce")
    df["DGtr"] = pd.to_numeric(df["DGtr"], errors="coerce")
    df = df.dropna(subset=["Cdil", "DGtr"]).copy()

    # The Tag column distinguishes fluorescent tags if present. If the file does
    # not include a Tag column, the analysis still works by labeling all points
    # as one group.
    if "Tag" not in df.columns:
        df["Tag"] = "all"
    else:
        df["Tag"] = df["Tag"].astype(str)

    df = df.sort_values("Cdil").reset_index(drop=True)
    return df


# Compute common error metrics for comparing models.
def summarize_fit(y_true, y_pred, parameters):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    output = dict(parameters)
    output.update({"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
    return output


# Fit the BNN-inspired model and simpler comparison models to the same data.
def fit_models(df):
    Cdil = df["Cdil"].to_numpy(dtype=float)
    DGtr = df["DGtr"].to_numpy(dtype=float)
    fits = {}

    c0 = np.mean(DGtr)
    pred_constant = constant_model(Cdil, c0)
    fits["constant"] = summarize_fit(DGtr, pred_constant, {"c0": c0})

    linear_params, _ = scipy.optimize.curve_fit(
        linear_model,
        Cdil,
        DGtr,
        p0=[0.01, np.min(DGtr)],
        maxfev=10000,
    )
    pred_linear = linear_model(Cdil, *linear_params)
    fits["linear"] = summarize_fit(
        DGtr,
        pred_linear,
        {"m": linear_params[0], "c": linear_params[1]},
    )

    hill_start = [np.min(DGtr), np.max(DGtr) - np.min(DGtr), np.median(Cdil), 1.0]
    hill_bounds = ([-10, -10, 1e-6, 0.1], [10, 10, np.max(Cdil) * 10, 5.0])
    hill_params, _ = scipy.optimize.curve_fit(
        hill_saturation_model,
        Cdil,
        DGtr,
        p0=hill_start,
        bounds=hill_bounds,
        maxfev=20000,
    )
    pred_hill = hill_saturation_model(Cdil, *hill_params)
    fits["hill_saturation"] = summarize_fit(
        DGtr,
        pred_hill,
        {
            "DG_min": hill_params[0],
            "DG_range": hill_params[1],
            "K_half": hill_params[2],
            "n": hill_params[3],
        },
    )

    # The BNN/shared-resource model keeps the original style of parameters from
    # the Samaniego code. We fit only the parameters needed to map the model onto
    # the Riback Fig. 2C dataset.
    bnn_start = [0.4, np.median(Cdil), np.min(DGtr), np.max(DGtr) - np.min(DGtr)]
    bnn_bounds = ([0.0, 1e-6, -10.0, -10.0], [5.0, np.max(Cdil) * 10, 10.0, 10.0])
    bnn_params, _ = scipy.optimize.curve_fit(
        dgtr_model,
        Cdil,
        DGtr,
        p0=bnn_start,
        bounds=bnn_bounds,
        maxfev=20000,
    )
    pred_bnn = dgtr_model(Cdil, *bnn_params)
    fits["bnn_shared_resource"] = summarize_fit(
        DGtr,
        pred_bnn,
        {
            "b": bnn_params[0],
            "C_scale": bnn_params[1],
            "DG_min": bnn_params[2],
            "DG_range": bnn_params[3],
            "g1_fixed": 100.0,
            "g2_fixed": 10.0,
            "d_fixed": 1.0,
            "ct_fixed": 1 / 5,
        },
    )
    fits["bnn_shared_resource"]["raw_params_for_prediction"] = bnn_params

    return fits


# Add model predictions to each experimental point.
def make_prediction_table(df, fits):
    Cdil = df["Cdil"].to_numpy(dtype=float)
    pred_df = df.copy()

    pred_df["pred_constant"] = constant_model(Cdil, fits["constant"]["c0"])
    pred_df["pred_linear"] = linear_model(Cdil, fits["linear"]["m"], fits["linear"]["c"])
    pred_df["pred_hill"] = hill_saturation_model(
        Cdil,
        fits["hill_saturation"]["DG_min"],
        fits["hill_saturation"]["DG_range"],
        fits["hill_saturation"]["K_half"],
        fits["hill_saturation"]["n"],
    )
    pred_df["pred_bnn"] = dgtr_model(
        Cdil,
        *fits["bnn_shared_resource"]["raw_params_for_prediction"],
    )
    pred_df["residual_bnn"] = pred_df["DGtr"] - pred_df["pred_bnn"]
    return pred_df


# Convert the model-fit dictionary into a table that can be saved as a CSV.
def fits_to_dataframe(fits):
    rows = []
    for model_name, fit in fits.items():
        row = {"model": model_name}
        for key, value in fit.items():
            if key != "raw_params_for_prediction":
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


# Plot the raw Riback Figure 2C data before fitting any models.
def plot_raw_data(df, out_dir):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for tag, sub in df.groupby("Tag"):
        ax.scatter(sub["Cdil"], sub["DGtr"], label=tag, alpha=0.8)

    ax.set_xlabel(r"$[NPM1]_{dil}$")
    ax.set_ylabel(r"$\Delta G_{tr}$ (kcal/mol)")
    ax.set_title("Riback Figure 2C source data")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2c_raw_data.png", dpi=300)
    plt.close(fig)


# Plot the BNN-inspired fit and comparison models over the experimental data.
def plot_model_fit(df, fits, out_dir):
    Cdil = df["Cdil"].to_numpy(dtype=float)
    Cdil_grid = np.linspace(np.min(Cdil), np.max(Cdil), 250)

    pred_bnn = dgtr_model(
        Cdil_grid,
        *fits["bnn_shared_resource"]["raw_params_for_prediction"],
    )
    pred_hill = hill_saturation_model(
        Cdil_grid,
        fits["hill_saturation"]["DG_min"],
        fits["hill_saturation"]["DG_range"],
        fits["hill_saturation"]["K_half"],
        fits["hill_saturation"]["n"],
    )
    pred_linear = linear_model(Cdil_grid, fits["linear"]["m"], fits["linear"]["c"])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for tag, sub in df.groupby("Tag"):
        ax.scatter(sub["Cdil"], sub["DGtr"], label=f"data: {tag}", alpha=0.75)

    ax.plot(Cdil_grid, pred_bnn, linewidth=2.5, label="BNN/shared-resource fit")
    ax.plot(Cdil_grid, pred_hill, linewidth=2.0, linestyle="--", label="Hill baseline")
    ax.plot(Cdil_grid, pred_linear, linewidth=1.5, linestyle=":", label="linear baseline")

    ax.set_xlabel(r"$[NPM1]_{dil}$")
    ax.set_ylabel(r"$\Delta G_{tr}$ (kcal/mol)")
    ax.set_title("Condensate partitioning modeled with BNN-style competition")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2c_model_fit.png", dpi=300)
    plt.close(fig)


# Plot model-predicted DGtr against experimental DGtr. Points close to the
# diagonal indicate better predictions.
def plot_predicted_vs_experimental(pred_df, out_dir):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(pred_df["DGtr"], pred_df["pred_bnn"], alpha=0.75)

    min_val = min(pred_df["DGtr"].min(), pred_df["pred_bnn"].min())
    max_val = max(pred_df["DGtr"].max(), pred_df["pred_bnn"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5)

    ax.set_xlabel("Experimental DGtr")
    ax.set_ylabel("Predicted DGtr")
    ax.set_title("Predicted vs experimental")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2c_predicted_vs_experimental.png", dpi=300)
    plt.close(fig)


# Plot residuals to see whether the model has concentration-dependent errors.
def plot_residuals(pred_df, out_dir):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.scatter(pred_df["Cdil"], pred_df["residual_bnn"], alpha=0.75)
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"$[NPM1]_{dil}$")
    ax.set_ylabel("Experimental - predicted DGtr")
    ax.set_title("BNN model residuals")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2c_residuals.png", dpi=300)
    plt.close(fig)


# Run the full analysis workflow.
def main():
    parser = argparse.ArgumentParser(
        description="Apply a Samaniego-style BNN model to Riback Figure 2C condensate data."
    )
    parser.add_argument("--data", required=True, help="Path to the Riback Figure 2 source-data Excel file.")
    parser.add_argument("--sheet", default="Panel c", help="Excel sheet to analyze.")
    parser.add_argument("--out", default="Results/Fig2C_condensate_extension", help="Folder for saved outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_panel_c_data(args.data, sheet_name=args.sheet)
    df.to_csv(out_dir / "cleaned_panel_c_data.csv", index=False)

    print("Loaded Riback Figure 2C data")
    print(df.head())
    print(f"Number of data points: {len(df)}")
    print(f"Tags present: {sorted(df['Tag'].unique())}")

    plot_raw_data(df, out_dir)

    fits = fit_models(df)
    fit_df = fits_to_dataframe(fits)
    fit_df.to_csv(out_dir / "fitted_parameters.csv", index=False)

    print("\nModel comparison")
    print(fit_df[["model", "MSE", "RMSE", "MAE", "R2"]])

    pred_df = make_prediction_table(df, fits)
    pred_df.to_csv(out_dir / "model_predictions.csv", index=False)

    plot_model_fit(df, fits, out_dir)
    plot_predicted_vs_experimental(pred_df, out_dir)
    plot_residuals(pred_df, out_dir)

    print("\nSaved analysis outputs to:")
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
