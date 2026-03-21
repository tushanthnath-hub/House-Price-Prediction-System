# ============================================================
# House Price Prediction System
# Tools: Python, Scikit-learn, Pandas, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib


# ─────────────────────────────────────────────
# 1. Generate Synthetic Housing Dataset
# ─────────────────────────────────────────────

def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    area_sqft      = rng.integers(500, 5000, n_samples)
    bedrooms       = rng.integers(1, 6, n_samples)
    bathrooms      = rng.integers(1, 4, n_samples)
    floors         = rng.integers(1, 4, n_samples)
    garage_size    = rng.integers(0, 4, n_samples)
    age_years      = rng.integers(0, 60, n_samples)
    lot_size       = rng.integers(1000, 20000, n_samples)
    basement_sqft  = rng.integers(0, 1500, n_samples)
    pool           = rng.integers(0, 2, n_samples)
    renovated      = rng.integers(0, 2, n_samples)
    school_rating  = rng.integers(1, 11, n_samples)
    crime_index    = rng.uniform(1, 10, n_samples).round(2)
    dist_city_km   = rng.uniform(1, 50, n_samples).round(2)

    neighborhood   = rng.choice(["Urban", "Suburban", "Rural"], n_samples, p=[0.40, 0.40, 0.20])
    condition      = rng.choice(["Excellent", "Good", "Fair", "Poor"], n_samples,
                                 p=[0.20, 0.45, 0.25, 0.10])
    style          = rng.choice(["Colonial", "Ranch", "Contemporary", "Craftsman", "Victorian"],
                                 n_samples)

    # Price formula (realistic dependencies)
    price = (
        80 * area_sqft
        + 15_000 * bedrooms
        + 12_000 * bathrooms
        + 8_000  * floors
        + 10_000 * garage_size
        - 1_500  * age_years
        + (10_000 if renovated.astype(bool) else 0) * renovated
        + 25_000 * pool
        + 2_000  * school_rating
        - 3_000  * crime_index
        - 500    * dist_city_km
        + 5      * lot_size
        + 50     * basement_sqft
        + rng.normal(0, 15_000, n_samples)
    )

    # Neighborhood multiplier
    nb_mult = np.where(neighborhood == "Urban", 1.25,
               np.where(neighborhood == "Suburban", 1.0, 0.75))
    price *= nb_mult

    # Condition multiplier
    cond_mult = {"Excellent": 1.20, "Good": 1.05, "Fair": 0.90, "Poor": 0.75}
    price *= np.array([cond_mult[c] for c in condition])
    price = price.clip(50_000, 2_000_000).round(2)

    df = pd.DataFrame({
        "AreaSqFt":     area_sqft,
        "Bedrooms":     bedrooms,
        "Bathrooms":    bathrooms,
        "Floors":       floors,
        "GarageSize":   garage_size,
        "AgeYears":     age_years,
        "LotSize":      lot_size,
        "BasementSqFt": basement_sqft,
        "Pool":         pool,
        "Renovated":    renovated,
        "SchoolRating": school_rating,
        "CrimeIndex":   crime_index,
        "DistCityKm":   dist_city_km,
        "Neighborhood": neighborhood,
        "Condition":    condition,
        "Style":        style,
        "Price":        price,
    })
    return df


# ─────────────────────────────────────────────
# 2. Data Preprocessing & Feature Scaling
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Ordinal encoding for 'Condition'
    condition_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
    df["Condition"] = df["Condition"].map(condition_map)

    # Label encode Neighborhood
    le = LabelEncoder()
    df["Neighborhood"] = le.fit_transform(df["Neighborhood"])

    # One-hot encode Style
    df = pd.get_dummies(df, columns=["Style"], drop_first=True)

    X = df.drop(columns=["Price"])
    y = df["Price"]

    num_cols = [
        "AreaSqFt", "LotSize", "BasementSqFt", "AgeYears",
        "CrimeIndex", "DistCityKm", "SchoolRating"
    ]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, scaler


# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PricePerSqFt"]     = 0  # placeholder — set after model training
    df["TotalRooms"]       = df["Bedrooms"] + df["Bathrooms"]
    df["LivingSpaceRatio"] = df["AreaSqFt"] / (df["LotSize"] + 1)
    df["AgeSinceRenovate"] = df["AgeYears"] * (1 - df["Renovated"])
    df["LuxuryScore"]      = (
        df["Pool"] * 3
        + df["GarageSize"]
        + (df["SchoolRating"] > 7).astype(int) * 2
        + df["Renovated"]
    )
    return df


# ─────────────────────────────────────────────
# 4. Model Training & Evaluation
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    print(f"  {name:<30} | R²={r2:.4f} | MAE=${mae:>10,.0f} | RMSE=${rmse:>10,.0f} | CV R²={cv.mean():.4f}±{cv.std():.4f}")

    return {
        "model": model, "y_pred": y_pred,
        "mae": mae, "rmse": rmse, "r2": r2,
        "cv_mean": cv.mean(), "cv_std": cv.std(),
    }


def train_all_models(X_train, X_test, y_train, y_test) -> dict:
    models = {
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":         Ridge(alpha=1.0),
        "Lasso Regression":         Lasso(alpha=100.0, max_iter=5000),
        "Decision Tree":            DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    print(f"\n{'Model':<30} | {'R²':>6} | {'MAE':>14} | {'RMSE':>14} | CV R²")
    print("─" * 90)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test)

    return results


# ─────────────────────────────────────────────
# 5. Hyperparameter Tuning
# ─────────────────────────────────────────────

def tune_decision_tree(X_train, y_train) -> DecisionTreeRegressor:
    param_grid = {
        "max_depth":         [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    }
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"\n[Tuning] Best DT params : {grid.best_params_}")
    print(f"[Tuning] Best CV R²     : {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_random_forest(X_train, y_train) -> RandomForestRegressor:
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [None, 10, 20],
        "max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"\n[Tuning] Best RF params : {grid.best_params_}")
    print(f"[Tuning] Best CV R²     : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────

def plot_all(df_raw, results, y_test, X_test, feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # ── 6a. Price Distribution ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_raw["Price"] / 1e3, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("House Price Distribution", fontweight="bold")
    axes[0].set_xlabel("Price ($K)"); axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(df_raw["Price"]), bins=50, color="seagreen", edgecolor="white")
    axes[1].set_title("Log-Price Distribution", fontweight="bold")
    axes[1].set_xlabel("log(Price)"); axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_distribution.png", dpi=150)
    plt.close()

    # ── 6b. Correlation Heatmap ─────────────────────────────────
    num_df = df_raw.select_dtypes(include=np.number)
    corr   = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()

    # ── 6c. Actual vs Predicted (best 3 models) ─────────────────
    top_models = sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True)[:3]
    fig, axes  = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, res) in zip(axes, top_models):
        ax.scatter(y_test / 1e3, res["y_pred"] / 1e3,
                   alpha=0.3, s=10, color="steelblue")
        lims = [min(y_test.min(), res["y_pred"].min()) / 1e3,
                max(y_test.max(), res["y_pred"].max()) / 1e3]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_title(f"{name}\nR²={res['r2']:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual ($K)"); ax.set_ylabel("Predicted ($K)")
    plt.suptitle("Actual vs Predicted Prices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=150)
    plt.close()

    # ── 6d. Residual Plot ───────────────────────────────────────
    best_name, best_res = top_models[0]
    residuals = y_test - best_res["y_pred"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(best_res["y_pred"] / 1e3, residuals / 1e3,
                alpha=0.3, s=10, color="tomato")
    ax1.axhline(0, color="black", lw=1.5, ls="--")
    ax1.set_xlabel("Predicted ($K)"); ax1.set_ylabel("Residual ($K)")
    ax1.set_title(f"Residuals – {best_name}", fontweight="bold")

    ax2.hist(residuals / 1e3, bins=50, color="tomato", edgecolor="white")
    ax2.set_xlabel("Residual ($K)"); ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_plot.png", dpi=150)
    plt.close()

    # ── 6e. Model Comparison Bar Chart ─────────────────────────
    model_names = list(results.keys())
    r2_scores   = [results[m]["r2"] for m in model_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["tomato" if r < 0.80 else "steelblue" for r in r2_scores]
    bars    = ax.barh(model_names, r2_scores, color=colors, edgecolor="white")
    ax.axvline(0.82, color="green", ls="--", lw=1.5, label="Target R²=0.82")
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_width() - 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", ha="right", color="white", fontsize=9)
    ax.set_title("Model Comparison – R² Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("R² Score"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150)
    plt.close()

    # ── 6f. Feature Importance (Random Forest) ──────────────────
    rf_res = results.get("Random Forest")
    if rf_res:
        imp_df = (
            pd.DataFrame({"Feature": feature_names,
                          "Importance": rf_res["model"].feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax, palette="Blues_r")
        ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
        plt.close()

    # ── 6g. Price vs Area scatter ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df_raw["AreaSqFt"], df_raw["Price"] / 1e3,
                         c=df_raw["SchoolRating"], cmap="RdYlGn",
                         alpha=0.4, s=10)
    plt.colorbar(scatter, ax=ax, label="School Rating")
    ax.set_xlabel("Area (sq ft)"); ax.set_ylabel("Price ($K)")
    ax.set_title("Price vs Area (colored by School Rating)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_vs_area.png", dpi=150)
    plt.close()

    print(f"\n[Plots] Saved to '{output_dir}/'")


# ─────────────────────────────────────────────
# 7. Prediction Interface
# ─────────────────────────────────────────────

def predict_price(model, scaler, feature_names, sample: dict) -> float:
    """Predict price for a single house given a feature dict."""
    df_sample = pd.DataFrame([sample])
    # Encode
    if "Condition" in df_sample:
        condition_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
        df_sample["Condition"] = df_sample["Condition"].map(condition_map)
    le = LabelEncoder()
    if "Neighborhood" in df_sample:
        df_sample["Neighborhood"] = le.fit_transform(df_sample["Neighborhood"])
    # Align columns
    df_sample = df_sample.reindex(columns=feature_names, fill_value=0)
    num_cols = [
        "AreaSqFt", "LotSize", "BasementSqFt", "AgeYears",
        "CrimeIndex", "DistCityKm", "SchoolRating"
    ]
    existing_num = [c for c in num_cols if c in df_sample.columns]
    df_sample[existing_num] = scaler.transform(df_sample[existing_num])
    prediction = model.predict(df_sample)[0]
    return round(prediction, 2)


# ─────────────────────────────────────────────
# 8. Main Pipeline
# ─────────────────────────────────────────────

def main():
    OUTPUT_DIR = "outputs/house_price"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/7] Generating housing dataset …")
    df_raw = generate_dataset(n_samples=5000)
    print(f"      Shape: {df_raw.shape} | Avg Price: ${df_raw['Price'].mean():,.0f}")

    print("\n[2/7] Feature engineering …")
    df_fe = engineer_features(df_raw)

    print("\n[3/7] Preprocessing …")
    X, y, scaler = preprocess(df_fe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    print("\n[4/7] Training & evaluating models …")
    results = train_all_models(X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_res  = results[best_name]
    print(f"\n  Best Model → {best_name}  (R²={best_res['r2']:.4f})")

    print("\n[5/7] Hyperparameter tuning …")
    best_dt = tune_decision_tree(X_train, y_train)
    best_rf = tune_random_forest(X_train, y_train)

    print("\n[6/7] Generating visualisations …")
    plot_all(df_raw, results, y_test, X_test, X.columns.tolist(), OUTPUT_DIR)

    # Save best model
    joblib.dump(best_rf, f"{OUTPUT_DIR}/best_house_price_model.pkl")
    joblib.dump(scaler,  f"{OUTPUT_DIR}/scaler.pkl")
    print(f"[Model] Saved to '{OUTPUT_DIR}/'")

    print("\n[7/7] Sample prediction …")
    sample = {
        "AreaSqFt": 2000, "Bedrooms": 3, "Bathrooms": 2, "Floors": 2,
        "GarageSize": 2, "AgeYears": 10, "LotSize": 6000, "BasementSqFt": 500,
        "Pool": 0, "Renovated": 1, "SchoolRating": 8, "CrimeIndex": 3.5,
        "DistCityKm": 15.0, "Neighborhood": "Suburban", "Condition": "Good",
        "TotalRooms": 5, "LivingSpaceRatio": 0.33, "AgeSinceRenovate": 0,
        "LuxuryScore": 3,
    }
    pred = predict_price(best_rf, scaler, X.columns.tolist(), sample)
    print(f"      Predicted price for sample house: ${pred:,.2f}")

    print("\n" + "="*55)
    print("  MODEL PERFORMANCE SUMMARY")
    print("="*55)
    for name, res in sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True):
        print(f"  {name:<30} R²={res['r2']:.4f}  MAE=${res['mae']:,.0f}")
    print()


if __name__ == "__main__":
    main()
