import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
import joblib
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
import warnings

warnings.filterwarnings("ignore")

# === Load dataset ===
df = pd.read_excel("modified_sum.xlsx")

# === Feature extraction ===
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['date'] = df['DateTime'].dt.date
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['week'] = df['DateTime'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

df.sort_values(by=['Camera', 'DateTime'], inplace=True)


# === Aggregate hourly ===
agg_df = df.groupby(['Camera', 'Phase', 'date', 'hour', 'day_of_week', 'is_weekend', 'week']).agg({
    'Lane_0_TotalWeighted': 'sum',
    'Lane_1_TotalWeighted': 'sum',
    'Lane_2_TotalWeighted': 'sum',
    'Lane_3_TotalWeighted': 'sum'
}).reset_index()

agg_df['date'] = pd.to_datetime(agg_df['date'])  # Ensure it's datetime
agg_df['week_of_year'] = agg_df['date'].dt.isocalendar().week
agg_df['month'] = agg_df['date'].dt.month

#agg_df['date_encoded'] = (pd.to_datetime(agg_df['date']) - pd.to_datetime(agg_df['date'].min())).dt.days
features = [ 'hour', 'day_of_week', 'is_weekend', 'week_of_year', 'month']

# === Assign valid lanes per camera ===
camera_lane_map = {
    'ccc2617079d5': ['Lane_2_TotalWeighted', 'Lane_3_TotalWeighted'],
    'ccc2617079dd': ['Lane_0_TotalWeighted', 'Lane_1_TotalWeighted']
}

model_classes = {
    'Random Forest': RandomForestRegressor,
    'XGBoost': XGBRegressor,
    'LightGBM': LGBMRegressor,
    'CatBoost': CatBoostRegressor
}

ts_cv = TimeSeriesSplit(n_splits=3)
optimization_modes = ['Optuna', 'Bayesian']

for optimization in optimization_modes:
    print(f"\n Using {optimization} optimization method")

    for camera_id, cam_df in agg_df.groupby('Camera'):
        print(f"\nProcessing Camera: {camera_id}")
        target_lanes = camera_lane_map[camera_id]
        unique_weeks = sorted(cam_df['week'].unique())

        if len(unique_weeks) < 2:
            print("Not enough weeks.")
            continue

        train_weeks = unique_weeks[:-1]
        test_week = unique_weeks[-1]

        train_df = cam_df[cam_df['week'].isin(train_weeks)]
        test_df = cam_df[cam_df['week'] == test_week]

        for lane_col in target_lanes:
            if train_df[lane_col].sum() == 0 and test_df[lane_col].sum() == 0:
                print(f"Skipping {lane_col} (all zeros).")
                continue

            X_train, y_train = train_df[features], train_df[lane_col]
            X_test, y_test = test_df[features], test_df[lane_col]

            test_df_sorted = test_df.copy()
            test_df_sorted['datetime_str'] = test_df_sorted['date'].astype(str) + ' ' + test_df_sorted['hour'].astype(str).str.zfill(2) + ':00'
            test_df_sorted = test_df_sorted.sort_values(by=['date', 'hour'])

            for model_name, model_class in model_classes.items():
                print(f"Tuning {model_name} | Lane: {lane_col} | Method: {optimization}")

                # === OPTUNA OPTIMIZATION
                if optimization == 'Optuna':
                    def objective(trial):
                        if model_name == 'Random Forest':
                            model = model_class(
                                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                                max_depth=trial.suggest_int('max_depth', 5, 50),
                                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                                random_state=42
                            )
                        elif model_name == 'XGBoost':
                            model = model_class(
                                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                                max_depth=trial.suggest_int('max_depth', 3, 20),
                                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                                random_state=42, verbosity=0
                            )
                        elif model_name == 'LightGBM':
                            model = model_class(
                                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                                max_depth=trial.suggest_int('max_depth', 3, 20),
                                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                                random_state=42
                            )
                        else:
                            model = model_class(
                                iterations=trial.suggest_int('iterations', 100, 500),
                                depth=trial.suggest_int('depth', 3, 12),
                                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                                verbose=0,
                                random_state=42
                            )
                        score = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=ts_cv, n_jobs=-1)
                        return -score.mean()

                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=20, show_progress_bar=False)

                   # === Plot MAE loss curve for Optuna ===
                    loss_df = study.trials_dataframe()
                    loss_plot_path = f"loss_plots/{optimization}/{model_name}/{camera_id}_{lane_col}_loss.png".replace(" ", "_")
                    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)

                    plt.figure(figsize=(10, 4))
                    plt.plot(loss_df['value'], marker='o')
                    plt.title(f'{optimization} | {model_name} | {camera_id} | {lane_col}\\nMAE over Trials')
                    plt.xlabel('Trial')
                    plt.ylabel('MAE')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(loss_plot_path)
                    plt.close()

                    best_model = model_class(**study.best_params)
                    if model_name == 'XGBoost':
                        best_model.set_params(verbosity=0)
                    elif model_name == 'CatBoost':
                        best_model.set_params(verbose=0)

                # === BAYESIAN OPTIMIZATION ===
                else:
                    # Initialize model with basic params
                    if model_name == 'Random Forest':
                        param_grid = {
                            'n_estimators': (100, 500),
                            'max_depth': (5, 50),
                            'min_samples_split': (2, 10),
                            'min_samples_leaf': (1, 5)
                        }
                        model = model_class(random_state=42)
                    elif model_name == 'XGBoost':
                        param_grid = {
                            'n_estimators': (100, 500),
                            'max_depth': (3, 20),
                            'learning_rate': (0.001, 0.3, 'log-uniform')
                        }
                        model = model_class(random_state=42, verbosity=0)
                    elif model_name == 'LightGBM':
                        param_grid = {
                            'n_estimators': (100, 500),
                            'max_depth': (3, 20),
                            'learning_rate': (0.001, 0.3, 'log-uniform')
                        }
                        model = model_class(random_state=42)
                    else:  # CatBoost
                        param_grid = {
                            'iterations': (100, 500),
                            'depth': (3, 12),
                            'learning_rate': (0.001, 0.3, 'log-uniform')
                        }
                        model = model_class(random_state=42, verbose=0)

                    # Run Bayesian optimization
                    bayes = BayesSearchCV(
                        estimator=model,
                        search_spaces=param_grid,
                        n_iter=20,
                        scoring='neg_mean_absolute_error',
                        cv=ts_cv,
                        n_jobs=-1,
                        random_state=42
                    )
                    
                    try:
                        bayes.fit(X_train, y_train)
                        
                        # === Plot MAE loss curve for Bayesian ===
                        if hasattr(bayes, 'cv_results_'):
                            losses = -np.array(bayes.cv_results_['mean_test_score'])
                            loss_plot_path = f"loss_plots/{optimization}/{model_name}/{camera_id}_{lane_col}_loss.png".replace(" ", "_")
                            os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)

                            plt.figure(figsize=(10, 4))
                            plt.plot(losses, marker='o')
                            plt.title(f'{optimization} | {model_name} | {camera_id} | {lane_col}\nRaw MAE over Iterations')
                            plt.xlabel('Iteration')
                            plt.ylabel('MAE')
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(loss_plot_path)
                            plt.close()
                        else:
                            print(f"Could not plot loss for {model_name} | {camera_id} | {lane_col} — cv_results_ not available.")

                        # === Get best model ===
                        if hasattr(bayes, 'best_estimator_'):
                            best_model = bayes.best_estimator_
                        else:
                            print("Bayesian optimization failed to produce a best estimator. Using default parameters.")
                            best_model = model
                    except Exception as e:
                        print(f"Bayesian optimization failed: {str(e)}")
                        print("Using default model parameters.")
                        best_model = model

                # === Fit best model
                best_model.fit(X_train, y_train)

                # === Plot and save feature importance
                if hasattr(best_model, "feature_importances_"):
                    importance = best_model.feature_importances_
                    feature_plot_path = f"feature_importance/{optimization}/{model_name}/{camera_id}_{lane_col}_feature_importance.png".replace(" ", "_")
                    os.makedirs(os.path.dirname(feature_plot_path), exist_ok=True)

                    plt.figure(figsize=(8, 5))
                    plt.barh(features, importance)
                    plt.xlabel("Feature Importance")
                    plt.title(f"{model_name} | {camera_id} | {lane_col}")
                    plt.tight_layout()
                    plt.savefig(feature_plot_path)
                    plt.close()
                    print(f"Saved feature importance to: {feature_plot_path}")

                # === Save model
                model_dir = os.path.join("saved_models", optimization, model_name.replace(" ", "_"))
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{camera_id}_{lane_col}.pkl")
                joblib.dump(best_model, model_path)
                print(f"Saved model to: {model_path}")

                # === Predict and plot
                y_pred = best_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                y_mean = y_test.mean()
                perc_mae = (mae / y_mean) * 100 if y_mean > 0 else 0

                out_dir = os.path.join("results", optimization, model_name.replace(" ", "_"))
                os.makedirs(out_dir, exist_ok=True)

                plt.figure(figsize=(14, 5))
                plt.plot(test_df_sorted['datetime_str'], y_test, label='Actual', marker='o')
                plt.plot(test_df_sorted['datetime_str'], y_pred, label='Predicted', linestyle='--')
                plt.title(f"{optimization} | {model_name} | {camera_id} | {lane_col}\nMAE: {mae:.2f} ({perc_mae:.1f}%)")
                plt.xticks(rotation=45)
                plt.xlabel('Date + Hour')
                plt.ylabel('PCU')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                filename = f"{camera_id}_{lane_col}.png".replace(" ", "_")
                filepath = os.path.join(out_dir, filename)
                plt.savefig(filepath)
                plt.close()
                print(f"Saved plot to: {filepath}")
