import pandas as pd
import numpy as np
import time


import matplotlib
matplotlib.use('Agg')  # Set before pyplot import

import optuna.visualization as vis
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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('Agg')  # Avoid tkinter dependency

# === Load dataset ===
df = pd.read_excel("camdd.xlsx")

# === Load and process holiday data ===
holiday_df = pd.read_excel("holiday_feature.xlsx")  # Replace with actual filename
holiday_df['start_date'] = pd.to_datetime(holiday_df['start_date'])
holiday_df['end_date'] = pd.to_datetime(holiday_df['end_date'])

# Generate holiday date lists
public_holidays = set()
school_holidays = set()

for _, row in holiday_df.iterrows():
    holiday_range = pd.date_range(row['start_date'], row['end_date'])
    if row['holiday_type'] == 'public_holiday':
        public_holidays.update(holiday_range)
    elif row['holiday_type'] == 'school_holiday':
        school_holidays.update(holiday_range)

#Ensure both sets are in datetime.date format (not pandas.Timestamp)
public_holidays = {d.date() for d in public_holidays}
school_holidays = {d.date() for d in school_holidays}

# === Feature extraction ===
df['DateTime'] = pd.to_datetime(df['DateTime'])
print("   Date range loaded from Excel:")
print("   Minimum DateTime:", df['DateTime'].min())
print("   Maximum DateTime:", df['DateTime'].max())

df['year'] = df['DateTime'].dt.year
df['date'] = df['DateTime'].dt.date
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['week'] = df['DateTime'].dt.isocalendar().week
df['month'] = df['DateTime'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)


# Add holiday-related features (after converting both sides to datetime.date)
df['date'] = pd.to_datetime(df['date']).dt.date  # Ensure it's date only
df['is_holiday'] = df['date'].isin(public_holidays).astype(int)
df['school_holiday'] = df['date'].isin(school_holidays).astype(int)


# Peak hour rules by camera
def assign_peak_hour(row):
    if row['day_of_week'] in [0, 1, 2, 3, 4]:  # Monday to Friday
        if row['Camera'] == 'ccc2617079d5' and row['hour'] in [7, 8]:
            return 1
        elif row['Camera'] == 'ccc2617079dd' and row['hour'] in [17, 18]:
            return 1
    return 0

df['is_peak_hour'] = df.apply(assign_peak_hour, axis=1)

df.sort_values(by=['Camera', 'DateTime'], inplace=True)


# === Aggregate hourly ===
agg_df = df.groupby(['Camera', 'Phase', 'date', 'hour', 'day_of_week', 'is_weekend', 'week','year', 'month', 'is_holiday', 'school_holiday', 'is_peak_hour']).agg({
    'Lane_0_TotalWeighted': 'sum',
    'Lane_1_TotalWeighted': 'sum',
    'Lane_2_TotalWeighted': 'sum',
    'Lane_3_TotalWeighted': 'sum'
}).reset_index()

agg_df['date'] = pd.to_datetime(agg_df['date'])  # Ensure it's datetime
agg_df['week_of_year'] = agg_df['date'].dt.isocalendar().week
agg_df['month'] = agg_df['date'].dt.month
print(df.columns)

#agg_df['date_encoded'] = (pd.to_datetime(agg_df['date']) - pd.to_datetime(agg_df['date'].min())).dt.days
features = ['hour', 'day_of_week', 'is_weekend', 'month', 'is_holiday', 'school_holiday', 'is_peak_hour']

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
# Only use Optuna optimization
optimization = 'Optuna'
model_timings = []

print("Rows marked as public holiday:")
print(df[df['is_holiday'] == 1][['date', 'Camera', 'is_holiday']].drop_duplicates().head())

print("\nRows marked as school holiday:")
print(df[df['school_holiday'] == 1][['date', 'Camera', 'school_holiday']].drop_duplicates().head())

# Initialize this list once at the start of your script
all_mae_rows = []

for camera_id, cam_df in agg_df.groupby('Camera'):

    print(f" Last 5 rows for camera {camera_id}:\n", cam_df[['date', 'hour', 'year', 'week']].tail())
    print(f" Max date for camera {camera_id}: {cam_df['date'].max()}")

    print(f"\nProcessing Camera: {camera_id}")
    target_lanes = camera_lane_map[camera_id]

    cam_df['year_week'] = cam_df['year'].astype(str) + '-W' + cam_df['week'].astype(str).str.zfill(2)
    
    # Ensure consistent string type
    cam_df['year_week'] = cam_df['year_week'].astype(str)
    unique_weeks = [str(w) for w in sorted(cam_df['year_week'].unique())]
    
    #Add this to debug the week-to-date mapping
    print("\n Mapping of year_week to dates:")
    print(cam_df[['year_week', 'date']].drop_duplicates().sort_values('year_week').tail(10))

    if len(unique_weeks) < 3:
        print("Not enough weeks.")
        continue

    train_weeks = unique_weeks[:-2]
    test_weeks = unique_weeks[-2:]

    print("Train weeks:", train_weeks)
    print("Test weeks:", test_weeks)

    hour_start, hour_end = 0, 23

    train_df = cam_df[cam_df['year_week'].isin(train_weeks) & cam_df['hour'].between(hour_start, hour_end)]
    test_df = cam_df[cam_df['year_week'].isin(test_weeks) & cam_df['hour'].between(hour_start, hour_end)]

    for lane_col in target_lanes:

        # === Filter out NaN values for target column ===
        train_df_filtered = train_df.dropna(subset=[lane_col]).copy()
        test_df_filtered = test_df.dropna(subset=[lane_col]).copy()

        X_train = train_df_filtered[features]
        y_train = train_df_filtered[lane_col]

        X_test = test_df_filtered[features]
        y_test = test_df_filtered[lane_col]

        print("\n Unique year_week in cam_df:")
        print(sorted(cam_df['year_week'].unique()))

        print("\n Check year_week values in train_df:")
        print(cam_df[cam_df['year_week'].isin(train_weeks)][['year_week', lane_col]].groupby('year_week').sum())

        print("\n Check year_week values in test_df:")
        print(cam_df[cam_df['year_week'].isin(test_weeks)][['year_week', lane_col]].groupby('year_week').sum())
        
        #Now it's safe to use lane_col
        print(f"Original sum {lane_col}: {cam_df[lane_col].sum()}")
        print(f"Non-zero entries for {lane_col}:\n", cam_df[cam_df[lane_col] > 0][['date', 'hour', lane_col]].head())

        print(f"{lane_col} | Train Sum: {train_df[lane_col].sum()} | Test Sum: {test_df[lane_col].sum()}")
        if train_df[lane_col].sum() == 0 and test_df[lane_col].sum() == 0:
            print(f"Skipping {lane_col} (all zeros).")
            continue

        X_train, y_train = train_df[features], train_df[lane_col]
        X_test, y_test = test_df[features], test_df[lane_col]


        test_df_sorted = cam_df[cam_df['year_week'].isin(test_weeks)].copy()
        test_df_sorted['datetime'] = pd.to_datetime(test_df_sorted['date'].astype(str) + ' ' + test_df_sorted['hour'].astype(str) + ':00')
        test_df_sorted = test_df_sorted.sort_values(by=['date', 'hour'])

        for model_name, model_class in model_classes.items():
            print(f"Tuning {model_name} | Lane: {lane_col} | Method: {optimization}")

            # === OPTUNA OPTIMIZATION ===
            def objective(trial):
                if model_name == 'Random Forest':
                    model = model_class(
                        n_estimators=trial.suggest_int('n_estimators', 100, 500),
                        max_depth=trial.suggest_int('max_depth', 5, 20),
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
                else:  # CatBoost
                    model = model_class(
                        iterations=trial.suggest_int('iterations', 100, 500),
                        depth=trial.suggest_int('depth', 3, 12),
                        learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                        verbose=0,
                        random_state=42
                    )
                score = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=ts_cv, n_jobs=-1)
                return -score.mean()

            model_start_time = time.time()

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)

            fig_loss = vis.plot_optimization_history(study)
            loss_plot_path = f"optuna_loss/{optimization}/{model_name}/{camera_id}_{lane_col}_loss.html".replace(" ", "_")
            os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
            fig_loss.write_html(loss_plot_path)
            print(f"Saved Optuna loss plot to: {loss_plot_path}")

            # === Get best model and fit it
            best_model = model_class(**study.best_params)
            if model_name == 'XGBoost':
                best_model.set_params(verbosity=0)
            elif model_name == 'CatBoost':
                best_model.set_params(verbose=0)

            best_model.fit(X_train, y_train)

            model_end_time = time.time()
            model_duration = round(model_end_time - model_start_time, 2)

            print(f"{model_name} | {camera_id} | {lane_col} — Training time: {model_duration:.2f} seconds")

            # Log timing
            model_timings.append({
                'Camera': camera_id,
                'Lane': lane_col,
                'Model': model_name,
                'Duration_Seconds': model_duration,
                'Best_Params': "; ".join([f"{k}={v}" for k, v in study.best_params.items()])
            })

            # === Calculate and print training MAE
            train_pred = best_model.predict(X_train)
            train_mae = mean_absolute_error(y_train, train_pred)
            print(f"{model_name} | {camera_id} | {lane_col} — Training MAE: {train_mae:.2f}")

            # === Prepare training dataframe for region MAE
            train_df_copy = train_df.copy()
            train_df_copy['Predicted'] = train_pred
            train_df_copy['Actual'] = y_train.values

            train_weekday_df = train_df_copy[train_df_copy['day_of_week'] < 5]
            train_weekend_df = train_df_copy[train_df_copy['day_of_week'] >= 5]


            shaded_periods = {
                '12AM-5AM': (0, 5),
                '6AM-9AM': (6, 9),
                '10AM-3PM': (10, 15),
                '4PM-6PM': (16, 18),
                '7PM-11PM': (19, 23)
            }

            # === Compute training MAE by time zone
            train_region_mae = {}

            for region_name, (start_hour, end_hour) in shaded_periods.items():
                region_rows = train_df_copy[train_df_copy['hour'].between(start_hour, end_hour)]

                if not region_rows.empty:
                    train_region_mae[region_name] = mean_absolute_error(region_rows['Actual'], region_rows['Predicted'])

            def compute_region_mae(df):
                result = {}
                for region_name, (start_hour, end_hour) in shaded_periods.items():
                    rows = df[df['hour'].between(start_hour, end_hour)]
                    if not rows.empty:
                        result[region_name] = mean_absolute_error(rows['Actual'], rows['Predicted'])
                return result

            train_weekday_mae = compute_region_mae(train_weekday_df)
            train_weekend_mae = compute_region_mae(train_weekend_df)


            # === Optional: Print result
            print("Training MAE by Time Period:")
            for r, v in train_region_mae.items():
                print(f"  {r}: {v:.2f}")

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

            # === Predict and interactive plot
            y_pred = best_model.predict(X_test)


            mae = mean_absolute_error(y_test, y_pred)
            y_mean = y_test.mean()
            perc_mae = (mae / y_mean) * 100 if y_mean > 0 else 0

            test_df_sorted = test_df.loc[X_test.index].copy()
            test_df_sorted['datetime'] = pd.to_datetime(test_df_sorted['date'].astype(str) + ' ' + test_df_sorted['hour'].astype(str) + ':00')
            test_df_sorted['datetime_str'] = test_df_sorted['date'].astype(str) + ' ' + test_df_sorted['hour'].astype(str).str.zfill(2) + ':00'
            test_df_sorted['Actual'] = y_test.values
            test_df_sorted['Predicted'] = y_pred
            test_df_sorted['MAE'] = (test_df_sorted['Actual'] - test_df_sorted['Predicted']).abs()

            # === Separate test set into weekday and weekend
            weekday_df = test_df_sorted[test_df_sorted['day_of_week'] < 5].copy()
            weekend_df = test_df_sorted[test_df_sorted['day_of_week'] >= 5].copy()

            # === Compute region-wise MAE for weekday & weekend
            def compute_region_mae(df, label):
                result = {}
                for region_name, (start_hour, end_hour) in shaded_periods.items():
                    region = df[df['hour'].between(start_hour, end_hour)]
                    if not region.empty:
                        result[region_name] = mean_absolute_error(region['Actual'], region['Predicted'])
                return result

            weekday_region_mae = compute_region_mae(weekday_df, "Weekday")
            weekend_region_mae = compute_region_mae(weekend_df, "Weekend")

            # === Collect MAE by model, time zone, and category ===
            for region in shaded_periods.keys():
                all_mae_rows.append({
                    'Camera': camera_id,
                    'Lane': lane_col,
                    'Model': model_name,
                    'TimeZone': region,
                    'Train_Weekday': train_weekday_mae.get(region, np.nan),
                    'Train_Weekend': train_weekend_mae.get(region, np.nan),
                    'Test_Weekday': weekday_region_mae.get(region, np.nan),
                    'Test_Weekend': weekend_region_mae.get(region, np.nan)
                })


            region_colors = {
                '12AM-5AM': 'rgba(76, 47, 146, 0.2)',
                '6AM-9AM': 'rgba(221, 26, 33, 0.2)',
                '10AM-3PM': 'rgba(255, 205, 3, 0.2)',
                '4PM-6PM': 'rgba(245, 125, 32, 0.2)',
                '7PM-11PM': 'rgba(0, 175, 77, 0.2)'
            }

            # === Plotly interactive figure
            fig = go.Figure()

            print("   Date range in test_df_sorted:")
            print("   Min datetime_str:", test_df_sorted['datetime_str'].min())
            print("   Max datetime_str:", test_df_sorted['datetime_str'].max())

            fig.add_trace(go.Scatter(
                x=test_df_sorted['datetime'],
                y=test_df_sorted['Actual'],
                mode='lines+markers',
                name='Actual',
                hovertemplate='Time: %{x}<br>Actual: %{y}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=test_df_sorted['datetime'],
                y=test_df_sorted['Predicted'],
                mode='lines+markers',
                name='Predicted',
                hovertemplate='Time: %{x}<br>Predicted: %{y}<br>MAE: %{customdata}<extra></extra>',
                customdata=np.round(test_df_sorted['MAE'], 2)
            ))

            test_df_sorted['datetime_str'] = test_df_sorted['date'].astype(str) + ' ' + test_df_sorted['hour'].astype(str).str.zfill(2) + ':00'
            # === Highlight Peak Hour Points with Triangle Marker ===
            peak_hour_data = test_df_sorted[test_df_sorted['is_peak_hour'] == 1]

            fig.add_trace(go.Scatter(
                x=peak_hour_data['datetime_str'],
                y=peak_hour_data['Predicted'],  # You can also use 'Actual' if preferred
                mode='markers',
                marker=dict(color='orange', size=15, symbol='triangle-up'),
                name='Peak Hour'
            ))

            # === Find and highlight point with highest MAE
            max_mae_idx = test_df_sorted['MAE'].idxmax()
            highlight_time = test_df_sorted.loc[max_mae_idx, 'datetime_str']
            highlight_actual = test_df_sorted.loc[max_mae_idx, 'Actual']
            highlight_pred = test_df_sorted.loc[max_mae_idx, 'Predicted']
            highlight_mae = test_df_sorted.loc[max_mae_idx, 'MAE']

            # === Add scatter marker and annotation
            fig.add_trace(go.Scatter(
                x=[highlight_time],
                y=[highlight_actual],
                mode='markers+text',
                marker=dict(color='red', size=12, symbol='x'),
                text=[f'Max MAE: {highlight_mae:.2f}'],
                textposition='top center',
                name='Max MAE Point'
            ))  

            # === Shade region and calculate MAE per region
            region_mae = {}

            for region_name, (start_hour, end_hour) in shaded_periods.items():
                region_rows = test_df_sorted[test_df_sorted['hour'].between(start_hour, end_hour)]

                if not region_rows.empty:
                    # Compute MAE
                    region_mae[region_name] = mean_absolute_error(region_rows['Actual'], region_rows['Predicted'])

                    # Add shading for each timepoint in region
                    for dt in region_rows['datetime_str']:
                        x_dt = pd.to_datetime(dt)
                        fig.add_vrect(
                            x0=x_dt - pd.Timedelta(minutes=30),
                            x1=x_dt + pd.Timedelta(minutes=30),
                            fillcolor=region_colors[region_name],
                            opacity=0.75,
                            layer='below',
                            line_width=0
                        )

            # === Annotate Weekday & Weekend MAE per region side-by-side
            for i, region_name in enumerate(shaded_periods.keys()):
                # Weekday MAE annotation
                if region_name in weekday_region_mae:
                    fig.add_annotation(
                        text=f"{region_name} Test WD MAE: {weekday_region_mae[region_name]:.2f}",
                        xref="paper", yref="paper",
                        x=(i + 0.6) / len(shaded_periods),
                        y=1.25,
                        showarrow=False,
                        font=dict(size=10),
                        align="center",
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
                # Weekend MAE annotation
                if region_name in weekend_region_mae:
                    fig.add_annotation(
                        text=f"{region_name} Test WE MAE: {weekend_region_mae[region_name]:.2f}",
                        xref="paper", yref="paper",
                        x=(i + 0.6) / len(shaded_periods),
                        y=1.19,
                        showarrow=False,
                        font=dict(size=10),
                        align="center",
                        bgcolor="#f5f5f5",
                        bordercolor="gray",
                        borderwidth=1
                    )

            for i, (region_name, mae_val) in enumerate(train_weekday_mae.items()):
                fig.add_annotation(
                    text=f"{region_name} Train WD MAE: {mae_val:.2f}",
                    xref="paper", yref="paper",
                    x=(i + 0.6) / len(shaded_periods),
                    y=1.13,
                    showarrow=False,
                    font=dict(size=10, color="green"),
                    align="center",
                    bgcolor="white",
                    bordercolor="green",
                    borderwidth=1
                )

            for i, (region_name, mae_val) in enumerate(train_weekend_mae.items()):
                fig.add_annotation(
                    text=f"{region_name} Train WE MAE: {mae_val:.2f}",
                    xref="paper", yref="paper",
                    x=(i + 0.6) / len(shaded_periods),
                    y=1.07,
                    showarrow=False,
                    font=dict(size=10, color="purple"),
                    align="center",
                    bgcolor="white",
                    bordercolor="purple",
                    borderwidth=1
                )


            # === Add legend entries for shaded time zones
            for region_name, color in region_colors.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=12, color=color),
                    name=region_name  # This will appear in legend
                ))

            fig.update_layout(
                title=(
                    f"{optimization} | {model_name} | {camera_id} | {lane_col}<br>"
                    f"Test MAE: {mae:.2f} ({perc_mae:.1f}%) | Training MAE: {train_mae:.2f}"
                ),
                xaxis_title="Date + Hour",
                yaxis_title="PCU",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(
                    title="Shaded Time Zones",
                    orientation="v",
                    x=1.02,
                    y=0.8
                ),
                margin=dict(t=250)
            )

            # === Save as interactive HTML
            out_dir = os.path.join("results_html", optimization, model_name.replace(" ", "_"))
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, f"{camera_id}_{lane_col}.html".replace(" ", "_"))
            fig.write_html(filepath)
            print(f"Saved interactive plot to: {filepath}")

    # === Export modeling time logs ===
    timing_df = pd.DataFrame(model_timings)
    timing_df.to_csv("model_training_times.csv", index=False)
    print("Modeling times saved to: model_training_times.csv")

    # Convert to DataFrame
    mae_summary_df = pd.DataFrame(all_mae_rows)

    # Save for external reference
    mae_summary_df.to_csv("mae_time_zone_summary.csv", index=False)

    # Optional: print in the format you showed (by TimeZone)
    for tz in mae_summary_df['TimeZone'].unique():
        print(f"\nMAE for {tz}")
        display_df = mae_summary_df[mae_summary_df['TimeZone'] == tz].copy()
        display_df = display_df[['Model', 'Train_Weekday', 'Train_Weekend', 'Test_Weekday', 'Test_Weekend']]
        display_df = display_df.round(2)
        print(display_df.to_string(index=False))
