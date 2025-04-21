import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

def load_and_prepare_data(filepath):
    """Load and preprocess data from Excel"""
    df = pd.read_excel(filepath)

    # Convert to datetime and extract time features
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['date'] = df['DateTime'].dt.date
    df['day_of_week'] = df['DateTime'].dt.dayofweek

    # Create lane-specific DataFrames
    lane_data = []
    for lane in range(4):
        lane_df = df.copy()
        lane_df['Lane'] = f'Lane{lane}'
        lane_df['Queue'] = df[f'Lane_{lane}_Queue']
        lane_df['PCU'] = df[f'Lane_{lane}_TotalWeighted']
        if f'Lane{lane}_Red_Time' in df.columns:
            lane_df.rename(columns={f'Lane{lane}_Red_Time': 'RedTime'}, inplace=True)
        else:
            lane_df['RedTime'] = np.nan # Handle cases where RedTime column might be missing
        lane_data.append(lane_df[['Camera', 'DateTime', 'date', 'hour', 'day_of_week',
                                    'Lane', 'Queue', 'PCU', 'RedTime']])

    return pd.concat(lane_data)

def plot_weekly_hourly_pcu_with_ma(data, camera_id, lanes, week_start, week_end, save_folder, moving_avg_window=None):
    camera_data = data[data['Camera'] == camera_id].copy()
    weekly_data = camera_data[(camera_data['date'] >= week_start) & (camera_data['date'] <= week_end)].copy()

    if weekly_data.empty:
        print(f"No data found for Camera {camera_id} during the week of {week_start} to {week_end}")
        return

    num_lanes = len(lanes)
    fig, axes = plt.subplots(num_lanes, 1, figsize=(15, 5 * num_lanes), sharex=True)
    if num_lanes == 1:
        axes = [axes]  # Make sure axes is always a list

    for i, lane in enumerate(lanes):
        lane_data = weekly_data[weekly_data['Lane'] == lane].copy()
        if lane_data.empty:
            print(f"No data found for Camera {camera_id}, {lane} during the week.")
            continue

        # Aggregate PCU per hour (sum of phases)
        hourly_data = lane_data.groupby(['date', 'hour']).agg({'PCU': 'sum'}).reset_index()
        hourly_data.rename(columns={'PCU': 'Total_PCU'}, inplace=True) # Rename for clarity
        hourly_data['plot_time'] = hourly_data.apply(
            lambda x: datetime.combine(x['date'], datetime.min.time()).replace(hour=x['hour']),
            axis=1
        )
        hourly_data = hourly_data.sort_values(by='plot_time')

        ax = axes[i]
        ax.plot(hourly_data['plot_time'], hourly_data['Total_PCU'], marker='o', linestyle='-', markersize=3, label=f'Total PCU - {lane}')

        if moving_avg_window is not None and moving_avg_window > 1:
            hourly_data['MovingAvg'] = hourly_data['Total_PCU'].rolling(window=moving_avg_window, min_periods=1).mean()
            ax.plot(hourly_data['plot_time'], hourly_data['MovingAvg'], color='red', linestyle='--', label=f'Moving Avg ({moving_avg_window} hours)')

        ax.set_title(f'Camera {camera_id} - {lane}\nWeek of {week_start} to {week_end}')
        ax.set_ylabel('Total Hourly PCU')
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.legend()

    plt.xlabel('Date and Hour')
    plt.tight_layout()

    filename = f"camera_{camera_id.replace('CAM_', '')}_week_{week_start.strftime('%Y-%m-%d')}_to_{week_end.strftime('%Y-%m-%d')}"
    if moving_avg_window is not None and moving_avg_window > 1:
        filename += f"_ma{moving_avg_window}"
    filename += ".png"
    filepath = os.path.join(save_folder, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved plot to: {filepath}")

if __name__ == "__main__":
    filepath = "modified_sum.xlsx"
    print(f"Loading data from {filepath}...")
    df = load_and_prepare_data(filepath)

    start_date_overall = pd.to_datetime('2025-01-06').date()
    end_date_overall = pd.to_datetime('2025-03-30').date()

    unique_cameras = df['Camera'].unique()
    unique_cameras.sort()

    save_folder = "weekly_pcu_plots_with_ma"
    os.makedirs(save_folder, exist_ok=True)

    moving_average_window = 24  # Window size for the plotting moving average (e.g., 24 hours)
    print(f"\nApplying moving average for plotting with a window of {moving_average_window} hours.")

    for camera in unique_cameras:
        print(f"\n--- Plotting weekly data for Camera: {camera} ---")
        current_week_start = start_date_overall
        week_number = 1

        while current_week_start <= end_date_overall:
            current_week_end = current_week_start + timedelta(days=(6 - current_week_start.weekday()))
            if current_week_end > end_date_overall:
                current_week_end = end_date_overall

            lanes_to_plot = [f'Lane{i}' for i in range(4)]

            print(f"\nPlotting Week {week_number} ({current_week_start} to {current_week_end})")
            plot_weekly_hourly_pcu_with_ma(df, camera, lanes_to_plot, current_week_start, current_week_end, save_folder, moving_average_window)

            current_week_start = current_week_end + timedelta(days=1)
            week_number += 1

    print(f"\nAll weekly PCU plots with original data and moving average applied, saved to the folder: {save_folder}")