import torch
import xarray as xr
import numpy as np
import joblib
import os
import create_aqi_model
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Rescaler:
    """Handles scaling and inverse transformations for dataset variables."""

    def __init__(self, scaler_path):
        self.scaler = joblib.load(scaler_path)  # Load saved scaler parameters

    def rescale_xarray(self, data_array, var_dict, time_indices, og_data):
        """
        Rescale xarray Dataset using saved scaling parameters.
        
        Args:
            data_array: Input array (shape: [time, vars, lat, lon])
            var_dict: Dictionary mapping variable names to indices in data_array
            time_indices: Time coordinates for the output Dataset
            og_data: Original xarray Dataset (for coordinate reference)
        
        Returns:
            Rescaled xarray Dataset
        """
        rescaled_data = {}

        for var_name, idx in var_dict.items():
            if var_name not in self.scaler or idx >= data_array.shape[1]:
                continue
            
            params = self.scaler[var_name]
            min_val, max_val = float(params['min']), float(params['max'])
            was_log_scaled = params.get('was_log_scaled', False)

            scaled_values = data_array[:, idx, :, :].astype(float)

            # Step 1: Inverse MinMax scaling
            rescaled_values = scaled_values * (max_val - min_val) + min_val
            
            if was_log_scaled:
                rescaled_values = np.exp(rescaled_values)
                rescaled_values = np.clip(rescaled_values, 0, None)

            # Handle missing values (replace -1.0 with NaN)
            rescaled_values[scaled_values == -1.0] = np.nan

            rescaled_data[var_name] = (['time', 'latitude', 'longitude'], rescaled_values)

        # Create coordinates
        coords = {
            'time': og_data['time'].isel(time=time_indices).values,
            'latitude': og_data['latitude'].values,
            'longitude': og_data['longitude'].values
        }

        return xr.Dataset(rescaled_data, coords=coords)


class AQIModelEvaluator:
    """Handles model evaluation, prediction generation, and loss computation."""

    def __init__(self, model_path, data_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ds = xr.open_dataset(data_path, engine='netcdf4')
        self.variables = list(self.ds.data_vars)

        # Convert xarray dataset to tensor
        self.ten_data = torch.stack(
            [torch.tensor(self.ds[var].values, dtype=torch.float32) for var in self.variables], dim=1
        )

        # Load trained model
        trainer = create_aqi_model.WeatherModelTrainer.load(model_path)
        self.model = trainer.model.to(self.device)
        self.model.eval()

        # Loss function
        self.loss_fn = create_aqi_model.ChannelWiseLoss(self.variables)

        # Initialize Rescaler
        self.rescaler = Rescaler(scaler_path)

    def evaluate(self, num_epochs=5):
        """Evaluates the model and returns rescaled predictions and targets."""
        random_index = torch.randint(8, self.ten_data.shape[0] - 8, (1,)).item()
        spatial_data = self.ten_data[random_index].unsqueeze(0).to(self.device)
        temporal_window = self.ten_data[random_index - 8: random_index]
        temporal_data = temporal_window.permute(1, 0, 2, 3).flatten(2).unsqueeze(0).to(self.device)
        target = self.ten_data[random_index + 1].unsqueeze(0).to(self.device)

        all_outputs = spatial_data.clone()
        all_targets = target.clone()
        time_indices = [random_index]

        for epoch in range(num_epochs):
            val_channel_losses = {name: 0.0 for name in self.variables}
            with torch.no_grad():
                outputs = self.model(spatial_data, temporal_data)
                loss_dict = self.loss_fn(outputs, target)

                for name in self.variables:
                    val_channel_losses[name] += loss_dict['per_channel'][name]

            print(f"\nEpoch {epoch + 1} Channel Losses:")
            for name in self.variables:
                print(f"{name:<15}  Val: {val_channel_losses[name]:.4f}")

            random_index += 1
            target = self.ten_data[random_index + 1].unsqueeze(0).to(self.device)

            # Update spatial and temporal data
            spatial_data = outputs.to(self.device)
            temporal_data_new = temporal_data[:, 1:, :, :]
            outputs_reshaped = outputs.unsqueeze(0).permute(0, 2, 1, 3, 4).flatten(2).unsqueeze(0)
            temporal_data = torch.cat([temporal_data_new, outputs_reshaped], dim=1)

            # Store outputs and targets
            all_outputs = torch.cat([all_outputs, spatial_data], dim=0)
            all_targets = torch.cat([all_targets, target], dim=0)
            time_indices.append(random_index)

        return all_outputs.cpu().numpy(), all_targets.cpu().numpy(), time_indices


def plot_comparison_maps(predictions, targets, variable, figsize=(10, 20), cmap='viridis'):
    """Plots a comparison of predictions vs targets."""
    common_times = predictions[variable].time
    n_times = len(common_times)
    fig, axes = plt.subplots(n_times, 2, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

    fig.suptitle(f'{variable.upper()} Comparison: Predictions vs Targets', y=1.02, fontsize=16, weight='bold')

    for i, time in enumerate(common_times):
        pred = predictions[variable].sel(time=time)
        target = targets[variable].sel(time=time)

        plot_map(pred, axes[i, 0], f'Prediction\n{time.item()}', cmap)
        plot_map(target, axes[i, 1], f'Actual\n{time.item()}', cmap)

    plt.tight_layout()
    return fig


def plot_map(data, ax, title, cmap='viridis'):
    """Plots a single map."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=True,
              cbar_kwargs={'label': f'{data.attrs.get("long_name", data.name)} ({data.attrs.get("units", "")})',
                           'shrink': 0.8, 'extend': 'both'})

    ax.set_title(title, pad=12, fontsize=10)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False


# === Running the Pipeline ===
if __name__ == "__main__":
    evaluator = AQIModelEvaluator(
        model_path='aqi_model_v2.pth',
        data_path='complete_data_aqi_log.nc',
        scaler_path='weather_data_aqi_log.save'
    )

    outputs_np, targets_np, time_indices = evaluator.evaluate()

    var_dict = {'co_conc': 0, 'o3_conc': 1, 'pm10_conc': 2, 'pm2p5_conc': 3, 'so2_conc': 4, 'no2_conc': 5,
                'u10': 6, 'v10': 7}

    rescaled_outputs = evaluator.rescaler.rescale_xarray(outputs_np, var_dict, time_indices, evaluator.ds)
    rescaled_outputs.to_netcdf('rescaled_pred.nc')

    rescaled_targets = evaluator.rescaler.rescale_xarray(targets_np, var_dict, time_indices, evaluator.ds)
    rescaled_targets.to_netcdf('rescaled_tar.nc')

    fig = plot_comparison_maps(rescaled_outputs, rescaled_targets, variable='so2_conc')
    fig.savefig('so2_conc_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
