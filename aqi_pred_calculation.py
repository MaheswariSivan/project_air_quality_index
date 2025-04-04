import xarray as xr
import numpy as np
import torch
import create_aqi_model
import joblib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
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
            print(var_name, min_val, max_val)
            scaled_values = data_array[:, idx, :, :].astype(float)

            # Step 1: Inverse MinMax scaling
            rescaled_values = (scaled_values * (max_val - min_val)) + min_val

            if was_log_scaled:
                rescaled_values = np.exp(rescaled_values)
                rescaled_values = np.clip(rescaled_values, 0, None)
                
            # Handle missing values (replace -1.0 with NaN)
            rescaled_values[scaled_values == -1.0] = np.nan

            rescaled_data[var_name] = (['time', 'latitude', 'longitude'], rescaled_values)

        ds = xr.Dataset(
        rescaled_data,
        coords={
            'latitude': og_data['latitude'].values,
            'longitude': og_data['longitude'].values,
            'new_time': og_data['time'].isel(time=time_indices).values  # Use temporary dim
        }
        )
        
        # 4. Rename dimension safely
        ds = ds.rename({'new_time': 'time'})
        
        # 5. Ensure proper encoding
        ds['time'].attrs.update(og_data['time'].attrs)

        return ds
    

class EAQICalculator:
    # (Previous EAQI_BREAKPOINTS and calculate_aqi methods remain the same)
    @staticmethod
    def calculate_aqi(concentration, pollutant):
        """Calculate European Air Quality Index (EAQI)"""
        EAQI_BREAKPOINTS = {
            'pm2p5_conc': [(0,20,0,10), 
                           (20,40,10,20), 
                           (40,60,20,25), 
                        (60,80,25,50), 
                        (80,100,50,75), 
                        (100,150,75,800)],
            'pm10_conc': [(0,20,0,20), 
                          (20,40,20,40), 
                          (40,60,40,50),
                    (60,80,50,100), 
                    (80,100,100,150), (100,150,150,1200)],
            'o3_conc': [(0,50,0,50), (50,100,50,100), (100,130,100,130),
                  (130,160,130,160), (160,200,160,200), (200,300,200,600)],
            'no2_conc': [(0,50,0,50), (50,100,50,100), (100,150,100,150),
                   (150,200,150,200), (200,300,200,400), (300,500,400,1000)],
            'so2_conc': [(0,50,0,50), (50,100,50,100), (100,150,100,150),
                   (150,200,150,200), (200,300,200,350), (300,500,350,500)],
            'co_conc': [(0,5,0,5), (5,10,5,10), (10,15,10,15),
                   (15,20,15,20), (20,30,20,30), (30,50,30,60)]
        }
        
        bp = EAQI_BREAKPOINTS.get(pollutant)
        if bp is None:
            raise ValueError(f"Unknown pollutant: {pollutant}")
            
        # Initialize output array with NaNs
        result = np.full_like(concentration, np.nan, dtype=np.float64)

        # Vectorized calculation
        for c_low, c_high, i_low, i_high in bp:
            mask = (concentration >= c_low) & (concentration <= c_high)
            # Only calculate for values in current range
            result[mask] = (
                ((i_high - i_low) / (c_high - c_low)) * 
                (concentration[mask] - c_low) + 
                i_low
            )
        
        return result

    @staticmethod
    def create_eaqi_dataset(prediction_ds, pollutant_dict, time_indices, reference_ds):
        """
        Convert model predictions to xarray Dataset with EAQI calculations.
        
        Args:
            prediction_tensor: Model output tensor (shape: [time, vars, lat, lon])
            pollutant_dict: Dictionary mapping pollutant names to their indices
            time_indices: Time indices for the output Dataset
            reference_ds: Original dataset for coordinate reference
            
        Returns:
            xarray Dataset with EAQI calculations
        """
        # Initialize storage for variables
        eaqi_data = {}
        pollutant_data = {}
        
        # Process each pollutant
        for poll_name, var_name in pollutant_dict.items():
            if poll_name not in prediction_ds:
                continue
                
            # Get the prediction data
            pred_values = prediction_ds[poll_name].values
            print(pred_values.shape)
            # Store raw predictions
            pollutant_data[poll_name] = (['time', 'latitude', 'longitude'], 
                                    pred_values)
            
            # Calculate EAQI
            eaqui_values = EAQICalculator.calculate_aqi(
                pred_values,
                poll_name
            )
            eaqi_data[f'EAQI_{poll_name}'] = (['time', 'latitude', 'longitude'],
                                            eaqui_values)
    
        
        # Create coordinates
        coords = {
            'time': reference_ds['time'].isel(time=time_indices).values,
            'latitude': reference_ds['latitude'].values,
            'longitude': reference_ds['longitude'].values
        }
        
        # Create base Dataset
        ds = xr.Dataset(pollutant_data, coords=coords)
        # Add EAQI calculations
        for var_name, (dims, data) in eaqi_data.items():
            ds[var_name] = (dims, data)
        
        # Calculate overall EAQI
        eaqi_vars = [f'EAQI_{poll}' for poll in pollutant_dict.keys() 
                     if f'EAQI_{poll}' in ds]
        if eaqi_vars:
            stacked = xr.concat([ds[v] for v in eaqi_vars], dim='pollutant')
            ds['EAQI'] = stacked.max(dim='pollutant')
        
        # Preserve attributes
        ds.attrs.update(reference_ds.attrs)
        
        return ds

def load_and_prepare_data(data_path):
    """Load and prepare input data"""
    ds = xr.open_dataset(data_path, engine='netcdf4')
    variables = list(ds.data_vars)
    ten_data = torch.stack([torch.tensor(ds[var].values, dtype=torch.float32) 
                           for var in variables], dim=1)
    return ds, ten_data

def prepare_model_inputs(tensor_data):
    """Prepare spatial and temporal model inputs"""
    spatial_data = tensor_data[-1].unsqueeze(0)  # Add batch dim
    temporal_window = tensor_data[-8:]  # Last 8 timesteps
    temporal_data = temporal_window.permute(1,0,2,3).flatten(2).unsqueeze(0)
    return spatial_data, temporal_data

def run_model_prediction(model, spatial_input, temporal_input, device):
    """Execute model prediction"""
    model.eval()
    with torch.no_grad():
        return model(
            spatial_input.to(device),
            temporal_input.to(device)
        ).cpu()

def visualize_results(aqi_data, output_path):
    """Generate EAQI visualization plot"""
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    
    # Create plot with dynamic scaling
    plot = aqi_data['EAQI'].plot(
        ax=ax,
        x='longitude',
        y='latitude',
        transform=ccrs.PlateCarree(),
        cmap='RdYlGn_r',
        cbar_kwargs={
            'label': 'EAQI Level',
            'shrink': 0.7,
            'spacing': 'proportional',
            'format': '%.1f'
        }
    )
    
    
    # Add map features
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    plt.title("European Air Quality Index\nActual range:")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    DATA_PATH = 'complete_data_aqi_log.nc'
    MODEL_PATH = 'aqi_model_v2.pth'
    SCALER_PATH = 'weather_data_aqi_log.save'
    OUTPUT_IMAGE = 'eaqi_prediction.png'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    ds, ten_data = load_and_prepare_data(DATA_PATH)
    
    # Prepare model inputs
    spatial_input, temporal_input = prepare_model_inputs(ten_data)
    
    # Load and run model
    trainer = create_aqi_model.WeatherModelTrainer.load(MODEL_PATH)
    raw_outputs = run_model_prediction(
        trainer.model, spatial_input, temporal_input, device)
    
    raw_outputs = raw_outputs.cpu().numpy()
    # Rescale outputs
    var_dict = {'co_conc':0, 'o3_conc':1, 'pm10_conc':2, 'pm2p5_conc':3, 'so2_conc':4, 'no2_conc':5}
    rescaler = Rescaler(SCALER_PATH)
    rescaler = rescaler.rescale_xarray( raw_outputs, var_dict, -1, ds)
    
    # Calculate EAQI
    aqi_data = EAQICalculator.create_eaqi_dataset(
        rescaler,
        pollutant_dict=var_dict,
        time_indices=slice(-1, None),  # Last timestep
        reference_ds=ds
    )
    
    # Visualize and save results
    visualize_results(aqi_data, OUTPUT_IMAGE)
    aqi_data.to_netcdf('aqi_predictions.nc')

if __name__ == "__main__":
    main()