import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.interpolate import griddata

class WeatherDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()  # Single scaler for all variables
        self.scaler_params = {}  # To store scaling parameters per variable
        self.log_scaled_vars = []  # Track which variables were log-transformed

    def time_lat_long(self, dataset):
        """Process time, latitude, and longitude coordinates"""
        timestamp = dataset.time.long_name[19:29]
        timestamp_init = datetime.strptime(timestamp, '%Y%m%d')
        
        time_coords = pd.date_range(
            timestamp_init, 
            periods=len(dataset.time), 
            freq='1h'
        ).strftime("%Y-%m-%d %H:%M:%S").astype('datetime64[ns]')

        dataset_time = dataset.assign_coords(time=time_coords)
        dataset_time_lat_long = dataset_time.assign_coords(longitude=(((dataset_time.longitude + 180) % 360) - 180))
        return dataset_time_lat_long.sortby('longitude')
    

    def minmax_scale(self, dataset, log_transform_vars=None, epsilon=1e-6):
        """
        Apply MinMax scaling.
    
        Args:
            dataset: xarray Dataset
        
        Returns:
            Scaled xarray Dataset
        """
        if log_transform_vars is None:
            log_transform_vars = ['so2_conc', 'no2_conc', 'o3_conc', 'pm10_conc', 'pm2p5_conc',]

        scaled_vars = {}

        for var in dataset.data_vars:
            values = dataset[var].values.copy()
            original_shape = values.shape

            # Optional unit conversions
            if var == 'co_conc':
                values *= (24.45 / 28010)
            elif var == 'no2_conc':
                values *= (24.45 / 46.005)
            elif var == 'so2_conc':
                values *= (24.45 / 64.06)
            elif var == 'o3_conc':
                values *= (24.45 / 48)

            was_log_scaled = False
            if var in log_transform_vars:
                values = np.clip(values, epsilon, None)
                values = np.log(values)
                was_log_scaled = True

            # Apply MinMax scaling
            scaled_values = self.scaler.fit_transform(values.reshape(-1, 1)).reshape(original_shape)

            # Store scaling metadata
            self.scaler_params[var] = {
                'min': float(values.min()),
                'max': float(values.max()),
                'scale': float(self.scaler.scale_[0]),
                'was_log_scaled': was_log_scaled,
                'epsilon': epsilon if was_log_scaled else None
            }

            scaled_vars[var] = (dataset[var].dims, scaled_values)

        return xr.Dataset(scaled_vars, coords=dataset.coords)

    
    def save_scalers(self, filepath):
        """Save all scalers to disk"""
        joblib.dump(self.scaler_params, filepath)
        print(f"Saved scalers to {filepath}")
    
    def load_scalers(self, filepath):
        """Load scalers from disk"""
        self.scaler_params = joblib.load(filepath)
        print(f"Loaded scalers from {filepath}")


    
    def remove_nan(self, dataset):
        """
        1. Temporal interpolation (along time dimension).
        2. 2D spatial interpolation (latitude-longitude grid).
        3. Fallback filling for remaining NaNs.
        """
        # Step 1: Temporal interpolation
        dataset = dataset.interpolate_na(
        dim="latitude",
        method="linear"
            ).interpolate_na(
        dim="longitude",
        method="linear"
            )
        
        def get_mode(dataset, var_name):
            """Get mode of a variable in an xarray Dataset"""
            values = dataset[var_name].values.flatten()  # Flatten to 1D array
            valid_values = values[~np.isnan(values)]     # Remove NaN values
            unique, counts = np.unique(valid_values, return_counts=True)
            return unique[np.argmax(counts)]
        
        # 3. Mode imputation
        for var in dataset.data_vars:
            # Get mode for each variable
            var_mode = get_mode(dataset, var)
            if not np.isnan(var_mode):  # Only fill if mode exists
                dataset[var] = dataset[var].fillna(var_mode)
            #  Step 4: Final fill 
        return dataset.fillna(-1.0)

    def open_read(self, path):
        """Open and preprocess a single NetCDF file"""
        dataset = xr.open_dataset(path, engine='netcdf4', decode_timedelta=False,)
        dataset = dataset.drop_vars('level').squeeze('level')
        dataset = self.time_lat_long(dataset)
        return dataset


def main():
    processor = WeatherDataPreprocessor()
    
    # Path configuration
    base_path = r"C:\mahie\DataScienceCourse\march"
    subfolders = [
        'carbon_monoxide', 
        'ozone', 
        'particulate_matter_10um', 
        'particulate_matter_2.5um', 
        'sulphur_dioxide', 
        'nitrogen_dioxide'
    ]
    
    # Load pollution data
    pollution_files = []
    for folder in subfolders:
        subfolder_path = os.path.join(base_path, folder)
        pollution_files.extend([os.path.join(subfolder_path, f).replace("\\", "/") for f in os.listdir(subfolder_path)])
    
    pollution_data = xr.combine_by_coords([processor.open_read(f) for f in pollution_files],combine_attrs='override')

    # Load wind data
    wind_path = os.path.join(base_path, "wind")
    wind_files = [os.path.join(wind_path, f) for f in os.listdir(wind_path)]
    wind_data = xr.combine_by_coords([xr.open_dataset(f) for f in wind_files], combine_attrs='override')
    
    # Process wind data
    wind_data.coords['time'] = wind_data.coords['valid_time']
    wind_data = wind_data.swap_dims({'valid_time': 'time'})
    
    # Align time dimensions (the pollutant data is in 3 hour interval, while the wind is hourly data)
    common_time = np.intersect1d(wind_data.time, pollution_data.time)
    wind_data = wind_data.sel(time=common_time)
    
    # Combine datasets
    combined = xr.combine_by_coords([pollution_data, wind_data], combine_attrs='override')
    
    # Apply the NaN handling part
    combined = processor.remove_nan(combined) 

    combined = processor.minmax_scale(combined)
    processor.save_scalers("weather_data_aqi_log.save")
    
    
    combined = combined.drop_vars(['expver', 'number', 'valid_time'])
    
    
    return combined, processor


if __name__ == '__main__':
    weather_xarray, _ = main()

    weather_xarray.to_netcdf('complete_data_aqi_log.nc', format="NETCDF4")