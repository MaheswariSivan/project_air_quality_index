# project_air_quality_index

## Overview
This project focuses on predicting air quality by analyzing key air pollutants and meteorological data retrieved via the European Environment Agency (EEA) API (*Data*: [EEA Air Quality API](https://www.eea.europa.eu/data-and-maps/apis), *API Documentation*: [EEA API Help](https://www.eea.europa.eu/help)). The primary pollutants considered include:

 - Carbon Monoxide (CO)
 - Nitrogen Dioxide (NO₂)
 - Ozone (O₃)
 - Particulate Matter (PM₂.₅ & PM₁₀)
 - Sulfur Dioxide (SO₂)

Additionally, wind speed and direction data are incorporated to enhance prediction accuracy, as wind dynamics significantly influence pollutant dispersion.
To ensure modularity and ease of understanding, the project is structured into distinct phases, each addressing a critical component of the predictive modeling pipeline.

## Data Preprocessing : (*data_preprocessing.py*)
In this step, all datasets are combined based on the shared dimensions: time, latitude, and longitude, to prepare the data for training a machine learning model.

The raw input comes in NetCDF4 format, ideal for storing spatiotemporal environmental data. It is loaded into an xarray.Dataset, which helps preserve the structure, metadata, and geospatial alignment. Some default dimensions that are not relevant for the model—such as "level"—are dropped to reduce redundancy and data size.

To ensure the data is clean and consistent:
 - Missing values (NaNs) are handled using interpolation along the latitude and longitude dimensions.
 - After interpolation, a logarithmic transformation is applied to selected variables to compress wide-ranging values and stabilize variance.
 - The data is then scaled using a MinMaxScaler, which transforms all variables to a consistent scale.

Both the scaled dataset and the corresponding scaling parameters (used later for inverse transformation) are saved as part of the output.

Input: Multiple NetCDF4 files containing raw environmental or atmospheric variables (preferablly in a single folder)

Output:
 - A cleaned and merged xarray Dataset with 3 core coordinates (time, latitude, longitude) and 8 selected variables, converted to a scaled NetCDF4 file.
 - Saved scaling metadata for later use

## Model Creation and Architecture: (*creat_aqi_model.py*)
In this stage, a deep learning model is constructed to learn from the spatiotemporal structure of the environmental data prepared in the previous step. The model is designed to process input data that has both spatial dependencies (e.g., how pollutants vary across geographical regions) and temporal patterns (e.g., how pollution changes over time).

The raw data is first converted from a NetCDF4 format into an xarray.Dataset, then transformed into a 4D PyTorch tensor of shape [time, variables, latitude, longitude]. The model inputs are split into two types:
 - Spatial Input: A single timestep's snapshot, capturing the current spatial distribution.
 - Temporal Input: A sequence of eight past timesteps, representing recent historical trends.

To handle this dual-input structure, the model employs a hybrid architecture:

A CNN branch extracts spatial features from the last timestep using convolutional layers enhanced with channel attention and spatial attention mechanisms. These attention blocks allow the model to focus on important variable interactions and geographical areas.

An LSTM branch captures temporal patterns from the time series input, processing each variable independently to learn trends over time and summarizing them into a compact representation.

A fusion module combines spatial and temporal features by projecting the LSTM output back to spatial dimensions and concatenating it with CNN features. A final convolutional layer then predicts the target variables at the next timestep.

The dataset is wrapped in a custom WeatherDataset class that prepares these input-output pairs, and WeatherDataHandler takes care of loading, splitting, and batching the data. The model is trained using a custom ChannelWiseLoss function that computes individual losses for each variable, allowing fine-grained monitoring of the model’s learning performance.

Model training is orchestrated by the WeatherModelTrainer class, which handles optimizer setup, learning rate scheduling, training and validation loops, and model checkpointing. The final trained model is saved and can be reloaded later using the provided static load() method.

This architecture enables the model to learn both the spatial correlations and temporal dynamics of air quality variables.

Inputs:
 - num_epochs: Number of training iterations over the dataset.
 - complete_data_aqi_log.nc: Normalized dataset used for evaluation (output from first part).
Output:
 - aqi_model_v2.pth: (If resuming) Previously saved model checkpoint.

## Auto-Regressive Evaluation (Self-Feeding Model Predictions) : (*model_check.py*)

In this part, the goal is to assess how well the model performs when it relies on its own past predictions rather than ground truth data—a process known as auto-regressive evaluation. A random time step is chosen, and the model is initially given one spatial frame along with a sequence of temporal inputs representing previous conditions. It predicts the next time step, then uses that prediction as part of the input for the following step, repeating this for several iterations. Throughout this loop, the model's predictions and the corresponding ground truth targets are collected. To convert the model’s normalized outputs into interpretable physical values, an inverse transformation is applied using a previously saved Rescaler. The rescaled outputs and actual values are then saved as NetCDF files, and a comparison visualization is generated to visually evaluate model performance over time.

Inputs:
 - aqi_model_v2.pth: Trained PyTorch model checkpoint (saved model /weights from second part).
 - complete_data_aqi_log.nc: Normalized dataset used for evaluation (output from first part).
 - weather_data_aqi_log.save: Scaler file containing min-max and log-scaling info for inverse transformation  (output from first part).

Outputs:
 - rescaled_pred.nc: NetCDF file containing inverse-transformed model predictions.
 - rescaled_tar.nc: NetCDF file of actual target values for comparison.
 - so2_conc_comparison.png: Side-by-side visual map of predicted vs actual SO₂ concentrations over time (the variable can be changed).

## EAQI Caluclation and visualization: (*aqi_pred_calculation.py*)

In the final stage, the output from the trained AQI model is used to compute the European Air Quality Index (EAQI) using standardized EU guidelines. First, the model’s raw predictions are loaded and rescaled back to their original concentration units using saved normalization parameters. Each pollutant’s concentration is then mapped to a standardized EAQI scale through breakpoint-based interpolation. The EAQI is calculated separately for all pollutants: PM2.5, PM10, NO₂, SO₂, CO, and O₃. These values are stored in an xarray dataset alongside the raw concentrations, and the overall EAQI is determined as the maximum index across all pollutants for each location and time step. Finally, a color-coded map visualizing the EAQI is generated and saved for interpretation and reporting.

Inputs:
 - aqi_model_v2.pth: Trained model for pollutant concentration prediction.
 - complete_data_aqi_log.nc: Input weather and pollution dataset in NetCDF format.
 - weather_data_aqi_log.save: Saved normalization (scaling) parameters.

Outputs:
 - aqi_predictions.nc: NetCDF file containing EAQI values and raw pollutant concentrations.
 - eaqi_prediction.png: EAQI map visualizing air quality levels across the region.
