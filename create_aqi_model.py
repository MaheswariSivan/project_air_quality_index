import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split



class WeatherDataset(Dataset):
    def __init__(self, data, target_vars=8, input_seq=8, forecast_int=1):
        """
        To get all the needed variables for the model to train.
        
        Args:
            data: 4D tensor (Time, Variables, Lat, Lon)
            input_seq: Past timesteps to use (default 8)
            forecast_int: Forecast horizon (default 1)
        """
        self.data = data.float()  # (Time, Variables, Lat, Lon)
        self.input_seq = input_seq
        self.forecast_int = forecast_int
        self.target_indices = torch.tensor(target_vars)

    def __len__(self):
        return self.data.shape[0] - self.input_seq - self.forecast_int

    def __getitem__(self, idx):
        """
        Returns:
            X: Tuple of (spatial_data, time_series_data)
                - spatial_data: (V, H, W) last timestep's snapshot
                - time_series_data: (V, T, H*W) historical sequences
            Y: Target (V, H, W) all variables at t+forecast_int
        """
        # 1. SPATIAL DATA - last timestep's full state
        spatial_data = self.data[idx + self.input_seq - 1]  # (V, H, W)
        
        # 2. TIME SERIES DATA - historical sequences (only taking variables 0-6 since last 2 are wind components)
        window = self.data[idx:idx + self.input_seq][0:6]  # (T, V, H, W)
        time_series_data = window.permute(1, 0, 2, 3)  # (V, T, H, W)
        time_series_data = time_series_data.flatten(2)  # (V, T, H*W)
        
        # 3. TARGET - 6 variables at future timestep (last 2 variables are wind components)
        target = self.data[idx + self.input_seq + self.forecast_int][0:self.target_indices]  # (6, H, W)
        
        return (spatial_data, time_series_data), target
    




class ChannelAttention(nn.Module):
    """
    Block for channel attention. Calculaltes the correlation (weights) between the channels.
    The block is similar to the Convolutional Block Attention Model (CBAM), without the spatial attention part. 
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = self.fc(y_avg + y_max).view(b, c, 1, 1)
        return x * y.expand_as(x)  # [batch, channels, lat, lon]
        




class SpatialAttention(nn.Module):
    """Self-attention in spatial dimension
    This block of code is to make the model understand the relationship between the adjucent pixels/ locations.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.reduction = reduction
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Adaptive pooling to fixed size
        target_h, target_w = max(1, H//self.reduction), max(1, W//self.reduction)
        x_down = F.adaptive_avg_pool2d(x, (target_h, target_w))
        
        # Attention operations
        q = self.query(x_down).view(B, -1, target_h * target_w)
        k = self.key(x_down).view(B, -1, target_h * target_w)
        v = self.value(x_down).view(B, -1, target_h * target_w)
        
        attn = torch.bmm(q.transpose(1,2), k)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.transpose(1,2))
        out = out.view(B, C, target_h, target_w)
        
        # Adaptive upsample
        out = F.interpolate(out, size=(H, W), mode='bilinear')
        
        return self.gamma * out + x  # Residual connection
    



class CNNBranch(nn.Module):
    """This block is to combine the channel attention and self attention in spatial dimension."""
    def __init__(self, in_channels=8, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.ca1 = ChannelAttention(32)
        self.spatial_attn1 = SpatialAttention(32)  # Added here
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.ca2 = ChannelAttention(out_channels)
        self.spatial_attn2 = SpatialAttention(out_channels)  # And here

    def forward(self, x):
        x = F.relu(self.ca1(self.conv1(x)))
        x = self.spatial_attn1(x)  # Added
        x = F.relu(self.ca2(self.conv2(x)))
        x = self.spatial_attn2(x)  # Added
        return x
    




class LSTMBranch(nn.Module):
    """
    To make the model learn the time series pattern of each variable.
    Process each time-series variable independently.
    """
    def __init__(self, input_dim , num_variables = 8, out_channels=32):
        super().__init__()
        self.num_variables = num_variables
        self.lstm = nn.LSTM(input_dim, out_channels, batch_first=True)
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # x shape: [batch, num_variables, time_steps, input_dim]
        batch_size = x.shape[0]
        outputs = []
        for i in range(self.num_variables):
            var_data = x[:, i, :, :]  # [batch, time_steps, input_dim]
            _, (h_n, _) = self.lstm(var_data)  # h_n: [1, batch, out_channels]
            var_feat = self.proj(h_n.squeeze(0))  # [batch, out_channels]
            outputs.append(var_feat)

        #print(f"lsmt output shape: {(torch.stack(outputs, dim=1)).shape}")
        # Stack per-variable features
        return torch.stack(outputs, dim=1)  # [batch, num_variables, out_channels]





class SpatialFusion(nn.Module):
    """To combine CNNBranch and LSTM features into desired (channels, lat, lon) output.
    Output from LSTM has to be reshaped in order to be combined with CNNBranch and get the final output.
    """
    def __init__(self, cnn_channels, lstm_channels, num_variables, out_channels):
        super().__init__()
        # Project LSTM outputs to spatial maps
        self.lstm_to_spatial = nn.Conv2d(
            6 * lstm_channels, 
            lstm_channels, 
            kernel_size=1
        )
        # Combine CNN and LSTM features
        self.fuse_conv = nn.Conv2d(
            cnn_channels + lstm_channels, 
            out_channels, 
            kernel_size=1
        )

    def forward(self, cnn_out, lstm_out):
        # cnn_out: [batch, cnn_channels, lat, lon]
        # lstm_out: [batch, num_variables, lstm_channels]
        batch, num_vars, lstm_dim = lstm_out.shape
        h, w = cnn_out.shape[-2:]

        # Reshape LSTM outputs to spatial dimensions
        lstm_out = lstm_out.reshape(batch, num_vars * lstm_dim, 1, 1)
        lstm_out = F.interpolate(lstm_out, size=(h, w), mode='bilinear')  # [batch, lstm_dim, lat, lon]
        lstm_out = self.lstm_to_spatial(lstm_out)

        # Concatenate and fuse
        fused = torch.cat([cnn_out, lstm_out], dim=1)  # [batch, cnn+lstm, lat, lon]
        #print(f"spatial fusion function output shape: {fused.shape}")
        return self.fuse_conv(fused)  # [batch, out_channels, lat, lon]





class SpatioTemporalModel(nn.Module):
    """
    The final block. The inputs are taken and passed into their respective block for learning. 
    
    """
    def __init__(self, input_vars=8, target_vars=8, spatial_shape=(110, 128)):
        super().__init__()
        self.target_vars = target_vars
        
        # CNN processes all input variables
        self.cnn = CNNBranch(in_channels=input_vars, out_channels=64)
        
        # LSTM processes all variables' histories
        self.lstm = LSTMBranch(input_dim=spatial_shape[0]*spatial_shape[1], 
                             num_variables=6, 
                             out_channels=64)
        
        self.fusion = SpatialFusion(
            cnn_channels=64,
            lstm_channels=64,
            num_variables=input_vars,  # Still use all variables for fusion
            out_channels=128
        )
        
        # Final conv outputs only target variables
        self.output_conv = nn.Conv2d(128, target_vars, kernel_size=1)

    def forward(self, x_spatial, x_temporal):
        #print(f"cnn_input shape: {x_spatial.shape}")
        cnn_out = self.cnn(x_spatial)  # Processes all 8 vars
        #print(f"lstm_input shape: {x_temporal.shape}")
        lstm_out = self.lstm(x_temporal)  # Processes all 8 vars
        #print(f"lstm_output shape: {lstm_out.shape}")
        fused = self.fusion(cnn_out, lstm_out)
        #print(f"fused_output shape: {x_spatial.shape}")
        return self.output_conv(fused)  # Outputs 6 vars
   




class ChannelWiseLoss(nn.Module):
    """To get channel wise loss. This is see how well the model learned different variables. """
    def __init__(self, channel_names, weights=None):
        super().__init__()
        self.channel_names = channel_names
        self.weights = torch.ones(len(channel_names)) if weights is None else torch.tensor(weights)
        
    def forward(self, pred, target):
        #print(pred.shape)
        #print(target.shape)
        squared_error = (pred - target)**2  # [B, C, H, W]
        per_channel = squared_error.mean(dim=(0, 2, 3))  # [C]
        weighted_loss = (per_channel * self.weights.to(pred.device)).mean()
        return {
            'total_loss': weighted_loss,
            'per_channel': {name: val.item() for name, val in zip(self.channel_names, per_channel)}
        }




class WeatherDataHandler:
    def __init__(self, data_path, input_vars = 8, target_vars = 8, batch_size=8):
        """
        Handles loading and preparation of weather data.
        
        Args:
            data_path (str): Path to NetCDF file
            input_vars (int): Number of input variables
            target_vars (int): Number of target variables
            batch_size (int): Batch size for DataLoader (default 8)
        """
        self.data_path = data_path
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.train_ratio = 0.8
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_data()
        self.prepare_datasets()
        
    def load_data(self):
        """Load and process the NetCDF data."""
        ds = xr.open_dataset(self.data_path)
        variables = list(ds.data_vars)
        
        # Convert to tensor
        self.ten_data = torch.stack([torch.tensor(ds[var].values, dtype=torch.float32) for var in variables],dim=1)
        
        # Store channel names
        self.channel_names_all = variables
        self.channel_names = variables[:self.target_vars]
        
    def prepare_datasets(self):
        """Create datasets and data loaders."""
        full_dataset = WeatherDataset(self.ten_data)
        train_size = int(self.train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class WeatherModelTrainer:
    def __init__(self, data_handler):
        """
        Handles model training and evaluation.
        
        Args:
            data_handler (WeatherDataHandler): Prepared data handler instance
            spatial_shape (tuple): Spatial dimensions of input data
            learning_rate (float): Initial learning rate (default 1e-3)
            weight_decay (float): Weight decay for optimizer (default 1e-5)
            max_norm (float): Gradient clipping norm (default 1.0)
        """
        self.data_handler = data_handler
        
        self.device = data_handler.device
        self.channel_names = data_handler.channel_names
        
        self.initialize_model()
        self.setup_training()
        
    def initialize_model(self):
        """Initialize model and loss function."""
        self.model = SpatioTemporalModel(
            input_vars=self.data_handler.input_vars,
            target_vars=self.data_handler.target_vars,
            spatial_shape= (110,128)
        ).to(self.device)
        
        self.loss_fn = ChannelWiseLoss(self.channel_names).to(self.device)
        
    def setup_training(self):
        """Configure optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr = 1e-3,
            weight_decay= 1e-5 ,
            betas=(0.9, 0.999)
            )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr = 1e-3,
            steps_per_epoch=len(self.data_handler.train_loader),
            epochs=200  # Max epochs for scheduler
        )
        
        self.train_channel_history = {name: [] for name in self.channel_names}
        
    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        epoch_channel_losses = {name: 0.0 for name in self.channel_names}
        
        for batch_idx, ((spatial, temporal), target) in enumerate(self.data_handler.train_loader):
            spatial = spatial.to(self.device)
            temporal = temporal.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(spatial, temporal)
            
            loss_dict = self.loss_fn(outputs, target)
            loss_dict['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for name in self.channel_names:
                epoch_channel_losses[name] += loss_dict['per_channel'][name]
        
        # Store average losses
        for name in self.channel_names:
            self.train_channel_history[name].append(epoch_channel_losses[name] / len(self.data_handler.train_loader))
        
    def validate(self, epoch):
        """Run validation."""
        self.model.eval()
        val_channel_losses = {name: 0.0 for name in self.channel_names}
        
        with torch.no_grad():
            for (spatial, temporal), target in self.data_handler.val_loader:
                spatial = spatial.to(self.device)
                temporal = temporal.to(self.device)
                target = target.to(self.device)
                
                outputs = self.model(spatial, temporal)
                loss_dict = self.loss_fn(outputs, target)
                
                for name in self.channel_names:
                    val_channel_losses[name] += loss_dict['per_channel'][name]
        
        # Print results
        print(f"\nEpoch {epoch+1} Channel Losses:")
        for name in self.channel_names:
            train_loss = self.train_channel_history[name][-1]
            val_loss = val_channel_losses[name] / len(self.data_handler.val_loader)
            print(f"{name:<15} Train: {train_loss:.4f}  Val: {val_loss:.4f}")
    
    def train(self, num_epochs=10):
        """Run full training loop."""
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            self.validate(epoch)
            self.scheduler.step()

    def save(self, path):
        # Save model
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'input_vars': self.data_handler.input_vars,
        'target_vars': self.data_handler.target_vars,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'channel_names': self.channel_names
        }, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path):
        """Load model with this line of code"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(path, map_location=device)

        # Verify required keys
        required_keys = {'model_state_dict', 'input_vars', 'target_vars', 'channel_names'}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Saved model is missing required keys: {missing}")

        # Create a proper dummy handler with all required attributes
        class DummyHandler:
            def __init__(self, config):
                self.input_vars = config['input_vars']
                self.target_vars = config['target_vars']
                self.channel_names = config['channel_names']
                self.device = device
                # Add dummy attributes needed by trainer
                self.train_loader = []  # Empty list to avoid errors
                self.val_loader = []    # Empty list to avoid errors
                self.train_ratio = 0.8  # Default value
                self.batch_size = 8     # Default value
        
        # Create trainer instance without calling setup_training()
        trainer = WeatherModelTrainer.__new__(WeatherModelTrainer)
        trainer.data_handler = DummyHandler(data)
        trainer.spatial_shape = (110,128)
        trainer.device = device
        trainer.channel_names = data['channel_names']
        
        # Manually initialize just the model (skip training setup)
        trainer.model = SpatioTemporalModel(
            input_vars=data['input_vars'],
            target_vars=data['target_vars'],
            spatial_shape=(110, 128)
        ).to(device)
        
        # Load model state
        trainer.model.load_state_dict(data['model_state_dict'])
        
        # Initialize loss function
        trainer.loss_fn = ChannelWiseLoss(data['channel_names']).to(device)
        
        return trainer


if __name__ == '__main__':
    """this block is executed only when we run this particular file. The remaining can be separately imported and used."""
    # Data preparation
    data_config = {
        'data_path': "complete_data_aqi.nc",
        'input_vars': 8,
        'target_vars':8,
        'batch_size':8
    }
    data_handler = WeatherDataHandler(**data_config)
        
    # Model training
    trainer = WeatherModelTrainer(data_handler)
    trainer.train(num_epochs=20)

    trainer.save("aqi_model_v2.pth")