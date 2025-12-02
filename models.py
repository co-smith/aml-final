import torch
import torch.nn as nn
import torch.fft
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject time order information
    into the Transformer.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Model B: A Transformer Encoder for Time Series Forecasting.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Input Projection: Map raw features (temp, wind, etc.) to d_model size
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layers
        # We use batch_first=True so inputs are (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Decoder/Projection to scalar output (Temperature)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (Batch, Seq_Len, Input_Dim)
        
        # Project and add position info
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        
        # Pass through Transformer
        # memory shape: (Batch, Seq_Len, d_model)
        memory = self.transformer_encoder(src)
        
        # We only care about the output of the *last* time step for forecasting
        # Take the feature vector of the last token in the sequence
        last_time_step = memory[:, -1, :] 
        
        # Project to target
        output = self.decoder(last_time_step)
        return output

class SpectralConv1d(nn.Module):
    """
    The heart of the FNO: 1D Fourier Layer.
    """
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of low-frequency modes to keep

        # The learnable weights in the Fourier Domain
        # Shape: (in, out, modes) with complex numbers
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def complex_mul1d(self, input, weights):
        # (Batch, In_Ch, Modes) * (In_Ch, Out_Ch, Modes) -> (Batch, Out_Ch, Modes)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Fast Fourier Transform (Real to Complex)
        x_ft = torch.fft.rfft(x)

        # 2. Filter: Multiply the lowest 'modes' by the learnable weights
        # We start with zeros for the output spectrum
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply only the kept modes
        out_ft[:, :, :self.modes] = self.complex_mul1d(x_ft[:, :, :self.modes], self.weights)

        # 3. Inverse Fast Fourier Transform (Complex to Real)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SimpleFNO(nn.Module):
    """
    Model C: A simple 1D Fourier Neural Operator.
    """
    def __init__(self, input_dim, d_model=64, modes=4):
        super(SimpleFNO, self).__init__()
        
        # Lift raw input to d_model channels
        # 1D time series: Channels = Features, Length = Time.
        
        self.fc0 = nn.Linear(input_dim, d_model) 

        # FNO Block 1
        self.conv0 = SpectralConv1d(d_model, d_model, modes)
        self.w0 = nn.Conv1d(d_model, d_model, 1) 

        # FNO Block 2
        self.conv1 = SpectralConv1d(d_model, d_model, modes)
        self.w1 = nn.Conv1d(d_model, d_model, 1)

        # Projection to Output
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.act = nn.GELU()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # Project features up
        x = self.fc0(x) 
        
        # Permute for FNO: (Batch, Channels/Features, Seq_Len)
        x = x.permute(0, 2, 1)
        
        # Block 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)

        # Block 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        # Permute back: (Batch, Seq_Len, Channels)
        x = x.permute(0, 2, 1)

        # We take the last time step for prediction
        x = x[:, -1, :]
        
        # Projection Head
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x