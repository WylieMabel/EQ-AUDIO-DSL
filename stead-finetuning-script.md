# Complete STEAD Dataset + Wav2Vec2 Fine-tuning Script for Reconstruction

This script loads STEAD waveforms, prepares them for Wav2Vec2, and sets up fine-tuning for the reconstruction task.

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torchaudio.transforms as T

# =====================================================================
# PART 1: DATASET CLASS
# =====================================================================

class STEADReconstructionDataset(Dataset):
    """
    Loads STEAD waveforms from HDF5 and prepares for Wav2Vec2 reconstruction.
    """
    
    def __init__(self, 
                 hdf5_file_path, 
                 csv_file_path, 
                 processor,
                 target_sample_rate=16000,
                 original_sample_rate=100,
                 max_samples=None,
                 trace_category='noise',  # Can be 'noise' or 'earthquake'
                 channel=2):  # 0=E, 1=N, 2=Z (vertical)
        """
        Args:
            hdf5_file_path: Path to STEAD.hdf5
            csv_file_path: Path to STEAD.csv
            processor: Wav2Vec2Processor from HuggingFace
            target_sample_rate: Target sample rate (16kHz for Wav2Vec2)
            original_sample_rate: Original STEAD sample rate (100Hz)
            max_samples: Maximum number of samples to load (None = all)
            trace_category: 'noise' or 'earthquake'
            channel: Which channel to use (0=E, 1=N, 2=Z)
        """
        self.hdf5_file_path = hdf5_file_path
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.original_sample_rate = original_sample_rate
        self.channel = channel
        
        # Load CSV and filter
        self.df = pd.read_csv(csv_file_path)
        self.df = self.df[self.df['trace_category'] == trace_category]
        
        if max_samples:
            self.df = self.df.iloc[:max_samples]
        
        self.trace_names = self.df['trace_name'].values
        
        print(f"Loaded {len(self.trace_names)} {trace_category} traces from CSV")
    
    def __len__(self):
        return len(self.trace_names)
    
    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        
        try:
            # Load waveform from HDF5
            with h5py.File(self.hdf5_file_path, 'r') as f:
                # Shape: (6000, 3) for STEAD
                data = np.array(f['data/' + trace_name])
            
            # Select channel (e.g., vertical = channel 2)
            waveform = data[:, self.channel].astype(np.float32)
            
            # Normalize to [-1, 1]
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val
            else:
                waveform = waveform
            
            # Resample from 100Hz to 16kHz
            if self.original_sample_rate != self.target_sample_rate:
                waveform_resampled = librosa.resample(
                    waveform, 
                    orig_sr=self.original_sample_rate, 
                    target_sr=self.target_sample_rate
                )
            else:
                waveform_resampled = waveform
            
            # Process with Wav2Vec2 processor
            # Returns dict with 'input_values' (normalized, ready for model)
            processed = self.processor(
                waveform_resampled,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=False  # Don't pad yet, do in collate
            )
            
            return {
                'trace_name': trace_name,
                'input_values': processed['input_values'].squeeze(0),  # (audio_length,)
                'original_waveform': torch.from_numpy(waveform_resampled).float(),  # For reconstruction loss
            }
        
        except Exception as e:
            print(f"Error loading {trace_name}: {str(e)}")
            # Return a fallback
            return {
                'trace_name': trace_name,
                'input_values': torch.zeros(16000),  # 1 second of silence
                'original_waveform': torch.zeros(16000),
            }


def collate_fn_reconstruction(batch):
    """
    Custom collate function for batching variable-length audio.
    Pads input_values to the longest in the batch.
    """
    # Get max length in this batch
    max_length = max(item['input_values'].shape[0] for item in batch)
    
    padded_inputs = []
    padded_originals = []
    
    for item in batch:
        input_vals = item['input_values']
        original_wav = item['original_waveform']
        
        # Pad to max_length
        if input_vals.shape[0] < max_length:
            padding = torch.zeros(max_length - input_vals.shape[0])
            input_vals = torch.cat([input_vals, padding])
        
        if original_wav.shape[0] < max_length:
            padding = torch.zeros(max_length - original_wav.shape[0])
            original_wav = torch.cat([original_wav, padding])
        
        padded_inputs.append(input_vals)
        padded_originals.append(original_wav)
    
    return {
        'input_values': torch.stack(padded_inputs),  # (batch, max_length)
        'original_waveform': torch.stack(padded_originals),  # (batch, max_length)
    }


# =====================================================================
# PART 2: CUSTOM DECODER
# =====================================================================

class WaveformDecoder(nn.Module):
    """
    Decoder that reconstructs waveforms from Wav2Vec2 embeddings.
    """
    
    def __init__(self, encoder_hidden_size=768, output_length=960000):
        """
        Args:
            encoder_hidden_size: Dimension of Wav2Vec2 embeddings (usually 768)
            output_length: Target waveform length after reconstruction
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.output_length = output_length
        
        # Dense layers to expand embedding
        self.fc1 = nn.Linear(encoder_hidden_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        
        # Transposed convolutions to upsample
        self.deconv1 = nn.ConvTranspose1d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Output in [-1, 1]
    
    def forward(self, embedding):
        """
        Args:
            embedding: (batch, hidden_size) from Wav2Vec2 encoder
        
        Returns:
            reconstructed: (batch, output_length) waveform
        """
        # Expand embedding
        x = self.relu(self.fc1(embedding))  # (batch, 512)
        x = self.relu(self.fc2(x))  # (batch, 1024)
        x = self.relu(self.fc3(x))  # (batch, 2048)
        
        # Reshape for transposed conv
        x = x.unsqueeze(2)  # (batch, 2048, 1)
        
        # Upsample through transposed convolutions
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.relu(self.deconv5(x))
        x = self.tanh(self.deconv6(x))  # (batch, 1, output_length)
        
        # Handle length mismatch
        if x.shape[-1] != self.output_length:
            x = torch.nn.functional.interpolate(
                x, 
                size=self.output_length, 
                mode='linear', 
                align_corners=False
            )
        
        return x.squeeze(1)  # (batch, output_length)


# =====================================================================
# PART 3: LOSS FUNCTIONS
# =====================================================================

class ReconstructionLosses(nn.Module):
    """
    Multi-component reconstruction loss (similar to the paper).
    """
    
    def __init__(self, weights=None, device='cuda'):
        super().__init__()
        
        if weights is None:
            self.weights = {
                'time': 1.0,
                'spec': 0.5,
                'mel': 0.3,
            }
        else:
            self.weights = weights
        
        self.device = device
        
        # For mel-spectrogram computation
        self.melspec = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=400,
            hop_length=160
        ).to(device)
    
    def forward(self, original, reconstructed):
        """
        Args:
            original: (batch, length) original waveforms
            reconstructed: (batch, length) reconstructed waveforms
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses for logging
        """
        original = original.to(self.device)
        reconstructed = reconstructed.to(self.device)
        
        # Loss 1: Time-domain L1 loss
        loss_time = torch.nn.functional.l1_loss(original, reconstructed)
        
        # Loss 2: Spectral (FFT) loss
        original_spec = torch.abs(torch.fft.rfft(original, dim=-1))
        reconstructed_spec = torch.abs(torch.fft.rfft(reconstructed, dim=-1))
        loss_spec = torch.nn.functional.l1_loss(original_spec, reconstructed_spec)
        
        # Loss 3: Mel-spectrogram loss (more perceptually relevant)
        try:
            original_mel = self.melspec(original)
            reconstructed_mel = self.melspec(reconstructed)
            loss_mel = torch.nn.functional.l1_loss(
                torch.log(original_mel + 1e-9),
                torch.log(reconstructed_mel + 1e-9)
            )
        except:
            loss_mel = torch.tensor(0.0, device=self.device)
        
        # Total weighted loss
        total_loss = (
            self.weights['time'] * loss_time +
            self.weights['spec'] * loss_spec +
            self.weights['mel'] * loss_mel
        )
        
        return total_loss, {
            'time': loss_time.item(),
            'spec': loss_spec.item(),
            'mel': loss_mel.item() if loss_mel.item() > 0 else 0.0,
            'total': total_loss.item(),
        }


# =====================================================================
# PART 4: TRAINING SCRIPT
# =====================================================================

def train_reconstruction_model(
    hdf5_path,
    csv_path,
    output_dir='./checkpoints',
    num_epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    max_samples=None,  # Set to small number for testing
    device='cuda'
):
    """
    Full training loop for Wav2Vec2 reconstruction fine-tuning.
    """
    
    print("=" * 60)
    print("Wav2Vec2 Reconstruction Fine-tuning")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # --------- SETUP ---------
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Load Wav2Vec2 encoder
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    
    # IMPORTANT: Freeze encoder weights for fine-tuning (train only decoder)
    for param in wav2vec2.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded Wav2Vec2 (encoder frozen)")
    print(f"  Encoder hidden size: {wav2vec2.config.hidden_size}")
    
    # Create decoder
    decoder = WaveformDecoder(
        encoder_hidden_size=wav2vec2.config.hidden_size,
        output_length=960000  # ~60 seconds at 16kHz
    )
    
    # Create dataset and dataloader
    dataset = STEADReconstructionDataset(
        hdf5_file_path=hdf5_path,
        csv_file_path=csv_path,
        processor=processor,
        max_samples=max_samples,
        trace_category='noise',  # Start with noise for faster testing
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_reconstruction,
        num_workers=4
    )
    
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    # Setup loss, optimizer, scheduler
    loss_fn = ReconstructionLosses(device=device)
    decoder.to(device)
    wav2vec2.to(device)
    
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    print(f"✓ Setup optimizer and scheduler")
    
    # --------- TRAINING LOOP ---------
    
    for epoch in range(num_epochs):
        decoder.train()
        wav2vec2.eval()  # Encoder is frozen, so eval mode
        
        total_loss = 0
        loss_dict_cumulative = {'time': 0, 'spec': 0, 'mel': 0, 'total': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            input_values = batch['input_values'].to(device)  # (batch, length)
            original_waveforms = batch['original_waveform'].to(device)  # (batch, length)
            
            # Forward pass through frozen encoder
            with torch.no_grad():
                encoder_output = wav2vec2(input_values, output_hidden_states=True)
                # Extract embedding from last hidden state
                # Shape: (batch, time_steps, hidden_size)
                hidden_states = encoder_output.last_hidden_state
                
                # Pool over time dimension: take mean
                embedding = hidden_states.mean(dim=1)  # (batch, hidden_size)
            
            # Decoder reconstruction
            reconstructed = decoder(embedding)  # (batch, length)
            
            # Compute loss
            loss, loss_dict = loss_fn(original_waveforms, reconstructed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_dict_cumulative:
                loss_dict_cumulative[key] += loss_dict[key]
            
            # Logging
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"    Time: {loss_dict['time']:.6f}, Spec: {loss_dict['spec']:.6f}, Mel: {loss_dict['mel']:.6f}")
        
        # End of epoch
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        print(f"  Avg Time Loss: {loss_dict_cumulative['time']/len(dataloader):.6f}")
        print(f"  Avg Spec Loss: {loss_dict_cumulative['spec']/len(dataloader):.6f}")
        print(f"  Avg Mel Loss: {loss_dict_cumulative['mel']/len(dataloader):.6f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'decoder_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")
    
    print("✓ Training complete!")
    return wav2vec2, decoder, processor


# =====================================================================
# PART 5: INFERENCE / EMBEDDING EXTRACTION
# =====================================================================

def extract_embeddings(
    wav2vec2_model,
    processor,
    dataloader,
    device='cuda'
):
    """
    Extract embeddings from trained Wav2Vec2 encoder for downstream tasks.
    """
    
    wav2vec2_model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_values = batch['input_values'].to(device)
            
            encoder_output = wav2vec2_model(input_values, output_hidden_states=True)
            hidden_states = encoder_output.last_hidden_state  # (batch, time_steps, hidden_size)
            
            # Pool embeddings
            embedding = hidden_states.mean(dim=1)  # (batch, hidden_size)
            embeddings.append(embedding.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f"Extracted {batch_idx * len(batch['input_values'])} embeddings...")
    
    embeddings = np.vstack(embeddings)
    print(f"✓ Extracted {embeddings.shape[0]} embeddings of shape {embeddings.shape[1]}")
    
    return embeddings


# =====================================================================
# USAGE EXAMPLE
# =====================================================================

if __name__ == "__main__":
    # Configure paths
    STEAD_HDF5 = "/path/to/STEAD.hdf5"  # Change this!
    STEAD_CSV = "/path/to/STEAD.csv"    # Change this!
    
    # Train
    encoder, decoder, processor = train_reconstruction_model(
        hdf5_path=STEAD_HDF5,
        csv_path=STEAD_CSV,
        output_dir='./checkpoints',
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        max_samples=5000,  # Use small subset for testing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Extract embeddings for downstream tasks
    # (Create new dataloader for test set)
    dataset_test = STEADReconstructionDataset(
        hdf5_file_path=STEAD_HDF5,
        csv_file_path=STEAD_CSV,
        processor=processor,
        max_samples=1000,
        trace_category='earthquake',
    )
    
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_reconstruction,
        num_workers=4
    )
    
    embeddings = extract_embeddings(encoder, processor, dataloader_test)
    np.save('embeddings_downstream.npy', embeddings)
    print("✓ Embeddings saved!")
```

## Key Points About This Script

1. **Dataset Class:**
   - Loads STEAD waveforms from HDF5
   - Normalizes to [-1, 1]
   - Resamples from 100Hz → 16kHz
   - Processes with Wav2Vec2Processor

2. **Custom Decoder:**
   - Takes embedding and reconstructs waveform
   - Uses dense layers + transposed convolutions
   - Outputs tanh(-1 to 1) like original

3. **Loss Functions:**
   - Time-domain (L1)
   - Spectral (FFT-based)
   - Mel-spectrogram (perceptual)
   - All combined with weights

4. **Training Loop:**
   - Freezes encoder (already pre-trained)
   - Trains only decoder
   - Saves checkpoints each epoch
   - ~5-10 GPU hours for 5 epochs on 5000 samples

5. **Embedding Extraction:**
   - Function to extract embeddings for downstream tasks
   - Use frozen encoder on new data
   - Save as numpy arrays for CPU-based training

## Why NOT Wav2Vec2ForCTC?

- **Wav2Vec2ForCTC** = For speech recognition (predicting characters/phonemes)
- **Wav2Vec2Model** + custom decoder = For reconstruction (what you need)

The difference: CTC adds a classifier on top for sequence-to-sequence tasks. You need the raw embeddings + custom reconstruction decoder.