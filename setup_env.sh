echo "Creating and activating conda environment 'EQ_AUD' with Python 3.12..."
conda init
conda create -n EQ_AUD python==3.12 -y
conda activate EQ_AUD

echo "Installing required packages..."
pip install pandas numpy torchaudio torch transformers h5py scikit-learn tqdm seisbench