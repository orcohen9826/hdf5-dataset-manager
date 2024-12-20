import h5py
import os
import librosa
import numpy as np
from Hd5_data_handler import create_or_update_hdf5_file , get_hdf5_summary , print_hdf5_summary ,get_1_feature_data_and_label_as_numpy


def add_folder_to_hdf5(hdf5_file_name, folder_path, dataset_name, feature_function, label=None, feature_name=None, description=None):
    """
    Add data from a folder to an HDF5 file using the create_or_update_hdf5_file function.

    This function processes files in a folder, applies a feature extraction function, and stores the resulting features
    in an HDF5 dataset using the create_or_update_hdf5_file function.

    Parameters:
    - hdf5_file_name (str): Path to the HDF5 file.
    - folder_path (str): Path to the folder containing the files.
    - dataset_name (str): Name of the dataset where data will be stored.
    - feature_function (function): A function that takes a file path as input and returns feature data as a NumPy array.
    - label (optional): Label to assign to all samples. If not provided, label will not be set.
    - feature_name (str): Name of the feature to store in the HDF5 file. Must be provided.
    - description (optional): Description of the dataset.

    Example Usage:
    ```python
    add_folder_to_hdf5(
        hdf5_file_name='audio_data.h5',
        folder_path='/path/to/audio/files',
        dataset_name='DataSet_A',
        feature_function=create_mel_spectrogram,
        label='drone',
        feature_name='mel_spectrogram',
        description='Dataset of drone audio spectrograms'
    )
    ```
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder '{folder_path}' does not exist.")

    # Add dataset description if provided
    create_or_update_hdf5_file(
        file_name=hdf5_file_name,
        dataset_name=dataset_name,
        description_dict={'description': description} if description else None
    )

    # Process files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue  # Skip non-file entries

        # Use the file name as the sample ID
        sample_id = os.path.splitext(file)[0]

        # Extract features using the provided function
        feature_data = feature_function(file_path)

        # Add or update the sample in the HDF5 file
        create_or_update_hdf5_file(
            file_name=hdf5_file_name,
            sample_id=sample_id,
            features_dict={feature_name: feature_data},
            label=label,
            dataset_name=dataset_name
        )

def create_mel_spectrogram(file_path, sample_rate=16000, duration=1.0):
    """
    Create a Mel spectrogram from an audio file.

    Parameters:
    - file_path (str): Path to the audio file.
    - sample_rate (int): Sampling rate for the audio file. Default is 16000.
    - duration (float): Duration of the audio to load in seconds. Default is 1.0.

    Returns:
    - np.ndarray: Mel spectrogram in decibels.

    Example Usage:
    ```python
    mel_spectrogram = create_mel_spectrogram('audio_file.wav')
    ```
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' does not exist.")
    if not file_path.endswith('.wav'):
        raise ValueError("Only WAV files are supported.")
    y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB







if __name__ == "__main__":


    folder_path = 'C:\\Users\\ASUS\\OneDrive\\Desktop\\Final_Project2024\\prototype1\\crnn_test\\data\\Real_Cut'
    hdf5_file = 'audio_data.h5'
    dataset_name = 'MLP_2022_real_data'

    add_folder_to_hdf5(
        hdf5_file_name=hdf5_file,
        folder_path=folder_path,
        dataset_name=dataset_name,
        feature_function=create_mel_spectrogram,
        label='drone',
        feature_name='mel_spectrogram',
        description='Dataset of.....'
    
    )



    print_hdf5_summary(get_hdf5_summary(hdf5_file))

    # Load data from the HDF5 file
    data , labels = get_1_feature_data_and_label_as_numpy(hdf5_file, feature_name='mel_spectrogram')
    print(data.shape)
    print(labels.shape)




















