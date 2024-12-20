# hdf5-dataset-manager

## Overview
This module provides robust tools for managing and processing datasets in HDF5 files, enabling efficient management of multi-dataset repositories. It supports creating, updating, summarizing, and managing hierarchical dataset structures tailored for machine learning and data analysis workflows.

---

## HDF5 File Structure
The module organizes the data in the following structure:

```
/ (Root)
├── DataSet_A
│   ├── attributes:
│   │   ├── description: "Dataset A description"
│   │   ├── created_at: "2024-12-20"
│   │   └── source: "Source information"
│   ├── sample_001
│   │   ├── attributes:
│   │   │   ├── label: 0
│   │   │   └── other_attributes: ...
│   │   └── features
│   │       ├── mel_spectrogram: array([...])
│   │       └── FFT: array([...])
│   ├── sample_002
│   │   ├── attributes:
│   │   │   ├── label: 1
│   │   │   └── other_attributes: ...
│   │   └── features
│   │       ├── mel_spectrogram: array([...])
│   │       └── FFT: array([...])
├── DataSet_B
│   ├── attributes:
│   │   ├── description: "Dataset B description"
│   │   ├── created_at: "2024-12-21"
│   │   └── source: "Source information"
│   ├── sample_003
│   │   ├── attributes:
│   │   │   ├── label: 0
│   │   │   └── other_attributes: ...
│   │   └── features
│   │       ├── mel_spectrogram: array([...])
│   │       └── FFT: array([...])
│   ├── sample_004
│   │   ├── attributes:
│   │   │   ├── label: 1
│   │   │   └── other_attributes: ...
│   │   └── features
│   │       ├── mel_spectrogram: array([...])
│   │       └── FFT: array([...])
```

---

## Functions

### 1. `create_or_update_hdf5_file`
This function creates or updates HDF5 files with hierarchical dataset structures.

**Parameters:**
- `file_name` (str): Path to the HDF5 file.
- `sample_id` (optional): Unique identifier for the sample.
- `features_dict` (optional): Dictionary of features to add or update.
- `label` (optional): Label for the sample.
- `dataset_name` (optional): Name of the dataset where the sample resides.
- `description_dict` (optional): Dictionary for dataset-level descriptions.

**Behavior:**
- Creates datasets if they don't exist.
- Updates existing labels and features for samples.

**Example Usage:**
```python
create_or_update_hdf5_file('audio_data.h5', dataset_name='DataSet_A')
create_or_update_hdf5_file('audio_data.h5', 'sample_001', {'mel_spectrogram': some_matrix}, label='drone', dataset_name='DataSet_A')
```

---

### 2. `get_hdf5_summary`
Retrieves a structured summary of the HDF5 file.

**Parameters:**
- `file_name` (str): Path to the HDF5 file.
- `dataset_name` (optional): Specific dataset to summarize.
- `sample_id` (optional): Specific sample to summarize.

**Returns:**
- A dictionary summarizing general information, dataset details, or sample details.

**Example Usage:**
```python
summary = get_hdf5_summary('audio_data.h5')
summary = get_hdf5_summary('audio_data.h5', dataset_name='DataSet_A')
summary = get_hdf5_summary('audio_data.h5', sample_id='sample_001')
```

---

### 3. `print_hdf5_summary`
Prints the summary returned by `get_hdf5_summary` in a human-readable format.

**Parameters:**
- `summary_dict` (dict): Summary dictionary returned by `get_hdf5_summary`.

**Example Usage:**
```python
summary = get_hdf5_summary('audio_data.h5')
print_hdf5_summary(summary)
```

---

### 4. `delete_from_hdf5`
Deletes data from an HDF5 file based on the specified parameters.

**Parameters:**
- `file_name` (str): Path to the HDF5 file.
- `dataset_name` (optional): Name of the dataset to delete from.
- `sample_id` (optional): Sample ID to delete.
- `feature_name` (optional): Feature name to delete.

**Behavior:**
- Deletes features, samples, or entire datasets with user confirmation for larger operations.

**Example Usage:**
```python
delete_from_hdf5('audio_data.h5', dataset_name='DataSet_A', feature_name='mel_spectrogram')
delete_from_hdf5('audio_data.h5', feature_name='FFT')
```

---

### 5. `delete_by_label`
Deletes all samples with a specific label from a dataset or the entire file.

**Parameters:**
- `file_name` (str): Path to the HDF5 file.
- `label`: Label of the samples to delete.
- `dataset_name` (optional): Dataset to delete from.

**Example Usage:**
```python
delete_by_label('audio_data.h5', label='drone', dataset_name='DataSet_A')
delete_by_label('audio_data.h5', label='noise')
```

---

### 6. `copy_dataset_to_new_file`
Copies an entire dataset from one HDF5 file to another.

**Parameters:**
- `source_file_name` (str): Path to the source file.
- `dataset_name` (str): Name of the dataset to copy.
- `target_file_name` (str): Path to the target file.

**Example Usage:**
```python
copy_dataset_to_new_file('source.h5', 'DataSet_A', 'target.h5')
```

---

### 7. `get_features_and_labels`
Retrieves features and labels from an HDF5 file.

**Parameters:**
- `file_name` (str): Path to the HDF5 file.
- `dataset_name` (optional): Dataset to retrieve from.
- `label_list` (optional): Filter for specific labels.
- `feature_list` (optional): List of features to retrieve.

**Returns:**
- A tuple with features and labels.

**Example Usage:**
```python
features, labels = get_features_and_labels('audio_data.h5', dataset_name='DataSet_A', label_list=['drone'], feature_list=['mel_spectrogram'])
```

---

### 8. `get_1_feature_data_and_label_as_numpy`
Retrieves a single feature and corresponding labels as NumPy arrays.

**Parameters:**
- `file_name` (str): Path to the HDF5 file.
- `dataset_name` (optional): Dataset to retrieve from.
- `label_list` (optional): Filter for specific labels.
- `feature_name` (str): Feature to retrieve.
- `binary_key` (optional): Converts labels to binary based on the key.

**Returns:**
- A tuple with NumPy arrays for features and labels.

**Example Usage:**
```python
X, y = get_1_feature_data_and_label_as_numpy(
    'audio_data.h5',
    dataset_name='DataSet_A',
    label_list=['drone', 'noise'],
    feature_name='mel_spectrogram',
    binary_key='drone'
)
```

---

## Dependencies
- `h5py`
- `numpy`
- `librosa`
- `os`

---

Feel free to modify or extend this module for your specific use case!

