

'''
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

'''
import h5py
import os
import librosa
import numpy as np



def create_or_update_hdf5_file(file_name, sample_id=None, features_dict=None, label=None, dataset_name=None, description_dict=None):
    """
    Create or update an HDF5 file with audio sample data.

    This function updates an existing HDF5 file or creates a new one if it does not exist. 
    It manages datasets and samples in a hierarchical structure. If the sample already exists, 
    its features and label will be updated. If the sample is new, the dataset name must be provided 
    to create a new group for it. Additionally, a dataset description can be set or updated for the specific dataset.

    Parameters:
    - file_name (str): Path to the HDF5 file. If the file does not exist, it will be created.
    - sample_id (optional): Unique ID for the sample. This must be unique across the entire file.
    - features_dict (optional): Dictionary of features to be added or updated (e.g., {'mel_spectrogram': matrix}).
        Each key represents a feature name, and the corresponding value is the data (typically a NumPy array).
    - label (optional): Label for the sample. If provided, it will update or set the label attribute for the sample.
        If not provided for a new sample, the label will be set to NaN.
    - dataset_name (optional): Name of the dataset where the sample should be stored. Required for new samples.
    - description_dict (optional): Dictionary containing the description for the specific dataset. If provided,
        it will update or set the description attribute of the specified dataset.

    Behavior:
    - If dataset_name is provided without sample_id or description_dict, the dataset will be created if it doesn't exist.
    - If description_dict is provided:
        - The description for the specified dataset will be updated or set.
    - If the sample ID already exists in the file:
        - The existing label will be updated if a new label is provided.
        - The features will be updated or added:
            - If a feature already exists, it will be replaced with the new data.
    - If the sample ID does not exist:
        - The dataset name must be provided.
        - A new group will be created for the dataset if it does not exist.
        - A new group for the sample will be created under the dataset.
        - The label will be set to the provided value or to NaN if not provided.

    Exceptions:
    - Raises a ValueError if the sample ID is new and the dataset name is not provided.

    Example Usage:
    ```python
    create_or_update_hdf5_file('audio_data.h5', dataset_name='DataSet_A')
    create_or_update_hdf5_file('audio_data.h5', dataset_name='DataSet_A', description_dict={'description': 'Updated Dataset A'})
    create_or_update_hdf5_file('audio_data.h5', 'sample_001', {'mel_spectrogram': some_matrix}, label='drone', dataset_name='DataSet_A')
    ```
    """
    with h5py.File(file_name, 'a') as hdf5_file:
        # Ensure the dataset exists if only dataset_name is provided
        if dataset_name and sample_id is None and description_dict is None:
            if dataset_name not in hdf5_file:
                hdf5_file.create_group(dataset_name)
            return

        # Update dataset description if description_dict is provided
        if description_dict is not None:
            if dataset_name in hdf5_file:
                dataset_group = hdf5_file[dataset_name]
                for key, value in description_dict.items():
                    dataset_group.attrs[key] = value
            else:
                # Create dataset if it doesn't exist and add description
                dataset_group = hdf5_file.create_group(dataset_name)
                for key, value in description_dict.items():
                    dataset_group.attrs[key] = value

        # If sample_id is not provided, stop here (only updating dataset)
        if sample_id is None:
            #TODO: Add logging or warning message if other parameters are provided without sample_id
            if label is not None or features_dict is not None:
                raise ValueError("Sample ID must be provided to add features or label.")           
            return


        # Check if dataset_name is provided for efficient lookup
        if dataset_name and dataset_name in hdf5_file:
            dataset_group = hdf5_file[dataset_name]
            if sample_id in dataset_group:
                sample_group = dataset_group[sample_id]
            else:
                #validate that sample_id is unique across all datasets
                for ds_name in hdf5_file.keys():
                    if sample_id in hdf5_file[ds_name]:
                        if ds_name != dataset_name:
                            raise ValueError(f"Sample ID '{sample_id}' already exists in another dataset '{ds_name}'.")
                # Create new sample if not found
                sample_group = dataset_group.create_group(sample_id)
                sample_group.attrs['label'] = label if label is not None else np.nan
                sample_group.create_group('features')
        else:
            # Check if the sample ID already exists in the file
            for ds_name in hdf5_file.keys():
                if sample_id in hdf5_file[ds_name]:
                    sample_group = hdf5_file[ds_name][sample_id]
                    break
            else:
                # If the sample ID does not exist, ensure dataset_name is provided
                if dataset_name is None:
                    raise ValueError("Dataset name must be provided for a new sample.")
                # Create dataset if it does not exist
                if dataset_name not in hdf5_file:
                    dataset_group = hdf5_file.create_group(dataset_name)
                dataset_group = hdf5_file[dataset_name]
                # Create a new group for the sample
                sample_group = dataset_group.create_group(sample_id)
                sample_group.attrs['label'] = label if label is not None else np.nan
                sample_group.create_group('features')

        # Add or update features
        if features_dict:
            features_group = sample_group['features']
            for feature_name, feature_data in features_dict.items():
                if feature_name in features_group:
                    del features_group[feature_name]  # Remove the existing feature
                features_group.create_dataset(feature_name, data=feature_data)

        # Update the label if provided
        if label is not None:
            sample_group.attrs['label'] = label


def get_hdf5_summary(file_name, dataset_name=None, sample_id=None):
    """
    Retrieve a summary of the HDF5 file structure and contents.

    Parameters:
    - file_name (str): Path to the HDF5 file.
    - dataset_name (optional): Name of the dataset to get detailed information about.
    - sample_id (optional): ID of the sample to retrieve detailed information about.

    Returns:
    - dict: A dictionary containing general summary, dataset-specific, or sample-specific information.

    Example Usage:
    ```python
    summary = get_hdf5_summary('audio_data.h5')
    summary = get_hdf5_summary('audio_data.h5', dataset_name='DataSet_A')
    summary = get_hdf5_summary('audio_data.h5', sample_id='sample_001')
    ```
    """
    with h5py.File(file_name, 'r') as hdf5_file:
        # If only file_name is provided, return general summary with details for each dataset
        if dataset_name is None and sample_id is None:
            total_datasets = len(hdf5_file.keys())
            total_samples = 0
            unique_labels = set()
            unique_features = set()
            samples_per_label = {}
            samples_per_feature = {}
            datasets_info = {}

            for ds_name in hdf5_file.keys():
                dataset_group = hdf5_file[ds_name]
                num_samples = len(dataset_group.keys())
                dataset_description = dataset_group.attrs.get('description', None)
                dataset_labels = set()
                dataset_features = set()
                dataset_samples_per_label = {}
                dataset_samples_per_feature = {}

                for sample_id in dataset_group.keys():
                    sample_group = dataset_group[sample_id]
                    label = sample_group.attrs.get('label', None)
                    if label is not None:
                        unique_labels.add(label)
                        dataset_labels.add(label)
                        samples_per_label[label] = samples_per_label.get(label, 0) + 1
                        dataset_samples_per_label[label] = dataset_samples_per_label.get(label, 0) + 1
                    if 'features' in sample_group:
                        features_group = sample_group['features']
                        for feature_name in features_group.keys():
                            unique_features.add(feature_name)
                            dataset_features.add(feature_name)
                            samples_per_feature[feature_name] = samples_per_feature.get(feature_name, 0) + 1
                            dataset_samples_per_feature[feature_name] = dataset_samples_per_feature.get(feature_name, 0) + 1

                datasets_info[ds_name] = {
                    'description': dataset_description,
                    'num_samples': num_samples,
                    'unique_labels': len(dataset_labels),
                    'unique_features': len(dataset_features),
                    'samples_per_label': dataset_samples_per_label,
                    'samples_per_feature': dataset_samples_per_feature
                }
                total_samples += num_samples

            return {
                'general': {
                    'total_datasets': total_datasets,
                    'total_samples': total_samples,
                    'unique_labels': len(unique_labels),
                    'unique_features': len(unique_features),
                    'samples_per_label': samples_per_label,
                    'samples_per_feature': samples_per_feature
                },
                'datasets': datasets_info
            }

        # If dataset_name is provided, return dataset-specific summary
        if dataset_name is not None and sample_id is None:
            if dataset_name not in hdf5_file:
                raise ValueError(f"Dataset '{dataset_name}' does not exist.")

            dataset_group = hdf5_file[dataset_name]
            num_samples = len(dataset_group.keys())
            unique_labels = set()
            unique_features = set()
            samples_per_label = {}
            samples_per_feature = {}

            for sample_id in dataset_group.keys():
                sample_group = dataset_group[sample_id]
                label = sample_group.attrs.get('label', None)
                if label is not None:
                    unique_labels.add(label)
                    samples_per_label[label] = samples_per_label.get(label, 0) + 1
                if 'features' in sample_group:
                    features_group = sample_group['features']
                    for feature_name in features_group.keys():
                        unique_features.add(feature_name)
                        samples_per_feature[feature_name] = samples_per_feature.get(feature_name, 0) + 1

            return {
                'dataset': {
                    'description': dataset_group.attrs.get('description', None),
                    'num_samples': num_samples,
                    'unique_labels': len(unique_labels),
                    'unique_features': len(unique_features),
                    'samples_per_label': samples_per_label,
                    'samples_per_feature': samples_per_feature
                }
            }

        # If sample_id is provided, return sample-specific summary
        if sample_id is not None:
            for ds_name in hdf5_file.keys():
                dataset_group = hdf5_file[ds_name]
                if sample_id in dataset_group:
                    sample_group = dataset_group[sample_id]
                    label = sample_group.attrs.get('label', None)
                    num_features = len(sample_group['features'].keys()) if 'features' in sample_group else 0

                    return {
                        'sample': {
                            'label': label,
                            'num_features': num_features,
                            'dataset_name': ds_name
                        }
                    }

            raise ValueError(f"Sample ID '{sample_id}' does not exist in any dataset.")


def print_hdf5_summary(summary_dict):
    """
    Print the HDF5 summary dictionary in a human-readable format.

    Parameters:
    - summary_dict (dict): The dictionary returned by `get_hdf5_summary`.

    Example Usage:
    ```python
    summary = get_hdf5_summary('audio_data.h5')
    print_hdf5_summary(summary)
    ```
    """
    # Print general summary
    if 'general' in summary_dict:
        print("=== General Summary ===")
        general = summary_dict['general']
        print(f"Total Datasets: {general.get('total_datasets', 0)}")
        print(f"Total Samples: {general.get('total_samples', 0)}")
        print(f"Unique Labels: {general.get('unique_labels', 0)}")
        print(f"Unique Features: {general.get('unique_features', 0)}")
        print("Samples Per Label:")
        for label, count in general.get('samples_per_label', {}).items():
            print(f"  {label}: {count}")
        print("Samples Per Feature:")
        for feature, count in general.get('samples_per_feature', {}).items():
            print(f"  {feature}: {count}")
        print()

    # Print dataset-specific summaries
    if 'datasets' in summary_dict:
        print("=== Dataset Summaries ===")
        for ds_name, ds_info in summary_dict['datasets'].items():
            print(f"Dataset: {ds_name}")
            print(f"  Description: {ds_info.get('description', 'None')}")
            print(f"  Total Samples: {ds_info.get('num_samples', 0)}")
            print(f"  Unique Labels: {ds_info.get('unique_labels', 0)}")
            print(f"  Unique Features: {ds_info.get('unique_features', 0)}")
            print("  Samples Per Label:")
            for label, count in ds_info.get('samples_per_label', {}).items():
                print(f"    {label}: {count}")
            print("  Samples Per Feature:")
            for feature, count in ds_info.get('samples_per_feature', {}).items():
                print(f"    {feature}: {count}")
            print()

    # Print sample-specific summary
    if 'sample' in summary_dict:
        print("=== Sample Summary ===")
        sample = summary_dict['sample']
        print(f"Sample belongs to Dataset: {sample.get('dataset_name', 'Unknown')}")
        print(f"Label: {sample.get('label', 'None')}")
        print(f"Number of Features: {sample.get('num_features', 0)}")
        print()


def delete_from_hdf5(file_name, dataset_name=None, sample_id=None, feature_name=None):
    """
    Delete data from an HDF5 file.

    Parameters:
    - file_name (str): Path to the HDF5 file.
    - dataset_name (optional): Name of the dataset to delete from.
    - sample_id (optional): ID of the sample to delete from.
    - feature_name (optional): Name of the feature to delete.

    Behavior:
    - Deletes a feature if feature_name is provided.
    - Deletes a sample if sample_id is provided (and feature_name is None).
    - Deletes all occurrences of a feature across the dataset or the entire file if only feature_name is provided.
    - Deletes a dataset if only dataset_name is provided.
    - Prompts user confirmation for deletions affecting entire datasets or multiple samples.

    Example Usage:
    ```python
    delete_from_hdf5('audio_data.h5', dataset_name='DataSet_A', sample_id='sample_001', feature_name='mel_spectrogram')
    delete_from_hdf5('audio_data.h5', dataset_name='DataSet_A', feature_name='mel_spectrogram')
    delete_from_hdf5('audio_data.h5', feature_name='mel_spectrogram')
    ```
    """
    with h5py.File(file_name, 'a') as hdf5_file:
        # Delete feature across the entire file
        if feature_name and not dataset_name and not sample_id:
            print(f"Feature '{feature_name}' will be deleted from all datasets and samples.")
            confirmation = input("Are you sure? (Y/N): ")
            if confirmation.upper() != 'Y':
                print("Operation cancelled.")
                return

            for ds_name in hdf5_file.keys():
                dataset_group = hdf5_file[ds_name]
                for sample_id in dataset_group.keys():
                    sample_group = dataset_group[sample_id]
                    if 'features' in sample_group and feature_name in sample_group['features']:
                        del sample_group['features'][feature_name]
            print(f"Feature '{feature_name}' deleted from all datasets.")
            return

        # Delete feature from a specific dataset
        if feature_name and dataset_name and not sample_id:
            if dataset_name not in hdf5_file:
                raise ValueError(f"Dataset '{dataset_name}' does not exist.")

            print(f"Feature '{feature_name}' will be deleted from all samples in dataset '{dataset_name}'.")
            confirmation = input("Are you sure? (Y/N): ")
            if confirmation.upper() != 'Y':
                print("Operation cancelled.")
                return

            dataset_group = hdf5_file[dataset_name]
            for sample_id in dataset_group.keys():
                sample_group = dataset_group[sample_id]
                if 'features' in sample_group and feature_name in sample_group['features']:
                    del sample_group['features'][feature_name]
            print(f"Feature '{feature_name}' deleted from dataset '{dataset_name}'.")
            return

        # Delete feature from a specific sample
        if feature_name and sample_id:
            for ds_name in hdf5_file.keys():
                dataset_group = hdf5_file[ds_name]
                if sample_id in dataset_group:
                    sample_group = dataset_group[sample_id]
                    if 'features' in sample_group and feature_name in sample_group['features']:
                        del sample_group['features'][feature_name]
                        print(f"Feature '{feature_name}' deleted from sample '{sample_id}' in dataset '{ds_name}'.")
                        return
            raise ValueError(f"Sample ID '{sample_id}' or feature '{feature_name}' does not exist.")

        # Delete a specific sample
        if sample_id:
            for ds_name in hdf5_file.keys():
                dataset_group = hdf5_file[ds_name]
                if sample_id in dataset_group:
                    del dataset_group[sample_id]
                    print(f"Sample '{sample_id}' deleted from dataset '{ds_name}'.")
                    return
            raise ValueError(f"Sample ID '{sample_id}' does not exist.")

        # Delete an entire dataset
        if dataset_name:
            if dataset_name not in hdf5_file:
                raise ValueError(f"Dataset '{dataset_name}' does not exist.")

            print(f"Dataset '{dataset_name}' will be deleted.")
            confirmation = input("Are you sure? (Y/N): ")
            if confirmation.upper() != 'Y':
                print("Operation cancelled.")
                return

            del hdf5_file[dataset_name]
            print(f"Dataset '{dataset_name}' deleted.")
            return


def delete_by_label(file_name, label, dataset_name=None):
    """
    Delete all samples with the specified label from a dataset or the entire HDF5 file.

    Parameters:
    - file_name (str): Path to the HDF5 file.
    - label: Label of the samples to delete.
    - dataset_name (optional): Name of the dataset to delete from. If not provided, deletes from all datasets.

    Behavior:
    - Deletes samples with the specified label from the given dataset.
    - If dataset_name is not provided, prompts for confirmation and deletes from all datasets.

    Example Usage:
    ```python
    delete_by_label('audio_data.h5', label='drone', dataset_name='DataSet_A')
    delete_by_label('audio_data.h5', label='noise')
    ```
    """
    with h5py.File(file_name, 'a') as hdf5_file:
        total_deleted = 0

        if dataset_name:
            # Check if the dataset exists
            if dataset_name not in hdf5_file:
                raise ValueError(f"Dataset '{dataset_name}' does not exist.")

            dataset_group = hdf5_file[dataset_name]
            to_delete = [sample_id for sample_id in dataset_group.keys() if dataset_group[sample_id].attrs.get('label') == label]

            for sample_id in to_delete:
                del dataset_group[sample_id]
                total_deleted += 1

            print(f"Deleted {total_deleted} samples with label '{label}' from dataset '{dataset_name}'.")

        else:
            # Delete from all datasets
            to_delete_per_dataset = {}
            for ds_name in hdf5_file.keys():
                dataset_group = hdf5_file[ds_name]
                to_delete = [sample_id for sample_id in dataset_group.keys() if dataset_group[sample_id].attrs.get('label') == label]
                if to_delete:
                    to_delete_per_dataset[ds_name] = to_delete

            # Print summary and confirm
            if not to_delete_per_dataset:
                print(f"No samples with label '{label}' found in the file.")
                return

            print("The following samples will be deleted:")
            for ds_name, samples in to_delete_per_dataset.items():
                print(f"Dataset '{ds_name}': {len(samples)} samples")

            confirmation = input("Are you sure you want to delete these samples? (Y/N): ")
            if confirmation.upper() != 'Y':
                print("Operation cancelled.")
                return

            # Perform deletion
            for ds_name, samples in to_delete_per_dataset.items():
                dataset_group = hdf5_file[ds_name]
                for sample_id in samples:
                    del dataset_group[sample_id]
                    total_deleted += 1

            print(f"Deleted {total_deleted} samples with label '{label}' from the entire file.")


def copy_dataset_to_new_file(source_file_name, dataset_name, target_file_name):
    """
    Copy a dataset from one HDF5 file to another.

    Parameters:
    - source_file_name (str): Path to the source HDF5 file.
    - dataset_name (str): Name of the dataset to copy.
    - target_file_name (str): Path to the target HDF5 file.

    Behavior:
    - Copies the specified dataset and all its contents (samples, features, and attributes) to the target file.

    Example Usage:
    ```python
    copy_dataset_to_new_file('source.h5', 'DataSet_A', 'target.h5')
    ```
    """
    with h5py.File(source_file_name, 'r') as source_file:
        if dataset_name not in source_file:
            raise ValueError(f"Dataset '{dataset_name}' does not exist in the source file.")

        with h5py.File(target_file_name, 'a') as target_file:
            if dataset_name in target_file:
                raise ValueError(f"Dataset '{dataset_name}' already exists in the target file.")

            source_dataset = source_file[dataset_name]
            target_dataset = target_file.create_group(dataset_name)

            # Copy attributes of the dataset
            for attr_name, attr_value in source_dataset.attrs.items():
                target_dataset.attrs[attr_name] = attr_value

            # Copy samples and their features
            for sample_id in source_dataset.keys():
                source_sample = source_dataset[sample_id]
                target_sample = target_dataset.create_group(sample_id)

                # Copy attributes of the sample
                for attr_name, attr_value in source_sample.attrs.items():
                    target_sample.attrs[attr_name] = attr_value

                # Copy features
                if 'features' in source_sample:
                    source_features = source_sample['features']
                    target_features = target_sample.create_group('features')
                    for feature_name in source_features.keys():
                        target_features.create_dataset(feature_name, data=source_features[feature_name][...])

    print(f"Dataset '{dataset_name}' copied from '{source_file_name}' to '{target_file_name}'.")


# [{'mel_spectrogram': array([...]), 'FFT': array([...])}, {'mel_spectrogram': array([...])}, ...] and [label1, label2, ...]
def get_features_and_labels(file_name, dataset_name=None, label_list=None, feature_list=None):
    """
    Retrieve features and labels from an HDF5 file.

    Parameters:
    - file_name (str): Path to the HDF5 file.
    - dataset_name (optional): Name of the dataset to retrieve data from. If not provided, retrieves from all datasets.
    - label_list (optional): List of labels to filter samples. If not provided, retrieves all labels.
    - feature_list (optional): List of features to retrieve. If not provided, retrieves all features.

    Returns:
    - tuple: A tuple containing two elements:
        1. List of dictionaries with features for each sample: [{'feature_name': array([...]), ...}, ...]
        2. List of corresponding labels: [label1, label2, ...]

    Example Usage:
    ```python
    features, labels = get_features_and_labels('audio_data.h5', dataset_name='DataSet_A', label_list=['drone'], feature_list=['mel_spectrogram'])
    ```
    """
    features = []
    labels = []

    with h5py.File(file_name, 'r') as hdf5_file:
        datasets = [dataset_name] if dataset_name else hdf5_file.keys()

        for ds_name in datasets:
            if ds_name not in hdf5_file:
                raise ValueError(f"Dataset '{ds_name}' does not exist in the file.")

            dataset_group = hdf5_file[ds_name]
            for sample_id in dataset_group.keys():
                sample_group = dataset_group[sample_id]
                label = sample_group.attrs.get('label', None)

                if label_list and label not in label_list:
                    continue

                sample_features = {}
                if 'features' in sample_group:
                    features_group = sample_group['features']
                    for feature_name in features_group.keys():
                        if feature_list and feature_name not in feature_list:
                            continue
                        sample_features[feature_name] = features_group[feature_name][...]

                if sample_features:
                    features.append(sample_features)
                    labels.append(label)

    if not features:
        raise ValueError("No samples found matching the specified criteria.")

    return features, labels


#array([array([...]), array([...]), array([...]), ...]) and array([label1, label2, label3, ...])
def get_1_feature_data_and_label_as_numpy(file_name, dataset_name=None, label_list=None, feature_name=None, binary_key=None):
    """
    Retrieve a single feature and corresponding labels as NumPy arrays from an HDF5 file.

    Parameters:
    - file_name (str): Path to the HDF5 file.
    - dataset_name (optional): Name of the dataset to retrieve data from. If not provided, retrieves from all datasets.
    - label_list (optional): List of labels to filter samples. If not provided, retrieves all labels.
    - feature_name (str): Name of the feature to retrieve. Must be provided.
    - binary_key (optional): A label to convert matching labels to 1 and others in label_list to 0.

    Returns:
    - tuple: A tuple containing two elements:
        1. NumPy array of the feature data: array([array([...]), array([...]), ...])
        2. NumPy array of labels: array([label1, label2, ...])

    Behavior:
    - Skips samples where the feature or label is missing.
    - If binary_key is provided, converts labels to 1 for matches and 0 for others in label_list.

    Example Usage:
    ```python
    X, y = get_1_feature_data_and_label_as_numpy(
        'audio_data.h5',
        dataset_name='DataSet_A',
        label_list=['drone', 'noise'],
        feature_name='mel_spectrogram',
        binary_key='drone'
    )
    ```
    """
    if not feature_name:
        raise ValueError("Feature name must be provided.")

    features, labels = get_features_and_labels(
        file_name=file_name,
        dataset_name=dataset_name,
        label_list=label_list,
        feature_list=[feature_name]
    )

    X = []
    y = []

    for feature_dict, label in zip(features, labels):
        if feature_name not in feature_dict or label is None:
            continue

        feature_data = feature_dict[feature_name]
        if binary_key is not None:
            if label == binary_key:
                y.append(1)
            elif label in label_list:
                y.append(0)
            else:
                continue
        else:
            y.append(label)

        X.append(feature_data)

    if not X:
        raise ValueError("No valid samples found matching the criteria.")

    return np.array(X), np.array(y)



















# # Calls to the first function: create_or_update_hdf5_file

# # 1. Update only a dataset
# create_or_update_hdf5_file(
#     file_name='data.h5',
#     dataset_name='DataSet_A'
# )
# print_hdf5_summary(get_hdf5_summary('data.h5'))
# # 2. Update dataset description
# create_or_update_hdf5_file(
#     file_name='data.h5',
#     dataset_name='DataSet_A',
#     description_dict={'description': 'Updated description for DataSet_A'}
# )
# print_hdf5_summary(get_hdf5_summary('data.h5'))


# # 3. Add a sample with a label to the dataset
# create_or_update_hdf5_file(
#     file_name='data.h5',
#     sample_id='sample_001',
#     features_dict={'mel_spectrogram': np.random.rand(128, 128)},
#     label='drone',
#     dataset_name='DataSet_A'
# )
# print_hdf5_summary(get_hdf5_summary('data.h5'))

# # 3. Add a sample with a label to the dataset
# create_or_update_hdf5_file(
#     file_name='data.h5',
#     sample_id='sample_002',
    
#     label='drone',
#     dataset_name='DataSet_B'
# )
# print_hdf5_summary(get_hdf5_summary('data.h5'))

# # 4. Add a feature to an existing sample
# create_or_update_hdf5_file(
#     file_name='data.h5',
#     sample_id='sample_002',
#     features_dict={'FFT': np.random.rand(64, 64)}
# )
# print_hdf5_summary(get_hdf5_summary('data.h5'))
# # 5. Add a new sample with a new feature and label to a new dataset
# create_or_update_hdf5_file(
#     file_name='data.h5',
#     sample_id='sample_004',
#     features_dict={'MFCC': np.random.rand(32, 32)},
#     label='background',
#     dataset_name='DataSet_B',
#     description_dict={'description': 'New Dataset for background noise'}
# )
# print_hdf5_summary(get_hdf5_summary('data.h5'))


# create_or_update_hdf5_file(
#     file_name='data.h5',
#     sample_id='sample_005',
#     features_dict={'MFCC': np.random.rand(32, 32)},
#     label='background',
#     dataset_name='DataSet_B',
#     description_dict={'description': 'New Dataset for background noise'}
# )
# print_hdf5_summary(get_hdf5_summary('data.h5' , sample_id='sample_002'))


# create_or_update_hdf5_file(
#     file_name='data.h5',
#     sample_id='sample_006',

#     label='drone',
#     dataset_name='DataSet_B',
#     description_dict={'description': 'New Dataset for background noise'}
# )
# print_hdf5_summary(get_hdf5_summary('data.h5' , sample_id='sample_002'))
# # Calls to the second function: get_hdf5_summary


# print_hdf5_summary(get_hdf5_summary('data.h5'))
# delete_by_label('data.h5', label='drone', dataset_name='DataSet_B')
# print_hdf5_summary(get_hdf5_summary('data.h5'))

# delete_from_hdf5('data.h5', feature_name='FFT')
# print_hdf5_summary(get_hdf5_summary('data.h5'))

# delete_from_hdf5('data.h5', dataset_name='DataSet_B')
# print_hdf5_summary(get_hdf5_summary('data.h5'))









