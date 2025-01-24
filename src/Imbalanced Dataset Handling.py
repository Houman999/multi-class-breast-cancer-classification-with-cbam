# Library
import numpy as np

def get_class_distribution(label):
    """
    Compute the distribution of classes in a dataset.

    Parameters:
    ----------
    label : list or numpy array
        A list or array containing class labels for each sample in the dataset.

    Returns:
    -------
    dict
        A dictionary where the keys are unique class labels and the values are the counts of each class.
    """
    unique, counts = np.unique(label, return_counts=True)
    return dict(zip(unique, counts))


def balance_dataset(image_paths, labels):
    """
    Balance a dataset by oversampling or undersampling class instances to achieve uniform class distribution.

    Parameters:
    ----------
    image_paths : list
        A list of file paths to images in the dataset.
    labels : list or numpy array
        A list or array containing class labels corresponding to the images.

    Returns:
    -------
    tuple
        - balanced_image_paths (list): A list of file paths to the balanced dataset images.
        - balanced_labels (list): A list of class labels corresponding to the balanced dataset.

    Notes:
    -----
    - If a class has fewer instances than the mean count of samples per class, it will be oversampled with replacement.
    - If a class has more instances than the mean count, it will be undersampled without replacement.
    """
    # Compute class distribution
    class_distribution = get_class_distribution(labels)
    mean_count = np.mean(list(class_distribution.values()))

    balanced_image_paths, balanced_labels = [], []

    for class_id, count in class_distribution.items():
        # Indices of all samples belonging to the current class
        class_indices = [i for i, label in enumerate(labels) if label == class_id]
        class_images = [image_paths[i] for i in class_indices]

        if count < mean_count:
            # Oversample if count is less than mean count
            oversampled_images = np.random.choice(class_images, size=int(mean_count - count), replace=True)
            balanced_image_paths.extend(class_images + list(oversampled_images))
            balanced_labels.extend([class_id] * len(class_images + list(oversampled_images)))
        else:
            # Undersample if count exceeds mean count
            undersampled_images = np.random.choice(class_images, size=int(mean_count), replace=False)
            balanced_image_paths.extend(list(undersampled_images))
            balanced_labels.extend([class_id] * len(undersampled_images))

    return balanced_image_paths, balanced_labels


balanced_image_paths, balanced_labels = balance_dataset(all_images, all_labels)
