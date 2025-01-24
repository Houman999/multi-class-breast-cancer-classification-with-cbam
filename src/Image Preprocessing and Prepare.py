#Library 
import cv2 
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split

# CLAHE for Color Images
def apply_clahe_to_color_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)
    lab = cv2.merge((cl, ca, cb))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Preprocessing
def preprocess_image(image_path, target_size, augment=False, apply_clahe=False):
    image = cv2.imread(image_path.decode('utf-8'))
    if apply_clahe:
        image = apply_clahe_to_color_image(image)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    return image

# Preprocessing Wrapper
def preprocess_image_wrapper(image_path, label, target_size, augment, apply_clahe):
    def _preprocess(path):
        return preprocess_image(path, target_size, augment, apply_clahe)
    image = tf.numpy_function(_preprocess, inp=[image_path], Tout=tf.float32)
    image.set_shape((*target_size, 3))
    return image, label

# Create Dataset
def create_dataset(image_paths, labels, batch_size=32, target_size=(224, 224), augment=False, apply_clahe=False):
    labels = to_categorical(labels, num_classes=len(set(labels)))
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        lambda x, y: preprocess_image_wrapper(x, y, target_size, augment, apply_clahe),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Main - Load Dataset
multiclass_data, all_images, all_labels, label_map = read_multiclass_data(base_path)

# Balance Dataset
balanced_paths, balanced_labels = balance_dataset(all_images, all_labels)

# Split Data
train_paths, temp_paths, train_labels, temp_labels = train_test_split(balanced_paths, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Create Datasets
train_dataset = create_dataset(train_paths, train_labels, augment=True, apply_clahe=True)
val_dataset = create_dataset(val_paths, val_labels, apply_clahe=True)
test_dataset = create_dataset(test_paths, test_labels, apply_clahe=True)