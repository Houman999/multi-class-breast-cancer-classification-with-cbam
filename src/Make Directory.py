# Library 
import os 

# Func to prepare directory
def read_multiclass_data(base_path):
    """
    Reads a multiclass dataset and organizes it by zoom level and class.

    Args:
        base_path (str): Path to the dataset directory.

    Returns:
        tuple: (data dict, all image paths, all labels, label map)
    """
    data = {}
    label_map = {}
    all_images , all_labels = [] , []

    for zoom_level in os.listdir(base_path):
        zoom_path = os.path.join(base_path , zoom_level)
        if os.path.isdir(zoom_level):
            data[zoom_path] = {}
            for class_index , class_name in enumerate(os.listdir(zoom_path)):
                class_path = os.path.join(zoom_path , class_name)
                if os.path.isdir(class_path):
                    label_map[class_path] = class_index
                    image_path = [
                        os.path.join(class_path , f)
                        for f in os.listdir(class_path)
                        if f.endswith('jpg' , 'png' , "jpeg")
                    ]
                    data[zoom_path][class_name] = image_path
                    all_images.extend(image_path)
                    all_labels.extend([class_index] * len(image_path))
                    
    return data , label_map , all_images , all_labels

data , label_map , all_images , all_labels = read_multiclass_data(base_path)
