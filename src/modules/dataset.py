from torch.utils.data import Dataset as torch_Dataset
import torch
from pathlib import Path
from typing import Dict, List, Literal, Union, Tuple, Optional, Any
import pickle
from datasets import load_dataset as hf_load_dataset, Dataset as hf_Dataset
import warnings
if __name__ == "__main__":
    import paths
else:
    from modules import paths

IMAGES_PER_CLASS = {"validation":50,
                    "test":50,
                    "train":500
                    }

class TorchDatasetWrapper(torch_Dataset):
    def __init__(self, hf_dataset: Any, transform: Optional[torch.nn.Module] = None) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __repr__(self) -> str:
        return str(self.hf_dataset)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        example = self.hf_dataset[idx]
        image = example['image']
        if self.transform:
            image = self.transform(image)
        return image, example['label']
    

def gen_super_tiny(
    dataset: List[Dict[str, List[Union[str, int]]]],
    split: Literal["valid", "test", "train"],
    stop: int = 10,
    start: int = 0,
    classes: int = 200
) -> torch_Dataset:
    """
    Generates a super tiny version of a dataset by sampling images at regular intervals.
    
    This function creates a smaller subset of a larger dataset by selecting images
    from the original dataset with a fixed step size. The resulting dataset maintains
    the original structure but contains fewer samples, making it useful for quick
    testing or prototyping.
    
    Args:
        dataset: The original dataset as a list of dictionaries, where each dictionary
            contains "image" and "label" keys with lists of values.
        split: The dataset split to generate ("validation", "test", or "train").
            Determines how many images are sampled per class.
        stop: The ending index for each sampling window (exclusive).
        start: The starting index for each sampling window (inclusive).
        step: The interval between sampling windows.
        classes: The number of classes to include in the subset.
            
    Returns:
        A Hugging Face Dataset object containing the sampled images and labels.
        
    Example:
        >>> tiny_dataset = gen_super_tiny(full_dataset, "train", stop=10, step=500)
        >>> print(len(tiny_dataset))
        2000  # For train split with default classes=200
    """
    images_per_class = {
        "valid": 50,
        "test": 50,
        "train": 500
    }
    if start > stop: raise ValueError("start[{start}] > stop[{stop}]")
    if stop > images_per_class[split]: warnings.warn(f"Watch out you have an imbalanced dataset because stop[{stop}] > images_per_class[split]{images_per_class[split]}")
    dataset_length = images_per_class[split] * classes
    t = [dataset[start+i:stop+i] for i in range(0, dataset_length-stop, images_per_class[split])]
    all_images = [image for image_class_dict in t for image in image_class_dict["image"]]
    all_labels = [image for image_class_dict in t for image in image_class_dict["label"]]
    return hf_Dataset.from_dict({"image": all_images, "label": all_labels})


def make_animal_dataset(split: Literal["train", "validation", "test"]):
    


def load(
    split: Literal["train", "validation", "test"],
    tiny: bool = False,
    animal: bool = False,
    **tiny_kwargs
) -> hf_Dataset:
    """
    Loads the Tiny ImageNet dataset either from a local pickle cache or from Hugging Face.
    
    This function first checks for a locally cached version of the dataset. If not found,
    it downloads the dataset from Hugging Face, caches it as a pickle file, and returns it.
    Supports both the full dataset and a tiny subset version.
    
    Args:
        split: The dataset split to load ("train", "validation", or "test").
        tiny: If True, returns a smaller subset of the dataset for quick testing.
              If False, returns the full dataset.
        tiny_kwargs: Optional arguments to pass to gen_super_tiny when tiny=True.
            Possible arguments: stop, start, classes.
              
    Returns:
        A Hugging Face Dataset or DatasetDict object containing the requested split.
        Returns a tiny subset if tiny=True.
        
    Raises:
        FileNotFoundError: If the pickle file doesn't exist and Hugging Face download fails.
        pickle.PickleError: If there's an error loading the cached pickle file.
        
    Example:
        >>> train_data = load("train")
        >>> val_data = load("validation", tiny=True, stop=5)
        >>> print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    """
    pickle_path: Path = paths.DATA / (split + ".pkl")

    if pickle_path.is_file():
        print("Loading dataset from", pickle_path)
        dataset: hf_Dataset = pickle.load(open(pickle_path, "rb"))
    else:
        print("Downloading dataset from 'Maysee/tiny-imagenet'")
        dataset: hf_Dataset = hf_load_dataset('Maysee/tiny-imagenet', split=split)
        pickle.dump(dataset, open(pickle_path, "wb"))
        print("Dataset saved in", pickle_path)

    if tiny:
        # Pass the tiny_kwargs to gen_super_tiny
        dataset = gen_super_tiny(dataset, split, **tiny_kwargs)
        
    return dataset