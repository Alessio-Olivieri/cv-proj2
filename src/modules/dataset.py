from torch.utils.data import Dataset as torch_Dataset
import torch
from torch.utils.data import default_collate
from pathlib import Path
from typing import Dict, List, Literal, Union, Tuple, Optional, Any, Set, Callable
import pickle
from datasets import load_dataset as hf_load_dataset, Dataset as hf_Dataset
import warnings
import random

if __name__ == "__main__":
    import paths
else:
    from modules import paths

IMAGES_PER_CLASS = {"validation":50,
                    "test":50,
                    "train":500
                    }

class TorchDatasetWrapper(torch_Dataset):
    def __init__(self, hf_dataset_v: hf_Dataset, transform: Optional[torch.nn.Module] = None) -> None:
        self.hf_dataset = hf_dataset_v
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
) -> hf_Dataset:
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


def load(
    split: Literal["train", "validation", "test"],
    tiny: bool = False,
    transform: Callable = None,
    **tiny_kwargs
) -> torch_Dataset:
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
    pickle_path: Path = paths.data / (split + ".pkl")

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


def load_animal_dataset(split: Literal["train", "validation", "test"], transform:Callable, tiny=False, **tiny_kwargs) -> Tuple[hf_Dataset, Dict]:
    coarse_labels_original = {
    "Aquatic": {0, 15, 16, 20, 40},
    "Amphibians & Reptiles": {1, 2, 3, 4, 5},
    "Arthropods": {7, 8, 9, 10, 32, 33, 34, 35, 36, 37, 38, 39, 184},
    "Birds": {17, 18, 19, 170},
    "Mammals": {11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 187},
    "Marine Life & Fossils": {6, 12, 13, 14, 190}
    }
    def map_labels(example):
        example["label"] = new_animal_labels_map[example["label"]]
        return example
    animal_labels = list(coarse_labels_original.values())
    animal_labels = animal_labels[0].union(*animal_labels[1:])
    new_animal_labels_map = {label:i for i,label in enumerate(animal_labels)}
    coarse_labels = {label:{new_animal_labels_map[cl] for cl in cl_old_set} for label, cl_old_set in coarse_labels_original.items()}

    
    pickle_path: Path = paths.data / ("animal_"+ split + ".pkl")

    if pickle_path.is_file():
        print("Loading animal dataset from", pickle_path)
        dataset: hf_Dataset = pickle.load(open(pickle_path, "rb"))
    else:
        print("Generating animal dataset...")
        dataset: hf_Dataset = load(split)
        dataset: hf_Dataset = dataset.filter(lambda x: x["label"] in animal_labels)
        dataset = dataset.map(map_labels)
        pickle.dump(dataset, open(pickle_path, "wb"))

    if tiny:
        # Pass the tiny_kwargs to gen_super_tiny
        dataset = gen_super_tiny(dataset, split, **tiny_kwargs)

    return dataset, coarse_labels

class ContrastiveWrapper(torch_Dataset):
    """
    A Dataset wrapper for contrastive learning tasks.

    For a given index, it returns a tuple containing:
    1. The "good" sample (anchor) corresponding to that index.
    2. A list of "bad" samples (negatives), with one randomly chosen sample
       from each of the other classes.

    Args:
        dataset (torch_Dataset): The base dataset to wrap. It is expected
            to return a tuple of (data, label) from its __getitem__ method.
        coarse_labels (Dict[str, Set[int]]): A dictionary mapping coarse
            class names to a set of their fine-grained integer labels.
    """
    def __init__(self, dataset: torch_Dataset, coarse_labels: Dict[str, Set[int]]):
        self.dataset = dataset
        self.coarse_labels = coarse_labels
        self.coarse_class_names = list(coarse_labels.keys())

        print("Indexing dataset by class for contrastive sampling...")
        self._build_indices()
        print("Indexing complete.")

    def _build_indices(self):
        """
        Builds internal mappings for efficient sampling.
        1. `self.label_to_coarse_class`: Maps a fine-grained label (int) to its coarse class name (str).
        2. `self.class_to_indices`: Maps a coarse class name (str) to a list of dataset indices.
        """
        self.label_to_coarse_class: Dict[int, str] = {}
        for class_name, fine_labels in self.coarse_labels.items():
            for label in fine_labels:
                self.label_to_coarse_class[label] = class_name

        self.class_to_indices: Dict[str, List[int]] = {name: [] for name in self.coarse_class_names}
        for i in range(len(self.dataset)):
            # We access the label directly from the underlying dataset's __getitem__
            # This can be slow but is robust. For HuggingFace datasets,
            # accessing a column is faster: self.dataset.hf_dataset['label'][i]
            try:
                _data, fine_label = self.dataset[i]
            except Exception as e:
                print(f"Could not retrieve label for index {i}. Error: {e}")
                continue

            if fine_label in self.label_to_coarse_class:
                coarse_class = self.label_to_coarse_class[fine_label]
                self.class_to_indices[coarse_class].append(i)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Any, int], List[Tuple[Any, int]]]:
        """
        Returns a good sample and a list of bad samples.

        Args:
            idx (int): The index of the "good" sample (anchor).

        Returns:
            A tuple containing:
            - good_sample (Tuple[Any, int]): The anchor sample (data, fine_label).
            - bad_samples (List[Tuple[Any, int]]): A list of negative samples,
              one from each other class.
        """
        # 1. Get the "good" sample (anchor)
        good_sample = self.dataset[idx]
        _good_data, good_fine_label = good_sample
        
        # 2. Identify the coarse class of the good sample
        anchor_coarse_class = self.label_to_coarse_class.get(good_fine_label)
        if anchor_coarse_class is None:
            raise ValueError(f"Label {good_fine_label} at index {idx} not found in coarse_labels mapping.")

        # 3. Get one "bad" sample (negative) from each of the other classes
        bad_samples = []
        for coarse_class_name in self.coarse_class_names:
            if coarse_class_name == anchor_coarse_class:
                continue  # Skip the anchor's own class

            # Get the list of indices for this "other" class
            candidate_indices = self.class_to_indices[coarse_class_name]
            if not candidate_indices:
                # This can happen if a split (e.g., test) doesn't contain a certain class
                continue
            
            # Randomly choose one index from the list
            random_bad_idx = candidate_indices[random.randrange(len(candidate_indices))]
            
            # Get the sample
            bad_sample = self.dataset[random_bad_idx]
            bad_samples.append(bad_sample)

        return good_sample, bad_samples
        
    def __repr__(self) -> str:
        return (f"ContrastiveWrapper(\n"
                f"  Underlying Dataset: {repr(self.dataset)}\n"
                f"  Coarse Classes: {self.coarse_class_names}\n"
                f"  Number of samples: {len(self)}\n)")


def contrastive_collate_fn(batch: List[Tuple[Tuple[Any, int], List[Tuple[Any, int]]]]):
    """
    Custom collate function for the ContrastiveWrapper.

    Takes a batch of data from the ContrastiveWrapper and organizes it into
    structured tensors ready for model input.

    Args:
        batch: A list of items, where each item is the output of 
               ContrastiveWrapper.__getitem__.
               Structure of one item: `( (good_image, good_label), [(bad_image_1, bad_label_1), ...] )`

    Returns:
        A tuple containing:
        - good_batch (Tuple[torch.Tensor, torch.Tensor]): 
          A tuple of (batched_images, batched_labels) for the good samples.
        - bad_batches (List[Tuple[torch.Tensor, torch.Tensor]]):
          A list where each element is a batch of negative samples. 
          For example, bad_batches[0] will be a tuple of (images, labels) for the 
          first negative sample from each item in the original batch.
    """
    # 1. Separate the good (anchor) samples and the lists of bad (negative) samples
    good_samples = [item[0] for item in batch]
    bad_samples_lists = [item[1] for item in batch]

    # 2. Collate the good samples using the default PyTorch collate function.
    # This will handle stacking the images and labels correctly.
    # good_samples is a list of (image, label) tuples.
    collated_good_samples = default_collate(good_samples)
    # 3. Collate the bad samples. This is the tricky part.
    # bad_samples_lists is a list of lists: [[(img, lbl), ...], [(img, lbl), ...]]
    # We want to "transpose" it, so we get a list of batches.
    
    # Check if there are any bad samples to process
    if not bad_samples_lists or not bad_samples_lists[0]:
        return collated_good_samples, []

    # Number of negative samples per anchor (should be consistent across the batch)
    num_neg_per_anchor = len(bad_samples_lists[0])
    
    collated_bad_samples = [([], []) for _ in range(num_neg_per_anchor)]
    for bad_samples_for_sample_i in bad_samples_lists:
        for j, (bad_sample_j, bad_label_j) in enumerate(bad_samples_for_sample_i):
            collated_bad_samples[j][0].append(bad_sample_j)
            collated_bad_samples[j][1].append(bad_label_j)
    collated_bad_samples = [(torch.stack(bad_samples), label) for bad_samples, label in collated_bad_samples]

    return collated_good_samples, collated_bad_samples



tiny_imagenet_classes = """red fish
salamander
frog 1
frog 2
crocodile
snake
fossil
scorpion
spider
tarantula
centipede
koala
medusa
coral
snail(without shell)
lobster
lobster
Black-necked stork
penguin
albatro
dolphin(?)
yorkshire
golden retriver
labrador
german shepherd
Large poodle
cat 1
cat 2
cat 3
tiger
leon
bear
Ladybugs
cricket
Bacillus rossius
cockroach
mantis
dragonfly
butterfly
leaf butterfly
marine worm
rabbit
pig
cow
buffalo
ovis
darling
gazelle
camel
mokey 1
monkey 2
monkey 3
elephant
red panda
abacus
graduate
altar
backpack
stairs
barber
house
barrel
basketball
bathub
car
lighthouse
becker
beer
bikini
binoculars
bird house
stravagante
geroglyphs
bucket
train modern
meat
holidays
cannon
woman sweater
ATM
CD Player
chest
chirstmas sock
cave
keyboard
candies
car without roof
crane
panorama
monitor
dining table
gym
flag
fly
fountain
train cargo
pan
fluffy jacket
gas mask
go kart
boat
hourglass
smartphone or mp3
cab
giapponesi
lamp
tracotr
coast guard
limousine
compass
child games
soldier
sexy woman
truck
people photo
pillar
flute
organo
parkimeter
public phone
fence
drugs
toilet
ambulance
women clothes
analcholic drink
no idea
rocket
punching bag
fridge?
TV controller
chair
football
sandals
school bus
scoreboard
sewing machine
sub mask
socks
mexican hat
strange machine
spider net
supercar
bridge
chronometer
eye glasses
bridge2
swimming costume
syringe
teapot
teddybear
medieval houses
festival
tractor 2
triumph arc
tram
turnstiles
umbrella
pope
acquedupt
volleyball
water
water cystern
hot pan
spoon
comics
fishing
vegetarian dish?
sweet dish
icecream
duck
playing instrument
party dishes
crauti
plate with food
cabbage
peppers
lemon
fruit
pomegranate
lasagna?
pizza
pie
coffe
bee
kitchen towel
idk
chihuaua
winter panorama
ravine
reef
lake
sea
acorns
cleaning
mushroom
nails
chains
worm
orange"""