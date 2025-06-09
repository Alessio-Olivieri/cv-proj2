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


def load_animal_dataset(split: Literal["train", "validation", "test"]) -> Tuple[hf_Dataset, Dict]:
    coarse_labels = {
    "Aquatic": {0, 15, 16, 20, 40},
    "Amphibians & Reptiles": {1, 2, 3, 4, 5},
    "Arthropods": {7, 8, 9, 10, 32, 33, 34, 35, 36, 37, 38, 39, 184},
    "Birds": {17, 18, 19, 170},
    "Mammals": {11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 187},
    "Marine Life & Fossils": {6, 12, 13, 14, 190}
    }
    print(animal_labels)

    pickle_path: Path = paths.DATA / ("animal_"+ split + ".pkl")

    if pickle_path.is_file():
        print("Loading animal dataset from", pickle_path)
        dataset: hf_Dataset = pickle.load(open(pickle_path, "rb"))
    else:
        print("Generating animal dataset...")
        animal_labels = list(coarse_labels.values())
        animal_labels = animal_labels[0].union(*animal_labels[1:])
        dataset: hf_Dataset = load(split)
        dataset: hf_Dataset = dataset.filter(lambda x: x["label"] in animal_labels)
        pickle.dump(dataset, open("../data/animal_valid", "wb"))
    return dataset, coarse_labels



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