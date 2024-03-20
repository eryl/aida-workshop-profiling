import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import importlib
from pathlib import Path
import argparse
from statistics import mean
from typing import Union, TypeVar, Type, Optional
import datetime
from collections import deque
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.v2 as v2
from torchvision.datasets import OxfordIIITPet

import tqdm

from experiment import ExperimentConfig

try:
    @profile
    def foo():
        return
except NameError:
    def profile(f):
        return f

def timestamp():
    """
    Generates a timestamp.
    :return:
    """
    t = datetime.datetime.now().replace(microsecond=0)
    #Since the timestamp is usually used in filenames, isoformat will be invalid in windows.
    #return t.isoformat()
    # We'll use another symbol instead of the colon in the ISO format
    # YYYY-MM-DDTHH:MM:SS -> YYYY-MM-DDTHH.MM.SS
    time_format = "%Y-%m-%dT%H.%M.%S"
    return t.strftime(time_format)


def load_module(module_path: Union[str, Path]):
    """
    Loads a python module with the given module path
    :param module_path: Path of module file to load
    :return: Module object
    """
    spec = importlib.util.spec_from_file_location("module_from_file", module_path)
    module_from_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_from_file)
    return module_from_file


T = TypeVar('T')
def load_object(module_path: Path, object_type: Type[T], default: Optional[Type[T]]=None) -> T:
    """
    Given a file path, load it as a module and return the first matching object of *object_type*
    :param module_path: File containing the module to load
    :param object_type: The object type to look for. E.g. a custom object or dataclass instance
    :param default: If an instance of the desired class could not be found, return this value instead.
    :return: The first found instance of *object_type*. If no instance is found, ValueError is raised
    """
    mod = load_module(module_path)
    for k,v in mod.__dict__.items():
        if isinstance(v, object_type):
            return v
    if default is not None:
        return default
    else:
        raise ValueError(f"File {module_path} does not contain any attributes of type {object_type}")


class CustomSlowDataset(Dataset):
    def __init__(self, root='datasets', split='trainval', transform=None, wasted_time=.1):
        super().__init__()
        self.wrapped_dataset = OxfordIIITPet(root, split=split, transform=transform, download=True)
        self.wasted_time = wasted_time
        self.classes = self.wrapped_dataset.classes
        self.class_to_idx = self.wrapped_dataset.class_to_idx
    
    def __len__(self):
        return len(self.wrapped_dataset)
    
    def __getitem__(self, index):
        item = self.wrapped_dataset[index]
        time.sleep(self.wasted_time)
        return item
    

class CachedDataset(Dataset):
    def __init__(self, wrapped_datset, cachedir='dataset_cache'):
        self.wrapped_dataset = wrapped_datset
        self.cachedir = Path(cachedir)
        self.cachedir.mkdir(exist_ok=True, parents=True)
        self.classes = self.wrapped_dataset.classes
        self.epoch = 0
        
    def __len__(self):
        return len(self.wrapped_dataset)
    
    def __getitem__(self, index):
        epoch_dir = self.cachedir / f"epoch_{self.epoch}"
        epoch_dir.mkdir(exist_ok=True)
        item_file = epoch_dir / f"{index}.pkl"
        if item_file.exists():
            with open(item_file, 'rb') as fp:
                item = pickle.load(fp)
        else:
            item = self.wrapped_dataset[index]
            with open(item_file, 'wb') as fp:
                pickle.dump(item, fp)
        return item
    
    def next_epoch(self):
        self.epoch += 1
                    

def main():
    parser = argparse.ArgumentParser(description="Script for illustrating data loading")
    
    parser.add_argument('config', type=Path, help="Path to the config to use")
    parser.add_argument('--experiments', type=Path, help="Path to save experiment results to", default=Path('experiments'))
    parser.add_argument('--device', help="What device to use", default='cuda')
    
    args = parser.parse_args()
    config = load_object(args.config, ExperimentConfig)
    
    # We start by creating a timestamped folder to save the experiment data to
    experiment_directory = args.experiments / timestamp()
    experiment_directory.mkdir(parents=True)
    
    device = torch.device(args.device)
    
    # Now set up the model, we'll use a small resnet to really make compute problems obvious
    model = resnet18()
    weights = ResNet18_Weights.DEFAULT  
    #model = resnet18(weights=weights)  # If you want to use a pretrained net, uncomment this line

    #preprocess = weights.transforms()  # The recommended way is to use these transforms. They rely on the older transforms-framework though, which is mixes poorly with the updated v2
    
    # Here's a part to be mindful of. We're adding data-augmentation transforms to the training pipeline:
    training_preprocess = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        v2.AutoAugment(),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomResizedCrop(224)
    ])

    test_preprocess = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize(224)
    ])
   
   
    #training_dataset = OxfordIIITPet('datasets', 'trainval', transforms=training_preprocess, download=True)
    training_dataset = CustomSlowDataset('datasets', 'trainval', transform=training_preprocess)
    test_dataset = CustomSlowDataset('datasets', 'test', transform=test_preprocess)
    training_dataset = CachedDataset(training_dataset, cachedir='training_dataset_cache')
    
    # Determine the number of input features to the output layer:
    num_ftrs = model.fc.in_features
    num_classes = len(training_dataset.classes)
    
    model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=4)
    
    historic_losses = deque(maxlen=config.update_iterations)

    for epoch in tqdm.trange(config.max_epochs, desc='Epoch'):
        training_pbar = tqdm.tqdm(training_dataloader, desc='Training batch')
        for i, batch in enumerate(training_pbar):
            optimizer.zero_grad()
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            logits = model(input)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            historic_losses.append(loss.item())
            if i % config.update_iterations == 0:
                mean_loss = mean(historic_losses)
                training_pbar.set_description(f'Training batch ({mean_loss})')
                
        if hasattr(training_dataset, 'next_epoch'):
            training_dataset.next_epoch()
        
    

if __name__ == '__main__':
    main()
                
                
            
        
            
            
            
    
    
    
    