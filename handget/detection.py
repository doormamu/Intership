import torch
from torch import nn
import torchvision.transforms.v2 as T
import random
from torch.utils import data
import os
import numpy as np
import PIL.Image
import glob
from tqdm.auto import tqdm

NETWORK_SIZE = (224,224)
BATCH_SIZE = 16

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
  print("Using GPU")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
  print("Using MPS")
else:
  DEVICE = torch.device("cpu")
  print("Using CPU")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TRANSFORM = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.ToTensor(),
        T.Resize(size=NETWORK_SIZE),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

#augmentation if needed later

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "dataset")
ROCK_PATH = os.path.join(dataset_path, "rock")
PAPER_PATH = os.path.join(dataset_path, "paper")
SCISSORS_PATH = os.path.join(dataset_path, "scissors")
CLASSES = {"rock": 1, "paper": 0, "scissors": 2}

class MyCustomDataset(data.Dataset):
  def __init__(
      self,
      mode,
      root_dir= dataset_path,
      train_fraction = 0.8,
      split_seed = 42,
      transform = None
  ):
    paths = []
    labels = []
    rng = random.Random(split_seed)
    for cls_name, cls_idx in CLASSES.items():
      cls_paths = sorted(glob.glob(f'{root_dir}/{cls_name}/*'))
      split = int(train_fraction * len(cls_paths))
      rng.shuffle(cls_paths)

      if mode == 'train': cls_paths = cls_paths[:split]
      if mode == 'valid': cls_paths = cls_paths[split:]

      paths.extend(cls_paths)
      labels.extend(cls_idx for _ in range(len(cls_paths)))

    self._len = len(paths)
    self._paths = paths
    self._labels = np.array(labels)
    assert self._labels.shape == (self._len,)
      
    if transform is None:
      transform = DEFAULT_TRANSFORM
    
    self._transform = transform

  def __len__(self):
      return len(self._paths)
    
  def __getitem__(self, index):
      img_path = self._paths[index]
      label = self._labels[index]

      image = np.array(PIL.Image.open(img_path).convert("RGB"))

      image = self._transform(image)

      return image, label
  
'''
if __name__ == "__main__":
    my_dataset = MyCustomDataset(mode='train')
    
    print(f"Длина датасета: {len(my_dataset)}")
    
    img, label = my_dataset[0]
    
    print(f"Размер картинки: {img.shape}")
    print(f"Метка класса: {label}")

'''

ds_train = MyCustomDataset(mode='train')
ds_valid = MyCustomDataset(mode='valid')

dl_train = data.DataLoader(
  ds_train,
  batch_size=BATCH_SIZE,
  shuffle=True,
  drop_last=True,
  num_workers=0,
)

dl_valid = data.DataLoader(
  ds_valid,
  batch_size=BATCH_SIZE,
  shuffle=False,
  drop_last=False,
  num_workers=0,
)

class MyModel(nn.Module):
  def __init__(self, num_classes):
      super().__init__()

      self.model = nn.Sequential(
        
        # (3,224,224)
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), 

        # (32, 112, 112)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # (64, 56, 56)
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # (128, 28, 28)
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # (128, 14, 14) 
        nn.Flatten(start_dim=1, end_dim=-1),

        nn.Linear(128*14*14, 512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, 128),
        nn.ReLU(),

        nn.Linear(128, num_classes),
      )
  def forward(self, x):
    x = self.model(x)
    return x


def train(num_epochs):
  model = MyModel(num_classes=len(CLASSES)).to(DEVICE)
  loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  for e in range(num_epochs):
    model = model.train()
    train_loss = []
    progress_train = tqdm(
        total = len(dl_train),
        desc = f'Epoch {e}',
        leave = False,
    )
    for x_batch, y_batch in dl_train:
      x_batch = x_batch.to(DEVICE)
      y_batch = y_batch.to(DEVICE)

      p_batch = model(x_batch)
      loss = loss_fn(p_batch, y_batch)
      train_loss.append(loss.detach())

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      progress_train.update()
    progress_train.close()

    train_loss = torch.stack(train_loss).mean()
    print(
          f"Epoch {e},",
          f"train_loss: {train_loss.item():.8f}",
    )

    model = model.eval()
    valid_accs = []
    progress_valid = tqdm(
      total = len(dl_valid),
      desc = f"Epoch {e}",
      leave = False,
    )
    for x_batch, y_batch in dl_valid:
      x_batch = x_batch.to(DEVICE)
      y_batch = y_batch.to(DEVICE)

      with torch.no_grad():
        p_batch = model(x_batch)

      matches = p_batch.argmax(axis = -1) == y_batch
      valid_accs.append(matches)

      progress_valid.update()
    progress_valid.close()

    valid_accs = torch.concat(valid_accs).float().mean()
    print(
      f"Epoch {e},",
      f"valid_accs: {valid_accs.item():.8f}",
    )
  return model


if __name__ == "__main__":
  model = train(num_epochs=4)

  sd = model.state_dict()
  torch.save(sd, f'{SCRIPT_DIR}/handdetection.pt')



