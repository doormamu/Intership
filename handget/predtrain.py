import torch
from torch import nn
import os
from tqdm.auto import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

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
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(size=NETWORK_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "dataset")
ROCK_PATH = os.path.join(dataset_path, "rock")
PAPER_PATH = os.path.join(dataset_path, "paper")
SCISSORS_PATH = os.path.join(dataset_path, "scissors")
CLASSES = {"rock": 1, "paper": 0, "scissors": 2}

ds = datasets.ImageFolder(root=dataset_path, transform=DEFAULT_TRANSFORM)
train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
ds_train, ds_valid = random_split(ds, [train_size, val_size])

dl_train = DataLoader(
  ds_train,
  batch_size=BATCH_SIZE,
  shuffle=True,
  drop_last=True,
  num_workers=0,
)

dl_valid = DataLoader(
  ds_valid,
  batch_size=BATCH_SIZE,
  shuffle=False,
  drop_last=False,
  num_workers=0,
)



def train(num_epochs):
  model = models.resnet18(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False
  num_classes = len(ds.classes)
  model.fc = nn.Linear(model.fc.in_features, num_classes)

  model = model.to(DEVICE)
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
  torch.save(sd, f'{SCRIPT_DIR}/handdetection_resnet.pt')




