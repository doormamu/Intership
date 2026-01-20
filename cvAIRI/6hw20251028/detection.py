
import torch
from torch import nn
import PIL.Image
import os
import torchvision.transforms.v2 as T
import random
import numpy as np
from torch.utils import data
import tqdm
import glob
import gc
from PIL import ImageEnhance
import albumentations as A
import matplotlib.pyplot as plt

def show_images(image, kps):
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis('off')

    plt.scatter(kps[:, 0], kps[:, 1], s=10, c='red')


    plt.show()


'''
CLASSES = {"BrowEndL": 0, "BrowBegL": 1, "BrowBegR": 2, "BrowEndR": 3,
            "EyeEndL": 4, "EyeL" : 5, "EyeBegL" : 6, "EyeBegR": 7, "EyeR" : 8, "EyeEngR" : 9,
            "Nose" : 10,
            "LipsL" : 11, "Lips" : 12, "LipsR" : 13
            }
'''
CLASSES = 14*2
NETWORK_SIZE = (100,100)
BATCH_SIZE = 64
NUM_EPOCHS = 15
NUM_WORKERS = 4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using the GPU 😊")
else:
    DEVICE = torch.device("cpu")
    print("Using the CPU 😞")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TRANSFORM = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=NETWORK_SIZE),
        #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ]
)

AugTransform = A.Compose(
    [
        A.Affine(translate_percent=0.05, scale=(1.0, 1.05), rotate=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        #A.HorizontalFlip()

    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)



class MyCustomDataset(data.Dataset):
    def __init__(
            self,
            mode,
            root_dir = "./public_tests/00_test_img_input/train/gt.csv",
            img_dir = "./public_tests/00_test_img_input/train/images",
            train_fraction = 0.8,
            split_seed = 42,
            transform = None,
            fast_train = False
      ):
        
        if isinstance(root_dir, dict):
            samples = [(fname, np.array(coords, dtype=np.float32))
                    for fname, coords in root_dir.items()]
        else:
            with open(root_dir) as f:
                data = f.readlines()[1:]
            samples = []
            for line in data:
                line = list(line.strip().split(","))
                img_name = line[0]
                keypoints = np.array([float(x) for x in line[1:]], dtype=np.float32)
                samples.append((img_name, keypoints))

        rng = random.Random(split_seed)
        rng.shuffle(samples)

        split = int(train_fraction * len(samples))

        if mode == "train":
            samples = samples[:split]
        elif mode == "valid":
            samples = samples[split:]
        else:
            raise RuntimeError(f"Invalid mode: {mode!r}")

        self._samples = samples
        self._len = len(samples)
        #self._root_dir = root_dir
        self._img_dir = img_dir
        self._mode = mode
        self.fast_train = fast_train

        if transform is None:
            transform = DEFAULT_TRANSFORM
        self._transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index = None):
        img_name, keypoints = self._samples[index]

        image = PIL.Image.open(f"{self._img_dir}/{img_name}").convert("RGB")
        self._w, self._h= image.size
        image = image.resize(NETWORK_SIZE, PIL.Image.BILINEAR)
        image = np.array(image)

        keypoints = keypoints.copy().reshape(-1, 2)

        keypoints[:, 0] = keypoints[:, 0] * NETWORK_SIZE[0] / self._w
        keypoints[:, 1] = keypoints[:, 1] * NETWORK_SIZE[1] / self._h

        if self._mode == "train":
            keypoints = keypoints.copy().reshape(-1,2)

            img_aug, kp_aug = self.aug(image, keypoints)

            keypoints = kp_aug.flatten()

            image = self._transform(img_aug)
        else: image = self._transform(image)

        keypoints_scaled = keypoints.copy().flatten()
        keypoints_scaled[::2] = keypoints_scaled[::2] /NETWORK_SIZE[0]
        keypoints_scaled[1::2] = keypoints_scaled[1::2] /NETWORK_SIZE[1]

        return image, keypoints_scaled

    def aug(self, image, keypoints):
        transformed = AugTransform(image = image,keypoints = keypoints)
        img_aug = transformed["image"]
        kp_aug = np.array(transformed["keypoints"])

        if random.random() < 0.5:
            img_aug, kp_aug = self.flip(img_aug, kp_aug)
        return img_aug, kp_aug


    def flip(self,image, keypoints):
        flipped_image_np = np.flip(image, axis=1).copy()
        #flipped_image = PIL.Image.fromarray(flipped_image_np)

        #keypoints = keypoints.copy().reshape(-1,2)

        swap_pairs = [(0,3),(1,2),(4,9),(5,8),(6,7),(11,13)]
        for a, b in swap_pairs:
            keypoints[[a,b]] = keypoints[[b,a]]

        keypoints[:,0] = image.shape[1] - keypoints[:,0]
        return flipped_image_np, keypoints


class MyModel(nn.Sequential):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)      

        self.conv2 = nn.Conv2d(16, 38, kernel_size=3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(38)
        self.relu2 = nn.ReLU()
        #self.drop3 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(38, 196, kernel_size=3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(196)
        self.relu3 = nn.ReLU()
        #self.drop2 = nn.Dropout(0.3)

        self.adapt = nn.AdaptiveAvgPool2d((4,4))
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(196*4*4, 256)  
        self.relu4 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)

        self.dense2 = nn.Linear(256, num_classes)

def train(num_epochs, dl_train, dl_valid):
    model = MyModel(num_classes=CLASSES).to(DEVICE)
    loss_fn = nn.SmoothL1Loss() 
    valid_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)


    for e in range(num_epochs):
        model = model.train()
        ###
        train_loss = []
        progress_train = tqdm.tqdm(
            total= len(dl_train),
            desc=f"Epoch {e}",
            leave = False,
        )
        ###
        for x_batch, y_batch in dl_train:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            p_batch = model(x_batch)
            loss = loss_fn(p_batch, y_batch)
            ###
            train_loss.append(loss.detach())
            ###

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ###
            progress_train.update()
        progress_train.close()

        ###
        train_loss = torch.stack(train_loss).mean()
        print(
            f"Epoch {e},",
            f"train_loss {train_loss.item():.8f}",
        )
        ###

        model = model.eval()
        ###
        valid_losses = []
        progress_valid = tqdm.tqdm(
            total=len(dl_valid),
            desc=f"Epoch {e}",
            leave=False,
        )
        ###

        for x_batch, y_batch in dl_valid:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            with torch.no_grad():
                p_batch = model(x_batch)

        ###
            loss_val = valid_loss(p_batch, y_batch)
            valid_losses.append(loss_val)

        ###
            progress_valid.update()
        progress_valid.close()
    ###
        valid_losses = torch.stack(valid_losses).mean()
        print(
            f"Epoch {e},",
            f"valid_losses: {valid_losses.item():.8f}",
            )

        scheduler.step(valid_losses)

        torch.cuda.empty_cache()
        gc.collect()
    return model


#if __name__ == "__main__":
#model = train(num_epochs=3)

class MyTestDataset(data.Dataset):
    def __init__(
            self,
            path,
            transform = None,
            ):

        if transform is None:
            transform = DEFAULT_TRANSFORM
        self._transform = transform
        self._path = path
        self.img_paths = glob.glob(os.path.join(path, "*jpg"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = np.array(PIL.Image.open(img_path).convert("RGB"))
        image = self._transform(image)
        return image, img_path


def detect(path, test_path):
    BATCH_SIZE = 1
    model = MyModel(num_classes=CLASSES).to(DEVICE)
    sd = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(sd)
    model = model.eval()
    ds = MyTestDataset(test_path)
    dl_test = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    otv_dict = {}
    with torch.no_grad():
        for x_batch, paths in dl_test:
            img_path = paths[0]
            with PIL.Image.open(img_path) as img:
                w, h = img.size

            x_batch = x_batch.to(DEVICE)
            pred = model(x_batch)[0].cpu().numpy()

            pred[::2] = pred[::2] * w
            pred[1::2] = pred[1::2] * h
            
            fname = os.path.basename(img_path)
            otv_dict[fname] = pred.tolist()

            del x_batch, pred
            torch.cuda.empty_cache()
    '''
    for x_batch, y_batch, in dl_test:
        x_batch = x_batch.to(DEVICE)
        #y_batch = y_batch.to(DEVICE)

        with torch.no_grad():
            p_batch = model(x_batch)
        for y, p in zip(y_batch, p_batch):
            y = os.path.basename(y)
            otv_dict[y] = p
    '''
    return otv_dict

def train_detector(csv_path, img_paths, fast_train):
    if fast_train == False:
        BATCH_SIZE = 128
        NUM_EPOCHS = 50
        NUM_WORKERS = 4
    if fast_train == True:
        BATCH_SIZE = 16
        NUM_EPOCHS = 3
        NUM_WORKERS = 0

    ds_train = MyCustomDataset(mode="train", root_dir=csv_path, img_dir=img_paths, fast_train=fast_train)
    ds_valid = MyCustomDataset(mode="valid", root_dir=csv_path, img_dir=img_paths, fast_train=fast_train)

    dl_train = data.DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    dl_valid = data.DataLoader(
        ds_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    model = train(num_epochs=NUM_EPOCHS,dl_train=dl_train,dl_valid=dl_valid)
    return model

'''
model = train_detector("/content/drive/MyDrive/6hw20251028/gt.csv",
                                 "/content/images/images",
                                 False)
                              
sd = model.state_dict()
#print(list(sd.keys()))
#print(sd["conv1.bias"])
#print()
#print(model.conv1.bias)
torch.save(sd, "facepoints_model.pt")

p = detect("facepoints_model.pt", "/content/images/images")
print(p['00000.jpg'])
'''