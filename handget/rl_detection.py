from detection import MyModel, NETWORK_SIZE, CLASSES, DEVICE, IMAGENET_MEAN, IMAGENET_STD, SCRIPT_DIR
import torch
import torchvision.transforms.v2 as T
import cv2
from torchvision import datasets, models, transforms
from torch import nn

model1 = MyModel(num_classes=len(CLASSES)).to(DEVICE)
sd1 = torch.load(
  f'{SCRIPT_DIR}/handdetection.pt',
  map_location = DEVICE,
  weights_only=True
)
model1.load_state_dict(sd1)
model1.eval()

model2 = models.resnet18(pretrained=False)
num_classes = len(CLASSES)
model2.fc = nn.Linear(model2.fc.in_features, num_classes)
model2.load_state_dict(torch.load(f'{SCRIPT_DIR}/handdetection_resnet.pt', map_location=DEVICE))
model2.to(DEVICE)
model2.eval()

INVERT_ClASSES = {1: "rock", 0: "paper", 2: "scissors"}

preprocess = T.Compose([
    T.ToImage(),                         
    T.ToDtype(torch.float32, scale=True), 
    T.Resize(NETWORK_SIZE),                   
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), 
])

def image2tensor(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  tensor = preprocess(image)
  tensor = tensor.unsqueeze(0).to(DEVICE)
  return tensor


cap = cv2.VideoCapture(0)
if not cap.isOpened(): 
  print('cam is not found')
  exit()

start_point = (500, 150)
end_point = (900, 600)
color = (0, 255, 0) 
thickness = 3

with torch.no_grad(): 
  while True:

    ret, frame = cap.read()
    if not ret:
      print('smth went wrong while reading')
      break

    cv2.rectangle(frame, start_point, end_point, color, thickness)
    cframe = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    tensor = image2tensor(cframe)
    output1 = model1(tensor)
    output2 = model2(tensor)
    prediction1 = torch.argmax(output1).item()
    prediction2 = torch.argmax(output2).item()
    sign1 = INVERT_ClASSES[prediction1]
    sign2 = INVERT_ClASSES[prediction2]

    cv2.putText(frame,
                f'{sign1}',
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (0, 255, 0), 
                3
    )
    cv2.putText(frame,
                f'{sign2}',
                (650, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (0, 0, 255), 
                3
    )

    cv2.imshow('Recognition', frame)

    if cv2.waitKey(1) % 0xFF == ord('q'): break


cap.release()
cv2.destroyAllWindows()