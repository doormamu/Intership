import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "dataset")
ROCK_PATH = os.path.join(dataset_path, "rock")
PAPER_PATH = os.path.join(dataset_path, "paper")
SCISSORS_PATH = os.path.join(dataset_path, "scissors")
cap = cv2.VideoCapture(0)

if not cap.isOpened(): 
  print('cam is not found')
  exit()

os.makedirs(ROCK_PATH, exist_ok=True)
os.makedirs(PAPER_PATH, exist_ok=True)
os.makedirs(SCISSORS_PATH, exist_ok=True)
print('press q for exit')

current_path = PAPER_PATH
img_count = 0

start_point = (500, 150)
end_point = (900, 600)
color = (0, 255, 0) 
thickness = 3

while True:
  ret, frame = cap.read()
  if not ret: 
    print('smth went wrong while reading')
    break

  cv2.rectangle(frame, start_point, end_point, color, thickness)
  cv2.putText(frame, f"Saved: {img_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

  cv2.imshow('Cam test', frame)

  key = cv2.waitKey(1)
  if key & 0xFF == ord('s'):
    fname = f'{current_path}/img{img_count}.jpg'
    cv2.imwrite(fname, frame[start_point[1]:end_point[1], start_point[0]:end_point[0]])
    print(f'image {img_count} saved to {current_path}')
    img_count += 1

  if key & 0xFF ==  ord('q'):
    break