import torch
import cv2

model = torch.hub.load("ultralytics/yolov5", "yolov5x")
feed = cv2.VideoCapture(0)
while True:
  ret, frame = feed.read()
  if not ret:
    break
  results = model(frame)
  for result in results.xyxy[0]:
      x1, y1, x2, y2, conf, cls = result.cpu().numpy()
      color = (255,0,0)
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
      text = f"{model.names[int(cls)]} p: {conf:.2f}"
      cv2.putText(frame, text, (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  cv2.imshow("Object Detection", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
feed.release()
cv2.destroyAllWindows()
