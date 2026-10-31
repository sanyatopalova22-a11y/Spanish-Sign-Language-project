import sys,os
import cv2
import numpy as np
import mediapipe as mp
from features import landmarks_to_features
 
label = sys.argv[1]
N = 200
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
os.makedirs("data",exist_ok=True)
 
sample = []
recording = False
cap = cv2.VideoCapture(0)
 
with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while len(sample) < N:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if recording:
                        sample.append(landmarks_to_features(hand))
            if recording:
                  status = f"{label}: {len(sample)}/{N}"
            else:
                    status = f"{label}: press S to start, Q to quit"
                    cv2.putText(frame, status, (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                  recording = True
            if key == ord('q'):
                  break
cap.release()
cv2.destroyAllWindows()
 
if sample:
      path = f"data/{label}.npy"
      arr = np.array(sample)
      if os.path.exists(path):
            arr = np.vstack([np.load(path),arr])
      np.save(path,arr)
      print(f"Saved {len(arr)} samples to {path}")