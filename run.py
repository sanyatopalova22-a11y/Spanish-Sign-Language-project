import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from features import landmarks_to_features

#for aesthetic clarity
ACCENT = (255, 190, 65)
WHITE = (245, 245, 245)
GREY = (170, 170, 170)
DARK = (20, 20, 20)
FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

CONF_THRESHOLD = 0.7
WINDOW = 10
NEEDED = 8

def panel(img, x1, y1, x2, y2, color = DARK, alpha = 0.6, r = 16):
     overlay = img.copy()
     r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
     cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
     cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
     for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                    (x1 + r, y2 - r), (x2 - r, y2-r)]:
          cv2.circle(overlay, (cx, cy), r, color, -1, AA)
     cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def text_size(text, scale, thick=1):
     (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
     return w,h

def put(img, text, x, y, scale, color, thick=1):
     cv2.putText(img, text, (x,y), FONT, scale, color, thick, AA)

def draw_hand(img, hand):
     h,w = img.shape[:2]
     pts = [(int(ldm.x*w), int(ldm.y*h)) for ldm in hand.landmark]
     for a,b in mp.solutions.hands.HAND_CONNECTIONS:
          cv2.line(img, pts[a], pts[b], WHITE, 2, AA)
     for p in pts:
          cv2.circle(img, p, 5, ACCENT, -1, AA)
          cv2.circle(img, p, 5, WHITE, 1, AA)

def draw_status(img, hand_seen, candidate, progress):
     if not hand_seen:
          label, dot = "Show your hand", GREY
     elif candidate:
          label, dot = f"Detecting: {candidate}", ACCENT
     else:
          label, dot = "Listening...", WHITE

     tw, th = text_size(label, 0.6)
     x1, y1 = 16, 16
     x2, y2 = x1 + tw + 60, y1 + 50
     panel(img, x1, y1, x2, y2, alpha = 0.55, r = 14)
     cv2.circle(img, (x1 + 20, y1 + 21), 6, dot, -1, AA)
     put(img, label, x1 + 36, y1 + 28, 0.6, WHITE)
     sx1, sx2, sy = x1 + 16, x2 - 16, y2 - 10
     cv2.line(img, (sx1, sy), (sx2, sy), (80, 80, 80), 3, AA)
     if progress>0:
          fill = sx1 + int((sx2 - sx1)*min(progress, 1.0))
          cv2.line(img, (sx1, sy), (fill, sy), ACCENT, 3, AA)

def draw_subtitles(img, words, flash):
     h,w = img.shape[:2]
     scale, thick = 1.1,2
     shown = words[-6:]
     space, _ = text_size(" ", scale, thick)

     if shown: 
          widths = [text_size(wd, scale, thick)[0] for wd in shown]
          total = sum(widths) + space * (len(shown)-1)
     else:
          total = text_size("Subtitles appear here", 0.7)[0]

     pad = 28
     bw = max(total + pad*2, int(w*0.45))
     x1 = (w-bw)//2
     y1, y2 = h-100, h-30
     panel(img, x1, y1, x1 + bw, y2, alpha =0.65, r =20)

     baseline = y2-24
     if not shown:
          message = "Subtitles appear here"
          put( img, message, (w-total)//2, baseline - 4, 0.7, GREY)
          return

     x = (w-total)//2
     for i, (wd, ww) in enumerate(zip(shown, widths)):
          newest = i == len(shown) - 1
          color = ACCENT if newest and flash > 0 else WHITE
          put(img, wd, x, baseline, scale, color, thick)
          x += ww + space

def draw_hints(img):
     h,w = img.shape[:2]
     hint = "C clear    Q quit"
     tw, _ = text_size(hint, 0.45)
     put(img, hint, w - tw - 18, h - 10, 0.45, (215, 215, 215))


#for functionality
model = joblib.load("model.joblib")
mp_hands = mp.solutions.hands
 
recent = deque(maxlen=WINDOW)
sentence = []
last_word = None
flash = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Sign to subtitles", cv2.WINDOW_NORMAL)

with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            guess = None
            hand_seen = bool(result.multi_hand_landmarks)
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                draw_hand(frame, hand)
                probs = model.predict_proba([landmarks_to_features(hand)])[0]
                i = int(np.argmax(probs))
                if probs[i] > CONF_THRESHOLD:
                    guess = model.classes_[i]
            recent.append(guess)

            candidate, progress = None, 0.0
            votes = [g for g in recent if g and g != "neutral"]
            if votes:
                 candidate, count = Counter(votes).most_common(1)[0]
                 progress = count / NEEDED

            word = None
            if len(recent) == recent.maxlen:
                top, count = Counter(recent).most_common(1)[0]
                if top is not None and top!= "neutral" and count >= NEEDED:
                     word = top
            
            if word is None:
                 last_word = None
            elif word != last_word:
                sentence.append(word)
                last_word = word
                flash = 20

            draw_status(frame, hand_seen, candidate, progress)
            draw_subtitles(frame, sentence, flash)
            draw_hints(frame)
            flash = max(0, flash - 1)
            
            cv2.imshow("Sign to subtitles", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                sentence.clear()
 
cap.release()
cv2.destroyAllWindows()