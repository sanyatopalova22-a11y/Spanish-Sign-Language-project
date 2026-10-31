# Spanish-Sign-Language-project
A prototype for a video call app that will translate LSE (Lengua de Signos Española, Spanish Sign Language) to any spoken language requested, in the subtitles below the call. 

Sign to Subtitles: prototype

A first prototype of a tool that turns sign language into live subtitles, built with the long-term goal of helping the deaf community in Spain take part in video calls. This version recognises a small set of static hand shapes from a webcam and shows the recognised word as a subtitle on the video.

What it does:
- Opens your webcam.
- Tracks 21 points on your hand in every frame.
- Predicts which sign you are making.
- Shows the recognised word as a subtitle at the bottom of the video.

Recognised signs (Lengua de Signos Española, LSE): hello, thank you, no, "I can speak Spanish or French", most of the alphabet (some symbols require movement), plus a neutral class for "no sign".

How it works
- Hand tracking. MediaPipe Hands finds 21 landmarks (x, y, z) on the hand in each video frame.
- Feature extraction (features.py). The 21 points are converted into 63 numbers. They are made relative to the wrist and scaled by hand size, so the result does not depend on where the hand is in the frame or how far it is from the camera.
- Classification (train.py). A Random Forest classifier (scikit-learn) is trained on my own recorded examples of each sign.
- Smoothing (run.py). A single frame can be wrong, so a guess is ignored if its confidence is below 70%, and a word is only shown if it wins at least 8 of the last 10 frames. The neutral class never produces a subtitle. This removes flicker and accidental words.

File's Purposes
hand_test.py - Checks that hand tracking works on your camera
features.py - Converts hand landmarks into the 63 features
collect.py - Records training examples for one sign
train.py - Trains the classifier and saves model.joblib
run.py - Live subtitle app
data/ - Recorded training data (one .npy file per sign)
model.joblib - The trained model

This is not a large AI model. It is a small classifier on top of MediaPipe's hand tracking, trained on data I recorded myself.
This Read-Me was generated with AI, the source code was hand-written, supported by my mentor
