import cv2
import mediapipe as mp

# cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 
connected_landmarks = mpHands.HAND_CONNECTIONS


# while True:
#     success, img = cap.read()
#     imgRGB  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#     cv2.imshow("Image", img)
    
#     if cv2.waitKey(1) == ord("q"):
#         cv2.destroyAllWindows()
        # exit(0)

def get_hands(frame):
    imgRGB  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

   