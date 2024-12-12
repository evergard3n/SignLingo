import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
from threading import Lock

app = FastAPI()

main_event_loop = None

@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

# Shared state
latest_prediction = {"value": None}
prediction_lock = Lock()
previous_message = ""
message_lock = Lock()


active_connections: List[WebSocket] = []

async def broadcast_prediction(message: str):
    """Send a message to all connected clients."""
    global previous_message
    # await asyncio.sleep(0)
    for connection in active_connections:
        try:
            await connection.send_text(message)
            print(message)
        except Exception:
                active_connections.remove(connection)
    # async with message_lock:
    #     print(f"Previous: {previous_message}, Current: {message}")
    #     if message != previous_message:
    #         previous_message = message
    #         for connection in active_connections:
    #             try:
    #                 await connection.send_text(message)
    #                 print(message)
    #             except Exception:
    #                 active_connections.remove(connection)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for live predictions."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep the connection alive (waiting for messages, optional)
            await websocket.receive_text()
            # for client in active_connections:
            #     await client.send_text(latest_prediction["value"])
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Model setup
actions = np.array(['cha','me','chi'])
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('action.h5')

class SignLanguageDetector:
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.9
        self.count = 0
        self.last_prediction = ''
        
    def process_frame(self, frame, results):
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]  # Keep last 30 frames
        
        current_prediction = ''
        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            if res[np.argmax(res)] > self.threshold:
                prediction = actions[np.argmax(res)]
                if prediction == self.last_prediction:
                    self.count += 1
                    if self.count > 2:
                        self.sentence = [prediction]
                        current_prediction = prediction
                        self.count = 0
                self.last_prediction = prediction
        
        if self.sentence:  # Check if the list is not empty
            return self.sentence[-1]
        else:
            return None  # or some other default value

    @staticmethod
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

detector = SignLanguageDetector()

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not access webcam")
    
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Draw landmarks
                # if results.pose_landmarks:
                #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                # if results.face_landmarks:
                #     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                # if results.left_hand_landmarks:
                #     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # if results.right_hand_landmarks:
                #     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Process for prediction
                prediction = detector.process_frame(frame, results)
                #broadcast prediction through ws
                if prediction:
                    with prediction_lock:
                        latest_prediction["value"] = prediction
                        
                    print(latest_prediction["value"])
                    # import asyncio
                    if main_event_loop:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_prediction(prediction),
                            main_event_loop
                        )
                # broadcast_prediction(latest_prediction["value"])
                if prediction:
                    cv2.putText(image, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    broadcast_prediction(prediction)
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Sign Language Detection</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                .video-container {
                    margin-top: 20px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
            </style>
        </head>
        <body>
            <h1>Sign Language Detection</h1>
            <div class="video-container">
                <img src="/video_feed" width="640" height="480" alt="Video stream">
            </div>
            
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail="Camera not accessible")
        cap.release()
        return JSONResponse(status_code=200, content={"status": "healthy"})
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/video_feed")
async def video_feed():
    try:
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/latest_prediction")
async def get_latest_prediction():
    with prediction_lock:
        return JSONResponse(content={"latest_prediction": latest_prediction["value"]})