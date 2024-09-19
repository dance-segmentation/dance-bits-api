import tempfile
import os
import cv2
import asyncio
import torch
import librosa
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from fastapi import File, UploadFile, FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
from contextlib import asynccontextmanager
from .model.loader import load_model


class PredictionResult(BaseModel):
    segmented_frames: list  # List of frame numbers that have been segmented by the model


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model when the application starts, not on every request.
    # Store the model in the ml_models global variable. More info here: https://fastapi.tiangolo.com/advanced/events/#lifespan
    ml_models["segmentation_model"] = load_model()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict/", response_model=PredictionResult)
async def predict(video: UploadFile = File(...)):
    if video.content_type not in ["video/mp4", "video/avi", "video/mov"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    # Save the uploaded video to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(await video.read())
            tmp_video_path = tmp_video.name

        # Process the video and perform inference
        segmentation_probs = await process_video_and_predict(tmp_video_path)
        segmented_frames = postprocess_predictions(segmentation_probs, 0.01)
        return PredictionResult(segmented_frames=segmented_frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary video file
        os.unlink(tmp_video_path)


async def process_video_and_predict(video_path: str):
    # Load the model from the global ml_models variable
    model = ml_models.get("segmentation_model")

    # Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Preprocess frames for the model
    bone_vectors = preprocess_video(frames)

    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", "_audio.wav")
    video.audio.write_audiofile(audio_path)
    video.close()
    spectrogram = generate_normalized_mel_spectrogram(audio_path)

    # Convert to PyTorch tensors
    bone_vectors = torch.tensor(bone_vectors, dtype=torch.float32)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

    # Add batch dimension
    bone_vectors = torch.stack([bone_vectors], dim=0)
    spectrogram = torch.stack([spectrogram], dim=0)

    # Perform inference asynchronously
    predictions = await run_inference(bone_vectors, spectrogram)

    # Post-process predictions
    # segmented_frames = postprocess_predictions(predictions)

    return predictions


def preprocess_video(frames):
    # Extract bone vectors and mel spectrograms from frames
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    max_frames = 3600
    video_bone_vectors = np.zeros(
        (max_frames, 35*2))  # 35 bones * 2 coordinates
    # Prepare CSV file and write header

    # Process video frames
    for index, frame in tqdm(enumerate(frames)):
        # Process frame and detect poses
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # mp_pose.POSE_CONNECTIONS contains pairs of landmarks that represent the connections (or "bones")
            # between different keypoints in the human body.
            frame_bone_vectors = get_normalized_bone_vectors(
                landmarks, mp_pose.POSE_CONNECTIONS)

            assert len(frame_bone_vectors) == len(
                mp_pose.POSE_CONNECTIONS) * 2  # 35 bones * 2 coordinates

        video_bone_vectors[index] = np.array(frame_bone_vectors)

    print(f"Processing complete. Processed {len(frames)} frames.")

    return video_bone_vectors


def pad_or_truncate(data, target_length):
    """
    Pad or truncate the data to the target length.
    - If the data has fewer frames than target_length, pad with zeros.
    - If the data has more frames than target_length, truncate it.
    """
    current_length = data.shape[0]  # Number of frames
    if current_length < target_length:
        # Pad with zeros
        padded_data = np.pad(data, [(
            0, target_length - current_length)]  # Padding with zeros along the first dimension
            # No padding for the remaining dimensions
            + [(0, 0)] * (data.ndim - 1), mode='constant')
        return padded_data
    else:
        # Truncate
        return data[:target_length]


def get_normalized_bone_vectors(landmarks, connections, vector_length=0.5):
    """
    Compute normalized bone vectors from landmarks and connections.

    Args:
    landmarks (list): List of landmarks (keypoints) from MediaPipe Pose.
    connections (list): List of connections between landmarks to create "bones".

    Returns:
    bone_vectors (list): List of normalized bone vectors.
    """

    bone_vectors = []

    for connection in connections:
        start = connection[0]
        end = connection[1]

        start_x, start_y = landmarks[start].x, landmarks[start].y
        end_x, end_y = landmarks[end].x, landmarks[end].y

        bone_vector = np.array([end_x - start_x, end_y - start_y])
        norm = np.linalg.norm(bone_vector)
        bone_vector = bone_vector / norm if norm != 0 else bone_vector
        bone_vector *= vector_length

        bone_vectors.extend(bone_vector)

    return bone_vectors


def generate_normalized_mel_spectrogram(audio_path):
    """
    Makes a Mel spectrogram of a single wav file. The output is normalized and scaled to the range [-0.5, 0.5].

    Returns:
    numpy.ndarray: Normalized and scaled Mel spectrogram. Shape: (T, n_mels) where T is the number of time frames.

    """

    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=81, n_fft=2048)
    spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)

    # Remove the audio file
    os.remove(audio_path)

    # Step 1: Normalize the spectrogram to the range [0, 1]
    min = spectrogram_dB.min()
    max = spectrogram_dB.max()
    spectrogram_dB_normalized = (spectrogram_dB - min) / (max - min)

    # Step 2: Scale the normalized values to the range [-0.5, 0.5]
    spectrogram_dB_normalized = spectrogram_dB_normalized - \
        0.5  # Shape (n_mels, T)
    spectrogram_dB_normalized = spectrogram_dB_normalized.transpose()  # Shape (T, n_mels)

    max_frames = 3600
    spectrogram_dB_normalized = pad_or_truncate(
        spectrogram_dB_normalized, max_frames)

    return spectrogram_dB_normalized


def postprocess_predictions(segmentation_probs, min_segmentation_prob=0.5):
    # Return frame ids which have a segmentation probability greater than 0.5
    segmented_frames = np.argwhere(
        segmentation_probs >= min_segmentation_prob).flatten().tolist()
    return segmented_frames


async def run_inference(bone_vectors, spectrogram):
    # loop = asyncio.get_event_loop()
    # predictions = await loop.run_in_executor(None, model_inference, bone_vectors, spectrogram)
    predictions = model_inference(bone_vectors, spectrogram)
    return predictions


def model_inference(bone_vectors, spectrogram):
    with torch.no_grad():
        model = ml_models.get("segmentation_model")
        # Move data to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        bone_vectors = bone_vectors.to(device)
        spectrogram = spectrogram.to(device)

        # Run the model
        outputs = model(bone_vectors, spectrogram)

        # Move outputs to CPU if necessary
        outputs = outputs.cpu()

        outputs = outputs.squeeze().numpy()

    print("Inference complete.")
    print(outputs)
    return outputs
