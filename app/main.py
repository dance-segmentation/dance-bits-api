import tempfile
import os
import cv2
# import asyncio
import torch
import librosa
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from fastapi import File, UploadFile, FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
from contextlib import asynccontextmanager
from typing import List

from app.similarity.similarity_score import compute_similarity_for_all_frames, process_data
from .model.loader import load_model
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware


class PredictionResult(BaseModel):
    segmented_frames: List  # List of frame numbers that have been segmented by the model


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

# Allow all origins (you can specify certain domains if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/predict/", response_model=PredictionResult)
async def predict(video: UploadFile = File(...), min_segmentation_prob: float = 0.5):
    if video.content_type not in ["video/mp4", "video/avi", "video/mov"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    # Save the uploaded video to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(await video.read())
            tmp_video_path = tmp_video.name

        # Process the video and perform inference
        segmented_frames = await process_video_and_predict(tmp_video_path, min_segmentation_prob)

        segmented_frames = [int(frame_index) if isinstance(
            frame_index, np.int64) else frame_index for frame_index in segmented_frames]

        return PredictionResult(segmented_frames=segmented_frames)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary video file
        os.unlink(tmp_video_path)


@app.post("/compare")
async def compare_videos(user_video: UploadFile = File(...), teacher_video: UploadFile = File(...)):
    # Save the uploaded videos to temporary files
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_user_video:
            tmp_user_video.write(await user_video.read())
            tmp_user_video_path = tmp_user_video.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_teacher_video:
            tmp_teacher_video.write(await teacher_video.read())
            tmp_teacher_video_path = tmp_teacher_video.name

        # Process the videos and perform comparison
        similarity_score = await process_videos_and_compare(tmp_user_video_path, tmp_teacher_video_path)

        return {"similarity_score": similarity_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary video files
        os.unlink(tmp_user_video_path)
        os.unlink(tmp_teacher_video_path)


async def process_videos_and_compare(user_video_path: str, teacher_video_path: str):
    # Extract frames from the videos
    user_frames, user_fps = get_video_frames(user_video_path)
    teacher_frames, teacher_fps = get_video_frames(teacher_video_path)

    # Extract visual input (bone vectors)
    user_bone_vectors = extract_visual_input(user_frames, False)
    teacher_bone_vectors = extract_visual_input(teacher_frames, False)

    # Process the data
    user_bone_vectors, teacher_bone_vectors = process_data(
        user_bone_vectors, teacher_bone_vectors, user_fps, teacher_fps)

    # Compute similarity score
    similarity_score = compute_similarity_for_all_frames(
        user_bone_vectors, teacher_bone_vectors)

    return similarity_score


async def process_video_and_predict(video_path: str, min_segmentation_prob: float = 0.5):
    # Extract frames from the video
    frames, _ = get_video_frames(video_path)

    # Extract visual input (bone vectors)
    bone_vectors = extract_visual_input(frames)
    # Extract audio input (mel spectrogram)
    spectrogram = extract_audio_input(video_path)

    # Convert to PyTorch tensors
    bone_vectors = torch.tensor(bone_vectors, dtype=torch.float32)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

    # Add batch dimension
    bone_vectors = torch.stack([bone_vectors], dim=0)
    spectrogram = torch.stack([spectrogram], dim=0)

    # Perform inference asynchronously
    predictions = await run_inference(bone_vectors, spectrogram)

    # Post-process predictions
    segmented_frames = postprocess_predictions(
        segmentation_probs=predictions, num_frames=len(frames), min_segmentation_prob=min_segmentation_prob)

    return segmented_frames


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def extract_audio_input(video_path):
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", "_audio.wav")
    video.audio.write_audiofile(audio_path)
    video.close()
    spectrogram = generate_normalized_mel_spectrogram(audio_path)
    return spectrogram


def extract_visual_input(frames, fixed_len=True):
    """
    Extract visual input (bone vectors) from video frames using MediaPipe Pose.

    Args:
    frames (list): List of video frames.
    fixed_len (bool): If True, pad or truncate the bone vectors to a fixed length of max_frames.

    Returns:
    numpy.ndarray: Array of bone vectors. Shape: (max_frames or len(frames), 35*2) where 35 is the number of bones and 2 is the number of coordinates (x, y).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    max_frames = 5400
    if fixed_len:
        video_bone_vectors = np.zeros(
            (max_frames, 35*2))  # 35 bones * 2 coordinates
    else:
        video_bone_vectors = np.zeros(
            (len(frames), 35*2))  # 35 bones * 2 coordinates

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

    max_frames = 5400
    spectrogram_dB_normalized = pad_or_truncate(
        spectrogram_dB_normalized, max_frames)

    return spectrogram_dB_normalized


def postprocess_predictions(segmentation_probs, num_frames, max_frames=5400, min_segmentation_prob=0.3, window_size=20):
    """
    Post-process the model predictions to get the segmented frames.
    Outputs the segmented frames given probabilities for each frame by finding the frame with the maximum probability
      exceeding a threshold h in a window of size w.

    segmentation_probs: (Array[float]) The probability per frame. Shape (nr_frames, 1).
    num_frames: (int) The number of frames in the video.
    max_frames: (int) The maximum number of frames the video was padded or truncated to.
    min_segmentation_prob: (float) The probability threshold to consider a potential segmentation point.
    window_size: (int) The size of the window to calculate a probability maximum in number of frames.
    """

    # Remove padding
    num_frames = min(num_frames, max_frames)
    segmentation_probs = segmentation_probs[:num_frames]

    frame_indices = []

    for i, label in enumerate(segmentation_probs):

        if label > min_segmentation_prob:
            # Find the lower and upper limits of the index range
            # for the boundary cases.
            low_lim = max(0, i - int(window_size / 2))
            up_lim = min(i + int(window_size / 2), len(segmentation_probs) - 1)

            # Create the index window for finding the probability maximum.
            index_window = list(range(low_lim, up_lim + 1))

            # Remove
            for index in index_window:
                if index in frame_indices:
                    frame_indices.remove(index)

            # Find the index of the maximum probability in the index window.
            max_prob_index = np.argmax(segmentation_probs[index_window])
            max_prob = np.max(segmentation_probs[index_window])

            # Map the index in the local range to the global index.
            label_index = low_lim + max_prob_index

            # Check that the max probability locally and the one with
            # the mapped index are identical.
            assert max_prob, segmentation_probs[label_index]

            frame_indices.append(label_index)

    return frame_indices


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
