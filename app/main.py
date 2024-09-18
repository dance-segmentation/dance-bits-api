import tempfile
import os
import cv2
import asyncio
import torch
from fastapi import File, UploadFile, FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model.loader import load_model


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
        segmented_frames = await process_video_and_predict(tmp_video_path)

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
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Preprocess frames for the model
    processed_frames = preprocess_video(frames)

    # Perform inference asynchronously
    predictions = await run_inference(processed_frames)

    # Post-process predictions
    segmented_frames = postprocess_predictions(predictions)

    return segmented_frames


def preprocess_video(frames):
    # Extract bone vectors and mel spectrograms from frames
    pass


def postprocess_predictions(predictions):
    # Convert model outputs to frame numbers
    pass


async def run_inference(processed_frames):
    loop = asyncio.get_event_loop()
    predictions = await loop.run_in_executor(None, model_inference, processed_frames)
    return predictions


def model_inference(processed_frames):
    with torch.no_grad():
        model = ml_models.get("segmentation_model")
        # Move data to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        processed_frames = processed_frames.to(device)

        # Run the model
        outputs = model(processed_frames)

        # Move outputs to CPU if necessary
        outputs = outputs.cpu()

    return outputs
