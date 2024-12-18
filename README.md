# DanceBits API: ML-powered automated choreography video segmentation

Backend implementing a multimodal AI model in PyTorch and serving it using FastAPI and Docker to automatically identify and label dance moves in videos to create an interactive learning platform. Features: video preprocessing, pose estimation, audio processing, and multimodal segmentation model.

## Features

- Advanced pose estimation and motion feature extraction using MediaPipe
- Audio feature extraction for enhanced move detection
- Real-time multimodal dance move segmentation model
- User-friendly learning interface with customizable speeds and segment sizes
- Side-by-side webcam/video option with recording functionality
- Similarity score calculation for comparing dance performances

## Requirements

- Python 3.11 or higher
- FFmpeg
- CUDA-compatible GPU (optional, for faster inference)
- Docker (optional, for containerized deployment)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/dance-bits-api.git
   cd dancebits
   ```

2. Create a Conda environment and activate it:
   ```bash
   conda create --name dance-bits-api python
   conda activate dance-bits-api
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Note: You can also install via Conda, but some packages may not be available:
   ```bash
   conda install --file requirements.txt
   ```

## Local Deployment

### Running the Model Locally

1. Set up environment variables in a `.env` file:
   ```bash
   WANDB_API_KEY=your_key
   WANDB_ORG=your_org
   WANDB_PROJECT=your_project
   WANDB_MODEL_NAME=your_model
   WANDB_MODEL_VERSION=your_version
   ```

2. Install FFmpeg (required for video processing):
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get update
     sudo apt-get install ffmpeg libsm6 libxext6
     ```
   - On macOS:
     ```bash
     brew install ffmpeg
     ```
   - On Windows:
     Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH

3. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
   ```

4. Access the API:
   - API documentation: http://localhost:8080/docs
   - Alternative API docs: http://localhost:8080/redoc

### Testing the API

1. Test video segmentation:
   ```bash
   curl -X POST "http://localhost:8080/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@path/to/your/dance_video.mp4" \
     -F "min_segmentation_prob=0.5"
   ```

2. Test video comparison:
   ```bash
   curl -X POST "http://localhost:8080/compare/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "user_video=@path/to/user_video.mp4" \
     -F "teacher_video=@path/to/teacher_video.mp4"
   ```

### Troubleshooting

1. Model Loading Issues:
   - Ensure all environment variables are set correctly
   - Check if the model weights are downloaded properly
   - Verify CUDA availability if using GPU

2. Video Processing Issues:
   - Verify FFmpeg installation: `ffmpeg -version`
   - Check video format compatibility (MP4, AVI, MOV supported)
   - Ensure sufficient disk space for temporary files

3. Memory Issues:
   - Reduce video resolution if experiencing OOM errors
   - Consider using CPU inference if GPU memory is limited
   - Monitor system resources during processing

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t dancebits-api .
   ```

2. Run the container:
   ```bash
   docker run -d --name dancebits-api \
     -p 8080:8080 \
     -e WANDB_API_KEY=your_key \
     -e WANDB_ORG=your_org \
     -e WANDB_PROJECT=your_project \
     -e WANDB_MODEL_NAME=your_model \
     -e WANDB_MODEL_VERSION=your_version \
     dancebits-api
   ```

## Environment Variables

The following environment variables are required for the application:

- `WANDB_API_KEY`: Weights & Biases API key
- `WANDB_ORG`: Weights & Biases organization name
- `WANDB_PROJECT`: Weights & Biases project name
- `WANDB_MODEL_NAME`: Name of the model to use
- `WANDB_MODEL_VERSION`: Version of the model to use

## API Endpoints

### Predict Dance Segments

```http
POST /predict/
```

Segments a dance video into individual moves.

**Parameters:**
- `video`: Video file (MP4, AVI, or MOV)
- `min_segmentation_prob`: Minimum probability threshold for segmentation (default: 0.5)

**Response:**
```json
{
    "segmented_probs": [...],
    "segmented_percentages": [...]
}
```

### Compare Videos

```http
POST /compare/
```

Calculates similarity score between two dance videos.

**Parameters:**
- `user_video`: User's dance video file
- `teacher_video`: Teacher's reference video file

**Response:**
```json
{
    "similarity_score": float
}
```

## Technical Details

### Video Processing Pipeline

1. **Frame Extraction**: Videos are processed frame by frame using OpenCV
2. **Pose Estimation**: MediaPipe Pose is used to extract 35 bone vectors per frame
3. **Audio Processing**: 
   - Audio is extracted from video using MoviePy
   - Mel spectrogram is generated using Librosa
   - Tempo analysis for beat detection
4. **Model Inference**:
   - Processes both visual (pose) and audio features
   - Returns frame-by-frame segmentation probabilities
5. **Post-processing**:
   - Smoothing of segmentation probabilities
   - Dynamic adjustment based on beat detection
   - Segment identification based on probability thresholds

### Performance Considerations

- The API supports both CPU and GPU inference
- Video processing is optimized for real-time performance
- Temporary files are automatically cleaned up after processing
- CORS is enabled for all origins by default

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
