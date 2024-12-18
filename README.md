# DanceBits API: ML-powered automated choreography video segmentation. 

Backend implementing a multimodal AI model in PyTorch and serving it using FastAPI and Docker to automatically identify and label dance moves in videos to create an interactive learning platform. Features: video preprocessing, pose estimation, audio processing, and multimodal segmentation model.

## Features

- Advanced pose estimation and motion feature extraction
- Audio feature extraction for enhanced move detection
- Real-time multimodal dance move segmentation model
- User-friendly learning interface with customizable speeds and segment sizes
- Side-by-side webcam/video option with recording functionality
- Similarity score calculation for comparing dance performances
  
## Installation


1. Clone this repository:
   ```
   git clone https://github.com/your-username/dance-bits-api.git
   cd dancebits
   ```

2. Create a Conda environment and activate it:
   ```
   conda create --name dance-bits-api python
   conda activate dance-bits-api
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: You can also install via Conda, but some packages may not available:
   ```
   conda install --file requirements.txt
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
