import numpy as np
import cv2

def cosine_similarity(vec1, vec2):
    # Ensure the vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Compute the dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes (norms) of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity

def average_cosine_similarity(list1, list2):
    # Ensure both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same number of vectors.")
    
    # Initialize a variable to store the sum of similarities
    total_similarity = 0
    
    # Loop over the vectors in both lists
    for vec1, vec2 in zip(list1, list2):
        total_similarity += cosine_similarity(vec1, vec2)
    
    # Calculate the average similarity
    average_similarity = total_similarity / len(list1)
    
    return average_similarity

# Function to apply the cosine similarity computation across all frames of the video
def compute_similarity_for_all_frames(bone_vectors_uploaded, bone_vectors_teaching):
    # Ensure both videos have the same number of frames
    if len(bone_vectors_uploaded) != len(bone_vectors_teaching):
        raise ValueError("The two videos must have the same number of frames.")
    
    frame_similarities = []
    
    # Loop through each frame and compute the average similarity for the frame
    for frame_idx in range(len(bone_vectors_uploaded)):
        uploaded_frame = bone_vectors_uploaded[frame_idx]
        teaching_frame = bone_vectors_teaching[frame_idx]
        
        # Compute average cosine similarity for the frame
        frame_similarity = average_cosine_similarity(uploaded_frame, teaching_frame)
        frame_similarities.append(frame_similarity)
    
    # Compute the overall average similarity across all frames
    overall_similarity = np.mean(frame_similarities)
    
    return frame_similarities, overall_similarity

def get_fps(video_path1, video_path2):
    # Open the first video file
    video1 = cv2.VideoCapture(video_path1)
    if not video1.isOpened():
        raise ValueError(f"Could not open video file: {video_path1}")
    
    # Open the second video file
    video2 = cv2.VideoCapture(video_path2)
    if not video2.isOpened():
        raise ValueError(f"Could not open video file: {video_path2}")
    
    # Get the fps of each video
    fps1 = video1.get(cv2.CAP_PROP_FPS)
    fps2 = video2.get(cv2.CAP_PROP_FPS)
    
    # Release the video files
    video1.release()
    video2.release()
    
    return fps1, fps2




#TODO 
# based on the ratio of two fps values, downsample the list (with bone vectors) with higher fps

# find a best aproach to compare the similarity of two lists of bone vectors
# implement a spatial alignment
# implement a temporal alignment
# implement temporal and spatial tolerance