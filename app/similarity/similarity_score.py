import numpy as np

# Function to prepare (reshape) bone vectors from data


def reshape_vectors(bvs_data, frames):
    # Copy and reshape the bone_vectors from the data
    reshaped_vectors = bvs_data.reshape(
        frames, 35, 2)  # Assuming 35 bones and 2D vectors
    return reshaped_vectors


def downsample_vectors(vectors, factor):
    """
    Function to downsample vectors by a given factor
    """
    return vectors[::factor]  # Downsample by taking every 'factor'-th frame


def align_vectors(vectors, drop_frames_start, drop_frames_end):
    """
    Function to align vectors by removing 'drop_frames' from both sides
    """
    # Remove 'drop_frames' from both sides
    return vectors[drop_frames_start:-drop_frames_end, :, :]

# Function to process both Cristina's and teaching data


def process_data(user_bvs_data, teacher_bvs_data, fps_user, fps_teacher):
    """
    Reshapes the bone vectors from 2D to 3D array and matches the sampling rate of the two videos 
    by downsampling the video with higher FPS. The videos need to have the same length.

    Args:
    user_bvs_data (dict): Dictionary containing the bone vectors data for the user.
    teacher_bvs_data (dict): Dictionary containing the bone vectors data for the teacher.
    fps_user (int): FPS of the user's video.
    fps_teacher (int): FPS of the teacher's video.

    Returns:
    tuple: Tuple containing the processed bone vectors for the user and teacher.
    """
    # Step 1: Prepare vectors (reshape)
    vectors_user = reshape_vectors(user_bvs_data, len(user_bvs_data))
    vectors_teacher = reshape_vectors(teacher_bvs_data, len(teacher_bvs_data))

    # Step 2: Compute downsample_factor
    downsample_factor = int(max(
        fps_teacher, fps_user) // min(fps_teacher, fps_user))

    if downsample_factor != 1:
        print(f"Downsampling factor: {downsample_factor}")
        if fps_teacher > fps_user:
            print("Downsampling teacher vectors")
            vectors_teacher = downsample_vectors(
                vectors_teacher, downsample_factor)
        else:
            print("Downsampling user vectors")
            vectors_user = downsample_vectors(
                vectors_user, downsample_factor)

    # Step 3: Align vectors
    # Calculate the number of frames to drop from the start and end
    drop_frames = abs(len(vectors_user) - len(vectors_teacher))
    drop_frames_start = drop_frames // 2
    drop_frames_end = drop_frames - drop_frames_start

    # Align the vectors by removing frames from both sides
    if len(vectors_user) > len(vectors_teacher):
        print(
            f"Aligning user vectors. Dropping {drop_frames_start} frames from start and {drop_frames_end} frames from end.")
        vectors_user = align_vectors(
            vectors_user, drop_frames_start, drop_frames_end)
    else:
        print(
            f"Aligning teacher vectors. Dropping {drop_frames_start} frames from start and {drop_frames_end} frames from end.")
        vectors_teacher = align_vectors(
            vectors_teacher, drop_frames_start, drop_frames_end)

    return vectors_user, vectors_teacher


def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
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


def compute_similarity_for_all_frames(bone_vectors_uploaded, bone_vectors_teaching):
    """
    Computes the average cosine similarity between the bone vectors of the two videos across all frames.
    """
    # Ensure both videos have the same number of frames
    if len(bone_vectors_uploaded) != len(bone_vectors_teaching):
        raise ValueError("The two videos must have the same number of frames.")

    # List to store cosine similarities for all vectors across all frames
    all_similarities = []

    # Loop through each frame
    for uploaded_frame, teaching_frame in zip(bone_vectors_uploaded, bone_vectors_teaching):
        # Loop through each vector in the frame and compute similarity
        for vec1, vec2 in zip(uploaded_frame, teaching_frame):
            similarity = cosine_similarity(vec1, vec2)
            all_similarities.append(similarity)  # Add similarity to the list

    # Calculate the average similarity across all frames and vectors
    average_similarity = np.mean(all_similarities)

    return average_similarity
