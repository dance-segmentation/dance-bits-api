import numpy as np

# Function to prepare (reshape) bone vectors from data
def prepare_vectors(data, frames):
    # Copy and reshape the bone_vectors from the data
    vector_pairs = data['bone_vectors'].copy()
    reshaped_vectors = vector_pairs.reshape(frames, 35, 2)  # Assuming 35 bones and 2D vectors
    return reshaped_vectors

# Function to downsample vectors by a given factor
def downsample_vectors(vectors, factor):
    return vectors[::factor]  # Downsample by taking every 'factor'-th frame

# Function to align vectors by removing 'drop_frames' from the beginning and end
def align_vectors(vectors, drop_frames):
    return vectors[drop_frames:-drop_frames, :, :]  # Remove 'drop_frames' from both sides

# Function to process both Cristina's and teaching data
def process_data(cristina_data, teaching_data, downsample_factor=2, drop_frames=21):
    # Step 1: Prepare vectors (reshape)
    vectors_cristina = prepare_vectors(cristina_data, 1004)  # 1004 frames for Cristina's data
    vectors_teaching = prepare_vectors(teaching_data, 1921)  # 1921 frames for teaching data

    # Step 2: Downsample teaching vectors
    vectors_teaching_downsample = downsample_vectors(vectors_teaching, downsample_factor)

    # Step 3: Align Cristina's sets of vectors (drop frames)
    vectors_cristina_aligned = align_vectors(vectors_cristina, drop_frames)
    
    return vectors_cristina_aligned, vectors_teaching_downsample

# Cosine similarity function between two vectors
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

# Function to compute cosine similarities across all frames
def compute_similarity_for_all_frames(bone_vectors_uploaded, bone_vectors_teaching):
    # Ensure both videos have the same number of frames
    if len(bone_vectors_uploaded) != len(bone_vectors_teaching):
        raise ValueError("The two videos must have the same number of frames.")
    
    # List to store cosine similarities for all vectors across all frames
    all_similarities = []

    # Loop through each frame
    for frame_idx in range(len(bone_vectors_uploaded)):
        uploaded_frame = bone_vectors_uploaded[frame_idx]
        teaching_frame = bone_vectors_teaching[frame_idx]

        # Loop through each vector in the frame and compute similarity
        for vec1, vec2 in zip(uploaded_frame, teaching_frame):
            similarity = cosine_similarity(vec1, vec2)
            all_similarities.append(similarity)  # Add similarity to the list

    # Calculate the average similarity across all frames and vectors
    average_similarity = np.mean(all_similarities)

    return average_similarity






    







