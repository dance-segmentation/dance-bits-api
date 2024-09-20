import numpy as np

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

# Example usage:
# Suppose you have two bone vectors
bone_vector1 = [1, 2, 3]
bone_vector2 = [4, 5, 6]

cos_sim = cosine_similarity(bone_vector1, bone_vector2)
print("Cosine Similarity:", cos_sim)

#TODO 
# implement similary_score function that takes two lists of bone vectors and returns the similarity score
# find a best aproach to compare the similarity of two lists of bone vectors
# implement a function that takes a list of bone vectors and returns the average bone vector
# implement a spatial alignment
# implement a temporal alignment
# implement temporal and spatial tolerance