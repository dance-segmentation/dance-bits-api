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


#TODO 
# find a best aproach to compare the similarity of two lists of bone vectors
# implement a spatial alignment
# implement a temporal alignment
# implement temporal and spatial tolerance