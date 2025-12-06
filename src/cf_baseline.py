import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def train_cf(interactions):
    matrix = pd.crosstab(interactions.user, interactions.item)
    sim = cosine_similarity(matrix.T)
    return matrix, sim
