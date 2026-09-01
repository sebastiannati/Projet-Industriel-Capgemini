import numpy as np 

def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1[:2] - p2[:2])