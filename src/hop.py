import numpy as np


def calculate_angle(x, y):
    """
    Calculates the angle between two vectors.

    Parameters:
    x (float): first vector.
    y (float): second vector.

    Returns:
    float: The angle between the two vectors in degrees.
    """
    return (-np.abs(np.arctan2(y, x) * 180/np.pi) + 180)

def hop(head):
    """
    Performs the hop operation.

    Parameters:
    head (list): A list containing two vectors.

    Returns:
    tuple: A tuple containing the angles between the vectors.
    """
    x = np.array(head[0])[0] - np.array(head[1])[0]
    y = np.array(head[0])[1] - np.array(head[1])[1]
    z = np.array(head[0])[2] - np.array(head[1])[2]
    angle_hb = calculate_angle(y, x)
    angle_gd = calculate_angle(z, y)
    return angle_hb, angle_gd
