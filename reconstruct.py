import numpy as np


def eight_point_algorithm(pts0, pts1, K):
    """
    Implement the eight-point algorithm

    Args:
        pts0 (np.ndarray): shape (num_matched_points, 2)
        pts1 (np.ndarray): shape (num_matched_points, 2)
        K (np.ndarray): 3x3 intrinsic matrix

    Returns:
        Rs (list): a list of possible rotation matrices
        Ts (list): a list of possible translation matrices
    """
    raise NotImplementedError


def triangulation(pts0, pts1, Rs, Ts, K):
    """
    Implement the triangulation algorithm

    Args:
        pts0 (np.ndarray): shape (num_matched_points, 2)
        pts1 (np.ndarray): shape (num_matched_points, 2)
        Rs (list): a list of rotation matrices (normally 4)
        Ts (list): a list of translation matrices (normally 4)
        K (np.ndarray): 3x3 intrinsic matrices

    Returns:
        R (np.ndarray): a 3x3 matrix specify camera rotation
        T (np.ndarray): a 3x1 vector specify camera translation
        pts3d (np.ndarray): a (num_points, 3) vector specifying the 3D position of each point
    """
    raise NotImplementedError


def factorization_algorithm(pts, R, T, K):
    """
    Factorization algorithm for multiple-view reconstruction.
    (Algorithm 8.1 of MaKSK)

    Args:
        pts (np.ndarray): coordinate of matched points,
            with shape (num_images, num_matched_points, 2)
        R (np.ndarray): recovered rotation matrix from the first two views
        T (np.ndarray): recovered translation matrix from the first two views
        K (np.ndarray): 3x3 intrinsic matrices

    Returns:
        Rs: rotation matrix w.r.t the first view of other views with shape [N_IMAGES-1, 3, 3]
        Ts: translation vector w.r.t the first view of other views with shape [N_IMAGES-1, 3]
        pts3d: a (num_points, 3) vector specifying the refined 3D position of each point (found using
                the converged alpha^j's) in the first camera frame.
    """

    # Initialization:
    # attach the (R, T) of the first camera
    Rs = [np.eye(3), R]
    Ts = [np.zeros(3), T]

    # Compute alpha^j from equation (21)

    # Normalize alpha^j = alpha^j / alpha^1

    # For each of the remaining view i

    # While reprojection error > threshold:
    #     using equation (22) and (23):
    #         compute the eigenvector associated with the smallest singular value of P_i
    #         compute (R_i, T_i)
    #         compute refined alpha using equation (25)
    #     compute the reprojection error

    return np.array(Rs), np.array(Ts), pts3d
