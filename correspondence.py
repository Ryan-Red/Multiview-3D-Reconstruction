import cv2
import numpy as np


class FeatureTrack:
    """
    A convenience data structure that may help you in implementing match_n_images.
    Here, we are considering the following scenario.

    We assume we are processing a sequence of images indexed 0, 1, ..., n.
    In each image, we have extracted features kp0, kp1, ..., kpn. Each kpi is an array of
    shape (num_features_i, 2), containing the location of the num_features_i extracted keypoints
    (a possibly different number in each image).

    Then, we perform pairwise feature matching (using the match_two_images function) between consecutive
    image frames. Note that whenever two image features are matched, we assume that they belong
    to the same point in 3D space.

    Now, assume that feature f1 in image 1 is matched with feature f2 in image 2, which is
    further matched with feature f3 in image 3. Then, these three feature observations
    are assumed to belong to the same feature in 3D space. The sequence of observations
    [f1, f2, f3] is called a "feature track", starting from frame 1 and ending with frame 3.
    In general, a feature track is a sequence of observations [f1, f2, ..., fn] such that
    f1 <-> f2, f2 <-> f3, ..., fn-1 <-> fn are pairwise matches. Then the features f1, ..., fn
    are assumed to belong to the same feature in 3D space.

    When we do n-view reconstruction, we wish to find features that are visible in all n views.
    This is the task at hand in implementing match_n_images. In other words, we seek feature tracks
    of length n.

    This class implements a feature track data structure. It keeps track of the index of the
    starting frame of the track (the first image where the track was visible), the index of the
    final frame of the track, and the keypoint indices of the feature in each image along the way.

    Note that a feature track should be visible in consecutive views. If a feature goes out of
    view of the camera and then returns, it is considered a new feature track.
    """
    def __init__(self, img_index, kp_index):
        """
        Initialize a new feature track with a single feature observation. Feture kp_index
        in image img_index.
        """

        # Sequence index of the starting frame of the track.
        self.start_idx = img_index
        # Sequence index of the final frame of the track.
        self.end_idx = img_index
        # A list of keypoint indices of the keypoint corresponding to this feature track
        # in each image between start_idx and end_idx (inclusive).
        self.kp_indices = [kp_index]

    def size(self):
        return self.end_idx - self.start_idx + 1

    def add_frame(self, kp_index):
        """
        Add a new feature observation to this track. This function should only be
        called if the newly observed feature is from frame self.end_idx + 1.
        Recall that a feature track corresponds to a feature that is visible
        in a contiguous sequence of images in the image sequence.
        """
        self.end_idx += 1
        self.kp_indices.append(kp_index)

    def last_feature(self):
        """
        Returns the frame index of the last image in which this feature was observed
        along with the keypoint index of the last observation of this feature.
        """
        return self.end_idx, self.kp_indices[-1]


def match_two_images(kp1, des1, kp2, des2):
    """
    Returns a set of good matches given the keypoints and respective descriptors of two images.

    Args:
    kp1: list of cv2.KeyPoint objects describing the locations of the extracted keypoints
            in the first image.
    des1: array of shape (len(kp1), descriptor_length), containing the descriptor vector
            of each keypoint. So, des1[i] is the descriptor of keypoint kp1[i].
    kp2: list of cv2.KeyPoint objects describing the locations of the extracted keypoints
            in the second image.
    des2: array of shape (len(kp2), descriptor_length), containing the descriptor vector
            of each keypoint. So, des2[i] is the descriptor of keypoint kp2[i].

    Returns:
    matches: integer array of shape (num_matches, 2). [i, j] being an entry in matches
        implies that kp1[i] is a good match for kp2[j].

    cv2.FlannBasedMatcher, cv2.knnMatch may be useful. Note that you should also do outlier
    rejection using RANSAC in this function. You may use cv2.findFundamentalMat or
    cv2.findHomography for outlier rejection. Example calls to those functions:

    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1, confidence=0.995)
    retval, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1, confidence=0.995)

    Look up the opencv documentation for the exact meaning of the arguments and return values of the above.
    The choice between homography and fundamental matrix and the parameters of the RANSAC algorithm
    (such as ransacReprojThreshold (pixels) and confidence (a probability)) are left up to you as design variables.
    """

    # Step 1: Find matches of keypoint features usng nearest-neighbours and Lowe's matching ratio test.

    # Step 2: Using RANSAC to filter out the outliers

    raise NotImplementedError


def extract_features(img, extractor):
    """
    Returns keypoints and descriptors extracted from the input image.

    Args:
    img: The image from which features should be extracted.
    extractor: an instance of cv2.Feature2D. For instance, 
                sift = cv2.SIFT_create() creates a SIFT extractor.

    Returns:
    kp: list of cv2.KeyPoint objects describing the locations of the extracted keypoints
        in the image.
    des: array of shape (n_keypoints, descriptor_length), containing the descriptor vector
        of each keypoint. So, des[i] is the descriptor of keypoint kp[i].

    Hint: the detectAndCompute function of the extractor may be useful.
    """
    raise NotImplementedError


def kp_to_array(kp_array):
    """
    Converts an array of cv2.KeyPoint objects into a numpy array of shape (N, 2) where
    each entry is the (x, y) location of the keypoint.
    """
    return np.asarray([kp.pt for kp in kp_array])


def match_n_images(images):
    """
    Given a list of input images, this function extracts keypoint features from every image
    and aims to find features that represent points in 3D space visible in all the images.
    In other words, this function aims to extract feature tracks of length n from the input images.

    The output should be locations of the matched features in the format required below.

    Args:
        images (list): a list of images; each image is a (H x W x C) array.

    Returns:
        points (np.ndarray): an array of corresponding points with shape: (num_image, num_matched_points, 2).
                points[i, j, :] is the observation of jth feature track in the ith image as an (x, y) location
                in image coordinates.
    """
    raise NotImplementedError

    # Step 1: Initialize feature extractor.
    sift = cv2.SIFT_create()

    # Step 2: Extract features from first image.
    prev_kp_cv, prev_des = extract_features(images[0], sift)
    prev_kp_arr = kp_to_array(prev_kp_cv)

    # Step 3: Compute feature tracks starting from first frame.
    # Keep only tracks that are length n (i.e. that are visible in the entire image sequence).
    points = None
    return points

def match_wireframes(images, pts, pt_ids, lines):
    """
    EXTRA CREDIT

    Given wireframe detections between two images, returns point correspondences between junctions.

    Args:
        images: list of two images.
        pts: list of two arrays of shape (n1, 2) and (n2, 2) respectively, where n1 is the number of
            junctions in image 1 and n2 is the number of junctions in image 2. Each entry is the (x,y) location
            of a junction in image coordinates.
        pt_ids: list of two arrays of shape (n1,) and (n2,). Specifies an integer ID for each junction.
        lines: two integer arrays of shape (l1, 2) and (l2, 3) respectively, where l1 is the number of
            lines in the wireframe of image 1 and l2 is the number of lines in the wireframe of image 2.
            Each entry is a pair [i, j], which implies that the keypoint with ID i and the keypoint with ID j
            are connected by a line.

    Returns:
    points (np.ndarray): an array of corresponding points with shape: (2, num_matched_points, 2).
                points[i, j, :] is the observation of jth matched feature in the ith image as an (x, y) location
                in image coordinates.
    """
    pass
