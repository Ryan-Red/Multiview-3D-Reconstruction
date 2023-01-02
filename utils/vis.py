import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cmap = plt.get_cmap('RdBu')

def unskew(T):
    return np.array([T[2,1], T[0,2], T[1,0]]).T


def vis_3d(X):
    """
    Visualize the image and their 2D corresponding points
    """
    colors = cmap(np.linspace(0, 1, len(X)))
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=colors[[i]], s=6)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.show()


def visualize_reprojection(images, pts, points_3d, Rs, Ts, Ks):
    """
    Verify the algorithm by reprojecting 3D points using R, T, and K to I0 and I1.
    Visualization using matplotlib.

    pts contains the locations of the original feature observations in each image.
    points_3d contains the triangulated location of the feature points. We can
    measure the quality of both our reconstructed points and camera motion by "re-projecting"
    the 3D locations in points_3d onto each image and comparing the locations of the 
    re-projection to the original feature observation given in pts. The euclidean distance between
    the re-projection of a feature point and it's original observation is called the "reprojection error".

    This function should return the average re-projection error over all images and all features. If the
    reconstruction is good, then this number should be small. However, this number alone can be difficult
    to interpret, so you should also compare the re-projected features with their original images visually by
    creating a plot of the image with the original features (from pts) and the re-projected features (which you 
    will compute using points_3d) both plotted together. You can also consider how the per-image
    re-projection error varies between the different views. Are there some views for whom the average
    re-projection error is smaller than others?

    Args:
        images: original images
        pts: original matched points
        points_3d: 3d points
        Rs: rotation matrix for each camera
        Ts: translation matrix for each camera
        Ks: intrinsic matrix for each camera

    Return:
        err: the average l2 distance between the projected points and original points.

    Side effects: Should create a plot of the provided images with the true feature
        locations and re-projected feature locations marked together.
    """
    C = len(Rs) # number of cameras
    N = points_3d.shape[0]
    # print(pts.shape)
    P = np.zeros((3,4))
    reproj = np.zeros((2,N,2))
    for i in range(0,C,1):

        
        P[0:3,0:3] = Rs[i]
   
        t = Ts[i]
        P[:,3] = t

        P = Ks[i] @ P
        
        for j in range(0,N,1):
            augmented_3d = np.hstack((points_3d[j,:],np.array([1])))
            # print(augmented_3d)
            augmented_2d = P @ augmented_3d
            reproj[i,j,:] = augmented_2d[0:2]/augmented_2d[2]
            print(augmented_2d)
    # print(reproj.shape)
    vis_2d(images, reproj)



    # raise NotImplementedError


def vis_2d(images, juncs, lines=None):
    """
    Visualize the image and their 2D corresponding points
    """
    cmap = plt.get_cmap('RdBu')
    colors = cmap(np.linspace(0, 1, len(juncs[0])))
    max_col = 5
    # plt.figure(figsize=(24, 12))
    for i, (im, junc) in enumerate(zip(images, juncs)):
        plt.subplot(len(images) // 5 + 1, min(len(images), max_col), i + 1)
        plt.axis([0, im.shape[1], im.shape[0], 0])
        plt.imshow(im)
        for j, jc in enumerate(junc):
            plt.scatter(jc[0], jc[1], c=colors[[j]], s=64)

        if lines is not None:
            for ln in lines[i]:
                plt.plot(
                    [junc[ln[0], 0], junc[ln[1], 0]],
                    [junc[ln[0], 1], junc[ln[1], 1]],
                )

    plt.show()


def vis_2d_lines(images, juncs):
    """
    Visualize the image and their 2D corresponding points
    """

    def to_kp_array(points):
        return [cv2.KeyPoint(point[0], point[1], 1) for point in points]

    for i in range(len(images) - 1):
        outimage = np.zeros(1)
        matches = [cv2.DMatch(j, j, 0.0) for j in range(len(juncs[i]))]
        outimage = cv2.drawMatches(images[i], to_kp_array(juncs[i]),
                                   images[i + 1], to_kp_array(juncs[i + 1]),
                                   matches, outimage)
        cv2.imshow(f"Matches: Frame {i + 1} and frame {i + 2}", outimage)
        cv2.waitKey(0)
