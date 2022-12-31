"""
Starter Code for EECS 106B Spring 2021 Project 3
Author: Haozhi Qi <hqi@berkeley.edu>
"""
import numpy as np
import argparse

from correspondence import match_n_images
from reconstruct import eight_point_algorithm, triangulation, factorization_algorithm

from utils.vis import vis_3d, vis_2d, visualize_reprojection, vis_2d_lines
from utils.dataset import load_city, load_house, load_city_pts, load_city_raw_wireframes


def arg_parse():
    """
    Input Argument for See Project Documentation for usage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', choices=['city', 'house'],
                        help='which dataset to use')
    parser.add_argument('--images', nargs='*',
                        help='which image(s) to use for 3D reconstruction')
    parser.add_argument('--sift', action='store_true',
                        help='whether to use SIFT feature or ground-truth (only applicable for the city dataset)')
    parser.add_argument('--noisy-gt', action='store_true')
    return parser.parse_args()


def main():
    args = arg_parse()
    file_names = sorted(args.images)

    assert args.sift or args.dataset == 'city', \
        'There is no ground-truth correspondence provided in other datasets, ' \
        'you need to implement your own sift-based keypoints matching system'

    # Dataset Loading
    images, intrinsic = load_city(file_names) if args.dataset == 'city' else load_house(file_names)

    if args.dataset == 'city':
        # show the raw wireframe for each of the image
        # note that this function does not filter out T-junction
        # it does not extract common junctions, neither
        juncs, lines, junc_ids = load_city_raw_wireframes(file_names)
        # vis_2d(images, juncs, lines)

    # Finding Correspondence
    if args.sift:
        """
        You need to implement the feature matching process.
        """
        pts = match_n_images(images)
        raise NotImplementedError
    else:
        # For Task 1 and 2, this function returns the ground-truth
        pts = load_city_pts(file_names)

    # 2.1.1 Preliminary
    # at this point, you need to make sure that the extracted keypoints are the same as you expected
    # it is recommended to visualize it before passing it into the reconstruction system
    vis_2d(images, pts)

    # You need to make sure that the extracted keypoints are the same as you expected
    # It is recommended to visualize it before passing it into the reconstruction system

    # This is not for plotting connections between junctions!
    # Instead, it's just another way to visualize feature matches which draws
    # lines between the matched features in the two side-by-side images. Sorry
    # for the confusing function name!
    # If images is an array of more than two images, this visualization will show you 
    # subsequent image pairs one-by-one (first image and second image, then second image and third image,
    # and so on). Just press the space-bar to move to the next image.
    # vis_2d_lines(images, pts)

    # 2.1.2 Eight-Point Algorithm
    # use eight-point algorithm to recover the rotation and translation matrix
    Rs, Ts = eight_point_algorithm(pts[0], pts[1], intrinsic)


    # 2.1.3 Triangulation
    # at this point, you should be able to get four different rotation and translation matrix
    # however, there is only one of them placing all world points in front of the camera (positive depth)
    # find the correct one by reconstruct the 3D position using triangulation
    R, T, pts3d = triangulation(pts[0], pts[1], Rs, Ts, intrinsic)

    # 2.1.4 Visualize 3D points
    # similarly, we also want to make sure that the 3d points are correctly reconstructed
    vis_3d(pts3d)

    # visualize reprojection:
    Rs = [np.eye(3), R]
    Ts = [np.zeros(3), T]
    Ks = [intrinsic, intrinsic]
    err = visualize_reprojection(images[:2], pts[:2], pts3d, Rs, Ts, Ks)

    # For > 2 views only:
    if len(images) > 2:
        # 2.2.1 Factorization Algorithm
        # use factorization algorithm to get the remaining rotation and translation matrix
        Rs, Ts, pts3d = factorization_algorithm(pts, R, T, intrinsic)

        # visualize reprojection:
        Ks = [intrinsic for _ in range(len(images))]
        err = visualize_reprojection(images, pts, pts3d, Rs, Ts, Ks)


if __name__ == '__main__':
    main()
