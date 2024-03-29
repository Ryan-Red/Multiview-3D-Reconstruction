import numpy as np
from rpy_from_dcm import rpy_from_dcm

cos = np.cos
sin = np.sin
pi = np.pi

def skew(V):
    return np.array([[    0, -V[2],  V[1]],
                     [ V[2],     0, -V[0]],
                     [-V[1],  V[0],  0]])

def unskew(T):
    return np.array([T[2,1], T[0,2], T[1,0]]).T

def R_z(theta):
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta),  cos(theta), 0],
                     [         0,           0, 1]])

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
    N = pts0.shape[0] #Number of points

    K_inv = np.linalg.inv(K)

    A = np.zeros((N,9))
    #Generate the A matrix to get the Essential matrix
    for i in range(0,N,1):
        # print(np.hstack([pts0[i,:],1]))
        x_1_aug = K_inv @ np.hstack([pts0[i,:],1])
        # print(x_1_aug)
        x_2_aug = K_inv @ np.hstack([pts1[i,:],1])
        x_1, y_1, _ = x_1_aug
        x_2, y_2, _ =  x_2_aug
        A[i,:] = np.array([x_1 * x_2, x_1 * y_2, x_1, y_1 * x_2, y_1 * y_2, y_1, x_2, y_2, 1])
    
    U, S, Vt = np.linalg.svd(A)
    
    #Get the essential Matrix
    E_s = Vt[-1,:]
    E = np.reshape(E_s, (3,3))

    U, S, Vt = np.linalg.svd(E)

    Rs = []
    Ts = []

    #Calculate the Rotation matrix and translation vector based on E
    combinations = [(pi/2, pi/2), (pi/2, -pi/2), (-pi/2, pi/2), (-pi/2, -pi/2)]
    for combos in combinations:
        print(combos)
        T =  U @ R_z(combos[0]) @ np.diag([1,1,0]) @ U.T
        R =  U @ R_z(combos[1]).T @ Vt
        print(R)
        print(unskew(T))
        print("-----"*10)
        Ts.append(unskew(T))
        Rs.append(R)


    return Rs, Ts



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

    N = pts0.shape[0]
    A = np.zeros((4,3))
    b = np.zeros((4,1))

    list3Dpoints = []
    pts3d = np.zeros((N,3))
    P0 = np.zeros((3,4))
    P1 = np.zeros((3,4))

    P0[0:3,0:3] = np.eye(3)

    P0 = K @ P0


    passed = [0,0,0,0]
    for j in range(0,4,1):
        R = Rs[j]
        t = Ts[j]
        
        P1[0:3,0:3] = R
        P1[:,3] = t
        P1 = K @ P1 
        
        

        for i in range(0,N,1):
            x_i, y_i = pts0[i,:]
            
            A[0,:] = np.array([(P0[0,0] - P0[2,0] * x_i), (P0[0,1] - P0[2,1] * x_i), (P0[0,2] - P0[2,2] * x_i)])
            A[1,:] = np.array([(P0[1,0] - P0[2,0] * y_i), (P0[1,1] - P0[2,1] * y_i), (P0[1,2] - P0[2,2] * y_i)])
            b[0] = P0[2,3] * x_i - P0[0,3]
            b[1] = P0[2,3] * y_i - P0[1,3]

            x_i, y_i = pts1[i,:]
            A[2,:] = np.array([(P1[0,0] - P1[2,0] * x_i), (P1[0,1] - P1[2,1] * x_i), (P1[0,2] - P1[2,2] * x_i)])
            A[3,:] = np.array([(P1[1,0] - P1[2,0] * y_i), (P1[1,1] - P1[2,1] * y_i), (P1[1,2] - P1[2,2] * y_i)])
            b[2] = P1[2,3] * x_i - P1[0,3]
            b[3] = P1[2,3] * y_i - P1[1,3]
        
        
            A_pseudo_inv = np.linalg.inv(A.T @ A) @ A.T
            pts = A_pseudo_inv @ b
            
         
            if(pts[2] < 0): # Z < 0, means this R and t are wrong.
                passed[j] += -1
                break

            pts3d[i,:] = pts[0:3].T
        list3Dpoints.append(pts3d)

    maxIdx = 0

    for i in range(0,4,1):
        if passed[i] == 0:
            maxIdx = i

    print( Rs[maxIdx], Ts[maxIdx])

    return Rs[maxIdx], Ts[maxIdx], list3Dpoints[maxIdx]



def projection(pts0, pts1, R,t,K):

    N = pts0.shape[0]
    A = np.zeros((4,3))
    b = np.zeros((4,1))

    pts3d = np.zeros((N,3))
    P0 = np.zeros((3,4))
    P1 = np.zeros((3,4))

    P0[0:3,0:3] = np.eye(3)

    P0 = K @ P0
    
    P1[0:3,0:3] = R
    P1[:,3] = t
    P1 = K @ P1 
    
    

    for i in range(0,N,1):
        x_i, y_i = pts0[i,:]
        
        A[0,:] = np.array([(P0[0,0] - P0[2,0] * x_i), (P0[0,1] - P0[2,1] * x_i), (P0[0,2] - P0[2,2] * x_i)])
        A[1,:] = np.array([(P0[1,0] - P0[2,0] * y_i), (P0[1,1] - P0[2,1] * y_i), (P0[1,2] - P0[2,2] * y_i)])
        b[0] = P0[2,3] * x_i - P0[0,3]
        b[1] = P0[2,3] * y_i - P0[1,3]

        x_i, y_i = pts1[i,:]
        A[2,:] = np.array([(P1[0,0] - P1[2,0] * x_i), (P1[0,1] - P1[2,1] * x_i), (P1[0,2] - P1[2,2] * x_i)])
        A[3,:] = np.array([(P1[1,0] - P1[2,0] * y_i), (P1[1,1] - P1[2,1] * y_i), (P1[1,2] - P1[2,2] * y_i)])
        b[2] = P1[2,3] * x_i - P1[0,3]
        b[3] = P1[2,3] * y_i - P1[1,3]
    
    
        A_pseudo_inv = np.linalg.inv(A.T @ A) @ A.T
        pts = A_pseudo_inv @ b
        pts3d[i,:] = pts[0:3].T

    return pts3d
    



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

    # pts3d = np.zeros((pts.shape[0],pts.shape[1],3))

    # Compute alpha^j from equation (21)

    #Initial Estimate of alpha
    alpha = np.zeros((pts.shape[1]))

    for j in range(0,pts.shape[1]):
        x_1 = np.hstack([pts[0,j,:],1])
        x_2 = np.hstack([pts[1,j,:],1])
        x_2_hat = skew(x_2)
        x_2_hat_t = x_2_hat @ T

        alpha[j] = - ((x_2_hat_t).T @ x_2_hat @ R @ x_1) / np.square(np.linalg.norm(x_2_hat_t))


    alpha = alpha/alpha[0]    


    print(alpha)
    for w in range(0,20):
        for i in range(2, pts.shape[0]): #For each camera
            A = np.array([])
            for j in range(0,pts.shape[1]): #For each point

                x_1_1 = np.hstack([pts[0,j,:],1])
                x_1_i = np.hstack([pts[i,j,:],1])
                x_1_i_skew = skew(x_1_i)

                kron = np.kron(x_1_1.T, x_1_i_skew)

                if(j == 0):
                    A = np.hstack([kron, alpha[j] * x_1_i_skew])
                else:
                    A = np.vstack([A,np.hstack([kron, alpha[j] * x_1_i_skew])])

            U, S, Vt = np.linalg.svd(A)
        
            #Get the essential Matrix
            R_s_t_s = Vt[-1,:]
            R_s = R_s_t_s[0:9]
            T_s = R_s_t_s[9:]
            R_i_bar = np.reshape(R_s, (3,3))

            Ui, Si, ViT = np.linalg.svd(R_i_bar)

            prod = Ui @ ViT
            sign = np.sign(np.linalg.det(Ui @ ViT))
            R_i = sign * prod

            T_i = sign/np.power(Si,1/3) * T_s



            Rs.append(R_i)
            Ts.append(T_i)

            #update alpha
            for j in range(2,pts.shape[1]):
                x_j_1 = np.hstack([pts[0,j,:],1])
                x_j_i = np.hstack([pts[i,j,:],1])
                x_j_i_hat = skew(x_j_i)
                x_j_i_hat_t = x_j_i_hat @ T

                # alpha[j] = - (x_j_i_hat_t.T @ x_j_i_hat @ R @ x_j_1) / np.square(np.linalg.norm(x_j_i_hat_t)) + alpha[j]

            # alpha = alpha/alpha[0]

            #Get the reprojection error:

            real_world_points = projection(pts[i - 1], pts[i], Rs[i], Ts[i], K)

            
        
            N = real_world_points.shape[0]
            P = np.zeros((3,4))
            reproj = np.zeros((N,2))
            reproj_error = 0 # total reprojection error
    
                
            P[0:3,0:3] = Rs[i]

            t = Ts[i]
            P[:,3] = t

            P = K @ P
            

            for j in range(0,N,1):
                augmented_3d = np.hstack((real_world_points[j,:],np.array([1])))
                # print(augmented_3d)
                augmented_2d = P @ augmented_3d
                reproj[j,:] = augmented_2d[0:2]/augmented_2d[2]
                reproj_error += np.linalg.norm(reproj[j,:] - pts[i-1,j,:]) 

            print(reproj_error)
    pts3d = real_world_points
    print(A.shape)

    # Normalize alpha^j = alpha^j / alpha^1

    # For each of the remaining view i

    # While reprojection error > threshold:
    #     using equation (22) and (23):
    #         compute the eigenvector associated with the smallest singular value of P_i
    #         compute (R_i, T_i)
    #         compute refined alpha using equation (25)
    #     compute the reprojection error

    return np.array(Rs), np.array(Ts), pts3d
