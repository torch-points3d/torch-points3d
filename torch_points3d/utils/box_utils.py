import torch
import numpy as np
from scipy.spatial import ConvexHull

from .geometry import euler_angles_to_rotation_matrix


def box_corners_from_param(box_size, heading_angle, center):
    """ Generates box corners from a parameterised box.
    box_size is array(size_x,size_y,size_z), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box corners
    """
    R = euler_angles_to_rotation_matrix(torch.tensor([0.0, 0.0, float(heading_angle)]))
    if torch.is_tensor(box_size):
        box_size = box_size.float()
    l, w, h = box_size
    x_corners = torch.tensor([-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2])
    y_corners = torch.tensor([-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2])
    z_corners = torch.tensor([-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2])
    corners_3d = R @ torch.stack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = corners_3d.T
    return corners_3d


def nms_samecls(boxes, classes, scores, overlap_threshold=0.25):
    """ Returns the list of boxes that are kept after nms.
    A box is suppressed only if it overlaps with
    another box of the same class that has a higher score

    Parameters
    ----------
    boxes : [num_boxes, 6]
        xmin, ymin, zmin, xmax, ymax, zmax
    classes : [num_shapes]
        Class of each box
    scores : [num_shapes,]
        score of each box
    overlap_threshold : float, optional
        [description], by default 0.25
    """
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if torch.is_tensor(classes):
        classes = classes.cpu().numpy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    I = np.argsort(scores)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[: last - 1]])
        yy1 = np.maximum(y1[i], y1[I[: last - 1]])
        zz1 = np.maximum(z1[i], z1[I[: last - 1]])
        xx2 = np.minimum(x2[i], x2[I[: last - 1]])
        yy2 = np.minimum(y2[i], y2[I[: last - 1]])
        zz2 = np.minimum(z2[i], z2[I[: last - 1]])
        cls1 = classes[i]
        cls2 = classes[I[: last - 1]]

        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)

        inter = l * w * h
        o = inter / (area[i] + area[I[: last - 1]] - inter)
        o = o * (cls1 == cls2)

        I = np.delete(I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0])))

    return pick


def box3d_iou(corners1, corners2):
    """ Compute 3D bounding box IoU.

    Input:
        corners1: array (8,3), assume up direction is Z
        corners2: array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU
    """
    # corner points are in counter clockwise order
    assert corners1.shape == (8, 3)
    assert corners2.shape == (8, 3)
    rect1 = np.asarray([(corners1[i, 0], corners1[i, 1]) for i in range(4)])
    rect2 = np.asarray([(corners2[i, 0], corners2[i, 1]) for i in range(4)])
    inter_area = intersection_area(rect1, rect2)
    z_min = max(corners1[0, 2], corners2[0, 2])
    z_max = min(corners1[4, 2], corners2[4, 2])
    inter_vol = inter_area * max(0.0, z_max - z_min)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou


def box3d_vol(corners):
    """ corners: (8,3). No order required"""
    corners = np.asarray(corners)
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def intersection_area(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return intersection volume
    """
    assert len(p1[0]) == 2 and len(p2[0]) == 2
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return hull_inter.volume
    else:
        return 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


################################################################################################
# Intersection area without scipy. Could be used with numba
################################################################################################


def intersection_area_noscipy(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return intersection volume
    """
    assert len(p1[0]) == 2 and len(p2[0]) == 2
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = np.asarray(convex_hull_graham(inter_p))
        area = polygon_area(hull_inter[:, 0], hull_inter[:, 1])
        return area
    else:
        return 0.0


# Function to know if we have a CCW turn
def RightTurn(p1, p2, p3):
    if (p3[1] - p1[1]) * (p2[0] - p1[0]) >= (p2[1] - p1[1]) * (p3[0] - p1[0]):
        return False
    return True


# Main algorithm:
def convex_hull_graham(P):
    P.sort()  # Sort the set of points
    L_upper = [P[0], P[1]]  # Initialize upper part
    # Compute the upper part of the hull
    for i in range(2, len(P)):
        L_upper.append(P[i])
        while len(L_upper) > 2 and not RightTurn(L_upper[-1], L_upper[-2], L_upper[-3]):
            del L_upper[-2]
    L_lower = [P[-1], P[-2]]  # Initialize the lower part
    # Compute the lower part of the hull
    for i in range(len(P) - 3, -1, -1):
        L_lower.append(P[i])
        while len(L_lower) > 2 and not RightTurn(L_lower[-1], L_lower[-2], L_lower[-3]):
            del L_lower[-2]
    del L_lower[0]
    del L_lower[-1]
    L = L_upper + L_lower  # Build the full hull
    return L


def polygon_area(x, y):
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)
