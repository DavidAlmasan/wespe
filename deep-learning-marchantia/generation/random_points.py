import cv2
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import scipy.spatial as spatial

from PIL import Image

def voronoi_finite_polygons_2d(vor, index, radius=None):
    """Reconstruct infinite Voronoi regions in a
    2D diagram to finite regions.
    Source:
    [https://stackoverflow.com/a/20678647/1595060](https://stackoverflow.com/a/20678647/1595060)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices): # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        if not p1 in all_ridges:
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

seed = 1234
np.random.seed(seed)

img_height = 128
img_width = 128

number_of_cells = 50

number_images = 1000

for i in range(number_images):

    gt_img = np.zeros((img_height,img_width,3), np.uint8)
    gray_img = np.zeros((img_height,img_width,3), np.uint8)

    # x_list = np.random.randint(img_height//4, 3*img_height//4, number_of_cells)
    # y_list = np.random.randint(img_height//4, 3*img_height//4, number_of_cells)

    x_list = np.random.randint(0, img_height, number_of_cells)
    y_list = np.random.randint(0, img_height, number_of_cells)

    # indices = [i for i in range(x_list.size) if (x_list[i]-(img_height // 2))^2 + (y_list[i]-(img_height // 2))^2 < 50^2]
    # img[y_list[indices], x_list[indices]] = (255, 255, 255)

    #gt_img[y_list, x_list] = (255, 255, 255) # access by [row, column]
    # for c in range(len(x_list)):
    #     rand_b = np.random.randint(0, 255)
    #     cv2.circle(gt_img, (x_list[c], y_list[c]), 1, (rand_b, rand_b, rand_b), thickness=2)

    vor = spatial.Voronoi(np.c_[x_list, y_list])

    points = vor.vertices

    regions, vertices = voronoi_finite_polygons_2d(vor, i)

    for region in regions:
        cell_pts = vertices[region].reshape(-1,1,2)
        img_pts = np.round(cell_pts.astype(int))
        # rand_brightness = np.random.randint(0, 255)
        rand_brightness = np.random.randint(0, 63)
        cv2.polylines(gray_img, [img_pts], True, (rand_brightness, rand_brightness, rand_brightness), thickness=1)
        # cv2.polylines(gt_img, [img_pts], True, (255, 255, 255), thickness=1)


    # cv2.imwrite('gt_images/{:03d}_gt.png'.format(i), gt_img)
    # cv2.imwrite('gt_images/{:03d}_gray.png'.format(i), gray_img)
    cv2.imwrite('varying_gray_images/{:03d}_gray_0_63.png'.format(i), gray_img)

    #cv2.imwrite('outline.png', gt_img)
