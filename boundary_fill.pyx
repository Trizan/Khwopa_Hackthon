# boundary_fill.pyx

import numpy as np
np.import_array()

cimport numpy as np

def boundary_fill_cython(np.ndarray[np.uint8_t, ndim=2] img, tuple seed, np.uint8_t fill_color, np.uint8_t boundary_color):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef list stack = [seed]
    cdef set visited = set()
    cdef int x, y, nx, ny, dx, dy
    cdef list boundary_coordinates = []

    while stack:
        x, y = stack.pop()

        if (x, y) in visited:
            continue

        visited.add((x, y))

        if not (0 <= x < w and 0 <= y < h):
            continue

        if img[y, x] == boundary_color:
            boundary_coordinates.append((x, y))  # Append boundary coordinates
            continue

        img[y, x] = fill_color

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            stack.append((nx, ny))

    return img, boundary_coordinates  # Return both the image and the boundary coordinates
