"""Default simple_3dviz scene setup"""

ORTHOGRAPHIC_PROJECTION_SCENE = {
    "up_vector":       (0, 0, -1),
    "background":      (0, 0, 0, 1),
    "camera_target":   (0, 0, 0),
    "camera_position": (0, 4, 0),
    "window_size":     (256, 256),
}

SIDEVIEW_SCENE = {
    "up_vector":       (0, 1, 0),
    "background":      (1, 1, 1, 1),
    "camera_target":   (0, 0, 0),
    "camera_position": (8, 6, 8),
    "size":            (1024, 1024),
}
