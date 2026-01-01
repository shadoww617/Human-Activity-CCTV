import numpy as np
import cv2
import random

canvas = None
tracks = {}
colors = {}

def ensure_canvas(h, w):
    global canvas
    if canvas is None or canvas.shape[0] != h or canvas.shape[1] != w:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

def get_color(pid):
    if pid not in colors:
        colors[pid] = (
            random.randint(50,255),
            random.randint(50,255),
            random.randint(50,255)
        )
    return colors[pid]

def update_trajectory(pid, point):
    global canvas

    if pid not in tracks:
        tracks[pid] = []

    tracks[pid].append(point)

    if len(tracks[pid]) > 30:
        tracks[pid].pop(0)

    color = get_color(pid)

    for i in range(1, len(tracks[pid])):
        cv2.line(
            canvas,
            tracks[pid][i-1],
            tracks[pid][i],
            color,
            3
        )

def get_canvas():
    if canvas is None:
        return np.zeros((300,300,3), dtype=np.uint8)
    return canvas
