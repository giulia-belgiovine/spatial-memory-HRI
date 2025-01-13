import numpy as np
import cv2
from scipy.spatial.distance import cdist
from motpy.core import Track


def draw_track(img, track: Track, random_color: bool = True, fallback_color=(200, 20, 20)):
    # if track.label != "unknown":
    #     color = (0, 255, 0)
    #     img = draw_text(img, str(track.label), above_box=track.box, color=(0, 255, 0))
    #
    # else:
    color_dict = {'speaker': (0, 255, 0), 'addressee': (255, 0, 0), 'attendee': (0, 0, 255)}
    color_box = color_dict.get(track.status)
    img = draw_text(img, track.status + "- " + track.name, above_box=track.box, color=color_box)
    img = draw_rectangle(img, track.box, color=color_box, thickness=2)

    return img


def draw_text(img, text, above_box, color=(255, 255, 255)):
    tl_pt = (int(above_box[0]), int(above_box[1]) - 7)
    cv2.putText(img, text, tl_pt,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=1)
    return img


def draw_rectangle(img, box, color, thickness: int = 3):
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(
        box[2]), int(box[3])), color, thickness)
    return img


def calculate_iou(bboxes1, bboxes2, dim: int = 2):
    """ expected bboxes size: (-1, 2*dim) """
    bboxes1 = np.array(bboxes1).reshape((-1, dim * 2))
    bboxes2 = np.array(bboxes2).reshape((-1, dim * 2))

    coords_b1 = np.split(bboxes1, 2 * dim, axis=1)
    coords_b2 = np.split(bboxes2, 2 * dim, axis=1)

    coords_tl, coords_br = dict(), dict()  # tl and br points
    val_inter, val_b1, val_b2 = 1.0, 1.0, 1.0
    for d in range(dim):
        coords_tl[d] = np.maximum(coords_b1[d], np.transpose(coords_b2[d]))
        coords_br[d] = np.minimum(coords_b1[d + dim], np.transpose(coords_b2[d + dim]))

        val_inter *= np.maximum(coords_br[d] - coords_tl[d], 0)
        val_b1 *= coords_b1[d + dim] - coords_b1[d]
        val_b2 *= coords_b2[d + dim] - coords_b2[d]

    iou = val_inter / (val_b1 + np.transpose(val_b2) - val_inter)
    return iou


def angular_similarity(vectors1, vectors2):
    sim = 1 - cdist(vectors1, vectors2, 'cosine') / 2  # kept in range <0,1>
    return sim


def face_alignement(face_coord, frame, margin=15):
    """
    Preprocess tha face with standard face alignment preprocessing
    :param coord_faces: list of faces' coordinates
    :param frame: rgb image
    :return: list face aligned image
    """
    xmin = int(face_coord[0]) - margin
    ymin = int(face_coord[1]) - margin
    xmax = int(face_coord[2]) + margin
    ymax = int(face_coord[3]) + margin

    face = get_ROI(xmin, ymin, xmax, ymax, frame)
    return face


def get_ROI(x1, y1, x2, y2, frame):
    x1 = x1 if x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0

    x2 = x2 if x2 <= frame.shape[1] else frame.shape[1]
    y2 = y2 if y2 <= frame.shape[0] else frame.shape[0]

    return frame[y1:y2, x1:x2]





