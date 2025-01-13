import collections
import sys
from typing import Optional

import numpy as np
#from loguru import logger

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, xmax, ymax] format is accepted
Box = np.ndarray
# Vector is of shape (1, N)
Vector = np.ndarray
# Track is meant as an output from the object tracker
Track = collections.namedtuple('Track', 'label')

class Detection:
    # Detection is to be an input to the tracker
    def __init__(
            self,
            box: Box,
            name: Optional[str] = "Unknown",
            score: Optional[float] = None,
            feature: Optional[Vector] = None):
        self.box = box
        self.name = name
        self.score = score
        self.feature = feature
        
        self.center = None
        self.compute_center()

    def compute_center(self):
        box_left = self.box[0]
        box_top = self.box[1]
        box_right = self.box[2]
        box_bottom = self.box[3]

        # width = abs(box_right - box_left)
        # height = abs(box_top - box_bottom)

        x_center = (box_left + box_right) / 2
        y_center = (box_top + box_bottom) / 2
        self.center = (x_center, y_center)
        return self.center

    def __repr__(self):
        fmt = "(detection: box=%s, score=%s, feature=%s)"
        return fmt % (str(self.box),
                      str(self.score) or 'none',
                      str(self.feature) or 'none')


# """ utils """
# def set_log_level(level: str) -> None:
#     logger.remove()
#     logger.add(sys.stdout, level=level)
