import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score
from moviepy import VideoFileClip


class Tracker:
    """Generate detections and build tracklets."""

    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        detections = extract_detections(frame)

        if len(detections) == 0:
            return np.zeros((0, 5), dtype=np.int32)
        
        resuls = []
        
        for det in detections:
            det = det[1:]
            tracker_id = self.new_label()
            new = np.insert(det, 0, tracker_id)
            resuls.append(new)
            self.last_detected[tracker_id] = (new)
        
        resuls = np.array(resuls, dtype=np.int32)
        self.detection_history.append(resuls)

        return resuls

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = {}
        # Write code here
        for history in self.detection_history[-self.lookup_tail_size:]:
            for det in history:
                detections[det[0]] = det
                
        
        return detection_cast(list(detections.values()))

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here
        MIN_CONFIDENCE = 0.2
        used_prev_ids = set()
        # Step 1: calc pairwise detection IOU
        for det in detections:
            iou_list = [(iou_score(det[1:],prev[1:5]), prev) for prev in prev_detections]
            if not iou_list:
                det[0] = max([p[0] for p in prev_detections]+[-1]) + 1
                continue
            best_id = max(iou_list, key=lambda x: x[0])
            if iou_list and best_id[0] > MIN_CONFIDENCE and best_id[1][0] not in used_prev_ids:
                det[0] = best_id[1][0]
                used_prev_ids.add(best_id[1][0])
            else:
                det[0] = self.new_label()

        # Step 2: sort IOU list

        # Step 3: fill detections[:, 0] with best match
        # One matching for each id

        # Step 4: assign new tracklet id to unmatched detections

        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections

def main():
    

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.image_transform(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
