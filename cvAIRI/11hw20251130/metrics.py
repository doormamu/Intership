def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    inter_xmin = max(bbox1[0], bbox2[0])
    inter_ymin = max(bbox1[1],bbox2[1])
    inter_xmax = min(bbox1[2], bbox2[2])
    inter_ymax = min(bbox1[3], bbox2[3])

    inter_y = max(0, inter_ymax - inter_ymin)
    inter_x = max(0, inter_xmax - inter_xmin)
    
    cross = inter_y*inter_x
    union = (bbox1[3]-bbox1[1])*(bbox1[2]-bbox1[0]) + (bbox2[3]-bbox2[1])*(bbox2[2]-bbox2[0]) - cross
    return cross/union # пересечение / объединение


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {int(det[0]): det[1:] for det in frame_obj}
        hyp_dict = {int(det[0]): det[1:] for det in frame_hyp}

        curr_matches = {}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for oid, hid in list(matches.items()):
            if oid in obj_dict and hid in hyp_dict:
                iou = iou_score(obj_dict[oid], hyp_dict[hid])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    curr_matches[oid] = hid
                    del obj_dict[oid]
                    del hyp_dict[hid]
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairs = []
        for oid, o_box in obj_dict.items():
            for hid, h_box in hyp_dict.items():
                iou = iou_score(o_box, h_box)
                if iou > threshold:
                    pairs.append((iou, oid, hid))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_obj = set()
        used_hyp = set()

        for iou, oid, hid in pairs:
            if oid not in used_obj and hid not in used_hyp:
                dist_sum += iou
                match_count += 1
                curr_matches[oid] = hid
                used_obj.add(oid)
                used_hyp.add(hid)
        # Step 5: Update matches with current matched IDs
        matches = curr_matches
        #pass

    if match_count == 0:
        return 0.0
    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    text

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_idx, (frame_obj, frame_hyp) in enumerate(zip(obj, hyp)):
        #print(f"\n=== Frame {frame_idx} ===")  
        #print(f"GT objects: {frame_obj}")  
        #print(f"Hypotheses: {frame_hyp}")  
        #print(f"Previous matches: {matches}")  
        
        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj_dict = {int(det[0]): det[1:] for det in frame_obj}
        frame_hyp_dict = {int(det[0]): det[1:] for det in frame_hyp}
        new_matches = {}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        to_delete_obj = []
        to_delete_hyp = []
        for obj_id, hyp_id in matches.items():
            if obj_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                iou = iou_score(frame_obj_dict[obj_id], frame_hyp_dict[hyp_id])
                #print(f"  Checking previous match obj{obj_id}->hyp{hyp_id}: IOU={iou:.3f}")  # ИСПРАВЛЕНО: добавлен лог
                if iou >= threshold:
                    dist_sum += iou
                    match_count += 1
                    new_matches[obj_id] = hyp_id 
                    to_delete_obj.append(obj_id)
                    to_delete_hyp.append(hyp_id)
                    #print(f"    -> MATCHED")  

        for obj_id in to_delete_obj:
            frame_obj_dict.pop(obj_id)
        for hyp_id in to_delete_hyp:
            frame_hyp_dict.pop(hyp_id)
        
        #print(f"  Remaining objects: {list(frame_obj_dict.keys())}") 
        #print(f"  Remaining hypotheses: {list(frame_hyp_dict.keys())}")  
        
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairwise = []
        for obj_id, obj_box in frame_obj_dict.items():
            for hyp_id, hyp_box in frame_hyp_dict.items():
                iou = iou_score(obj_box, hyp_box)
                if iou >= threshold:
                    pairwise.append((iou, obj_id, hyp_id))
                    #print(f"  Pairwise: obj{obj_id}->hyp{hyp_id}: IOU={iou:.3f}")  
        
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pairwise.sort(reverse=True, key=lambda x: x[0])
        for iou, obj_id, hyp_id in pairwise:
            if obj_id in frame_obj_dict and hyp_id in frame_hyp_dict:
                if obj_id in matches and matches[obj_id] != hyp_id:
                    mismatch_error += 1
                    #print(f"    -> ID SWITCH! (was hyp{matches[obj_id]})") 
                
                dist_sum += iou 
                match_count += 1
                new_matches[obj_id] = hyp_id
                frame_obj_dict.pop(obj_id)
                frame_hyp_dict.pop(hyp_id)
                #print(f"    -> NEW MATCH obj{obj_id}->hyp{hyp_id}")  
        
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        current_frame_obj_ids = set(int(det[0]) for det in frame_obj)
        for obj_id in current_frame_obj_ids:
            if obj_id in new_matches:
                matches[obj_id] = new_matches[obj_id]
            elif obj_id in matches:
                matches.pop(obj_id, None)
        
        # Step 6: Update matches with current matched IDs
        missed_count += len(frame_obj_dict)
        false_positive += len(frame_hyp_dict)
        
        #print(f"  Misses: {len(frame_obj_dict)}, FP: {len(frame_hyp_dict)}")  
        #print(f"  New matches: {new_matches}")  
        #print(f"  Cumulative - matched: {match_count}, miss: {missed_count}, fp: {false_positive}, mismatch: {mismatch_error}")  
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        #pass

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count if match_count > 0 else 0
    total_objects = sum(len(frame) for frame in obj)
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / total_objects if total_objects > 0 else 0

    #print(f"\n=== FINAL ===")  
    #print(f"MOTP: {MOTP:.6f}, MOTA: {MOTA:.6f}")  
    #print(f"Total objects: {total_objects}")  

    return MOTP, MOTA