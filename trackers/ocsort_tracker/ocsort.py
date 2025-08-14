"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .association import *
from .matching import get_dists
from .clustering import Clustering
from .track import KalmanBoxTracker

from types import SimpleNamespace

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}

class OCSort(object):
    def __init__(
        self, det_thresh,
        
        
        cluster_eps=0.3,
        cluster_min_samples=5,
        frame_rate=25,
        ioc_thresh=0.7,
        is_ga=True,
        is_reid=False,
        fuse_score=True,
        
        max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0
        
        self.is_reid = is_reid
        
        # group association module
        self.is_ga = is_ga
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.frame_rate = frame_rate
        self.ioc_thresh = ioc_thresh
        
        self._temp_ioc = []
        self.prev_clustered_stracks = None
        self._num_clusters = 0
        
        self.is_ga = is_ga
        if self.is_ga:
            # clustering
            self.clustering = Clustering(
                eps=self.cluster_eps, min_samples=self.cluster_min_samples
            )
            
        self.fuse_score = fuse_score
        
        # ----------------------------
        # print info
        print(f"\nOC-SORT initialized with the following parameters:")
        print(f"  - det_thresh: {self.det_thresh}")
        print(f"  - max_age: {self.max_age}")
        print(f"  - min_hits: {self.min_hits}")
        print(f"  - iou_threshold: {self.iou_threshold}")
        print(f"  - delta_t: {self.delta_t}")
        print(f"  - asso_func: {asso_func}")
        print(f"  - inertia: {self.inertia}")
        print(f"  - is_ga: {self.is_ga}")
        print(f"  - is_reid: {self.is_reid}")
        print(f"  - fuse_score: {self.fuse_score}")
        print(f"  - cluster_eps: {self.cluster_eps}")
        print(f"  - cluster_min_samples: {self.cluster_min_samples}")
        print(f"  - frame_rate: {self.frame_rate}")
        print(f"  - ioc_thresh: {self.ioc_thresh}")
        print(f"  - use_byte: {self.use_byte}")
        print(f"  - is_ga: {self.is_ga}")
        print(f"  - _num_clusters: {self._num_clusters}")
        print(f"  - prev_clustered_stracks: {self.prev_clustered_stracks}")
        print(f"  - clustering: {self.clustering}")
        print(f"  - _temp_ioc: {self._temp_ioc}")
        print("-" * 50)

    def update(self, output_results, img_info, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        bboxes.astype(np.float64)
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        
        # object created for GA module
        ga_detections = [
            SimpleNamespace(
                ltrb=dets[i, :4].copy(),
                score=float(dets[i, 4])
            )
            for i in range(dets.shape[0])
        ]
        # map GA object to global row in dets
        self._det_index_map = {
            id(ga_det): i
            for i, ga_det in enumerate(ga_detections)
        }

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
            

        """
        NEW: First round of association befor OC-SORT using Group Association Module
        """
        # cluster association
        matched_ga = np.empty((0, 2), dtype=int)  # defined for later use

        if self.prev_clustered_stracks and self.is_ga:
            self._num_clusters += len(list(self.prev_clustered_stracks.keys()))
            cluster_matched_detections, _ = self._intersection_over_cluster(
                ga_detections, self.prev_clustered_stracks, overlap_thresh=self.ioc_thresh
            )
            
            if cluster_matched_detections:
                matched_ga, _, _ = self.cluster_matching(
                    prev_clustered_stracks=self.prev_clustered_stracks,
                    cluster_matched_detections=cluster_matched_detections,
                )
                for m in matched_ga:
                    self.trackers[m[1]].update(dets[m[0], :])
    
        # empty map
        self._det_index_map = None
        
        # handle unmatches
        all_det_idxs = np.arange(len(dets), dtype=int)
        used_det_idxs = matched_ga[:, 0] if matched_ga.size else np.array([], dtype=int)
        dets_left_idxs = np.setdiff1d(all_det_idxs, used_det_idxs, assume_unique=False)
        dets_left = dets[dets_left_idxs]
        
        all_trk_idxs = np.arange(len(self.trackers), dtype=int)
        used_trk_idxs = matched_ga[:, 1] if matched_ga.size else np.array([], dtype=int)
        trks_left_idxs = np.setdiff1d(all_trk_idxs, used_trk_idxs, assume_unique=False)
        trks_left = trks[trks_left_idxs]
        
        vel_left  = velocities[trks_left_idxs]
        kobs_left = k_observations[trks_left_idxs]


        
        # ---------------------------
        # Original OC-SORT association
        # ---------------------------
        """
            First round of association
        """
        # matched, unmatched_dets, unmatched_trks = associate(
        #     dets, trks, self.iou_threshold, vel, kobs, self.inertia)
        # for m in matched:
        #     self.trackers[m[1]].update(dets[m[0], :])
        
        # local
        matched2_local, unmatched_dets2_local, unmatched_trks2_local = associate(
            dets_left, trks_left, self.iou_threshold, vel_left, kobs_left, self.inertia
        )
        # Map local -> global
        if matched2_local.size:
            matched2 = np.column_stack([
                dets_left_idxs[matched2_local[:, 0]],
                trks_left_idxs[matched2_local[:, 1]],
            ]).astype(int)
        else:
            matched2 = np.empty((0, 2), dtype=int)
            
        unmatched_dets = dets_left_idxs[unmatched_dets2_local] if unmatched_dets2_local.size else np.empty((0,), dtype=int)
        unmatched_trks = trks_left_idxs[unmatched_trks2_local] if unmatched_trks2_local.size else np.empty((0,), dtype=int)

        for m in matched2:
            self.trackers[m[1]].update(dets[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
    
        """
        NEW: Group Association Module
        """
        if self.is_ga:
            # update clusters to prev_clusters
            
            # get current activated tracks
            output_trks = [
                trk
                for trk in self.trackers
                # if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
                if trk.time_since_update == 0
            ]
            clustered_trks = []
            if len(output_trks) > self.cluster_min_samples:
                clustered_trks, outliers = self.clustering.get_cluster(output_trks)
            self.prev_clustered_stracks = clustered_trks
    
        # return
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


    # ------------------------------------
    # Group Association Module
    def _intersection_over_cluster(
        self, detections:list[KalmanBoxTracker],
        prev_clustered_stracks:dict[int, list[KalmanBoxTracker]],
        overlap_thresh=0.7,
    ):
        """
        Association between detection with previous clusters

        Args:
            detections (list): detections
            prev_clustered_stracks (dict): previous frame clusters

        Returns:
            Tuple[dict, list]: matched and unmatched detections
        """
        detections_ltrb = [det.ltrb for det in detections]
        clusters_bbox = {}
        for idx, cluster in prev_clustered_stracks.items():
            cluster = [prev_trks.ltrb for prev_trks in cluster]
            # calculate overall bouding box in ltrb format
            clusters_bbox[idx] = [
                min([det[0] for det in cluster]),
                min([det[1] for det in cluster]),
                max([det[2] for det in cluster]),
                max([det[3] for det in cluster])
            ]
 
        unmatched_detections = []
        matched_detections = {k: [] for k in prev_clustered_stracks.keys()}
        for i, detection_ltrb in enumerate(detections_ltrb):
            l_detection = detection_ltrb[0]
            t_detection = detection_ltrb[1]
            r_detection = detection_ltrb[2]
            b_detection = detection_ltrb[3]
            area_det = (r_detection - l_detection) * (b_detection - t_detection)

            ioc = 0
            max_overlap_cluster_idx = None
            for j, cluster_bbox in clusters_bbox.items():
                # calculate IoC
                l_overlap = max(cluster_bbox[0], l_detection)
                t_overlap = max(cluster_bbox[1], t_detection)
                r_overlap = min(cluster_bbox[2], r_detection)
                b_overlap = min(cluster_bbox[3], b_detection)
                
                # validate overlap bb
                if l_overlap < r_overlap and t_overlap < b_overlap:
                    # overlap
                    # calculate area of overlap relative to the detection
                    area_overlap = (r_overlap - l_overlap) * (b_overlap - t_overlap)
                    # calculate percentage of detection overlap in cluster
                    ioc_local = area_overlap / area_det
                    
                    # # print("IOC local: ", ioc_local)
                    if ioc_local < overlap_thresh:
                        continue
                    # find the max ioc associates cluster
                    if ioc_local > ioc:
                        self._temp_ioc.append(ioc_local)
                        ioc = ioc_local
                        max_overlap_cluster_idx = j

            # update detection
            if max_overlap_cluster_idx is not None:
                # matched
                matched_detections[max_overlap_cluster_idx].append(detections[i])
            else:
                # unmatched
                unmatched_detections.append(detections[i])

        # drop empty clusters
        matched_detections = {k: v for k, v in matched_detections.items() if v}

        return matched_detections, unmatched_detections

    # def cluster_matching(
    #     self, prev_clustered_stracks:dict[int, list[STrack]],
    #     cluster_matched_detections: dict[int, list[STrack]],
    # ) -> tuple[list[STrack], list[STrack], list[STrack], list[STrack]]:
        
    #     ga_activated_stracks = []
    #     activated_stracks, refind_stracks = [], []

    #     u_tracks, u_detections = [], []
    #     res_tracks, res_detections = [], []
    #     # -------------------------
    #     # for each cluster
    #     for i, cluster_stracks in prev_clustered_stracks.items():
    #         if i not in cluster_matched_detections.keys():
    #             res_tracks.extend(cluster_stracks)
    #             continue
    #         track_ = cluster_stracks
    #         det_ = cluster_matched_detections[i]

    #         # calculate distance
    #         dists = get_dists(
    #             cluster_stracks, cluster_matched_detections[i],
    #             _fuse_score=self.fuse_score,
    #             is_reid=self.is_reid,
    #             feature_thresh=0.8, proximity_thresh=0.5
    #         )

    #         matches, u_track_, u_det_ = linear_assignment(
    #             cost_matrix=dists,
    #             thresh=self.match_thresh
    #         )
    #         # print(f"dists: {dists}")
    #         # print(f"matches: {matches}, ")
            
    #         for itracked, idet in matches:
    #             track = track_[itracked]
    #             det = det_[idet]
                
    #             # if track.state == TrackState.Tracked:
    #             #     # update track with matched detection
    #             #     track.update(det, self.frame_id, update_feature=self.is_reid)
    #             #     activated_stracks.append(track)
    #             # else:
    #             #     # not tracked
    #             #     track.re_activate(det, self.frame_id, new_id=False)
    #             #     refind_stracks.append(track)
                
    #             # adapt to OC-SORT's update method
    #             track.update(det)
    #             ga_activated_stracks.append(track)

    #         u_tracks_ = [track_[t] for t in u_track_]
    #         u_detections_ = [det_[t] for t in u_det_]
        
    #         u_tracks = u_tracks + res_tracks + u_tracks_
    #         u_detections = u_detections + res_detections + u_detections_
        
    #     return ga_activated_stracks, u_tracks, u_detections

    def cluster_matching(
        self,
        prev_clustered_stracks: dict[int, list],          # lists of KalmanBoxTracker (tracks)
        cluster_matched_detections: dict[int, list],      # lists of detection objects; each must have .det_index
    ):
        """
        Returns (matched, unmatched_dets, unmatched_trks) in the same format as OC-SORT's `associate`.
        matched:        (K, 2) int array, rows [det_idx, trk_idx]  (GLOBAL indices)
        unmatched_dets: (D,)   int array of GLOBAL detection row indices
        unmatched_trks: (T,)   int array of GLOBAL tracker indices (indexes into self.trackers)
        """

        # ---- Build global index maps (once per call) ----
        track_index_map = {id(trk): i for i, trk in enumerate(self.trackers)}
        det_index_map = self._det_index_map
        matched_pairs = []
        unmatched_trks_set, unmatched_dets_set = set(), set()

        # Initialize "everything is unmatched" for items present in these clusters
        for trk_list in prev_clustered_stracks.values():
            for trk in trk_list:
                unmatched_trks_set.add(track_index_map[id(trk)])
        for det_list in cluster_matched_detections.values():
            for det in det_list:
                unmatched_dets_set.add(det_index_map[id(det)])

        # ---- Match within clusters that exist on both sides ----
        for cid, track_list in prev_clustered_stracks.items():
            if cid not in cluster_matched_detections:
                # no detections in this cluster -> all tracks remain unmatched (already in set)
                continue

            det_list = cluster_matched_detections[cid]

            # Distance/cost matrix (use your existing helper)
            dists = get_dists(
                track_list, det_list,
                _fuse_score=self.fuse_score,
                is_reid=self.is_reid,
                feature_thresh=0.8,
                proximity_thresh=0.5
            )

            matches, u_track_local, u_det_local = linear_assignment(
                cost_matrix=dists,
                thresh=self.match_thresh
            )

            # Convert local -> global indices and record matches
            for itracked, idet in matches:
                trk_obj = track_list[itracked]
                det_obj = det_list[idet]
                g_t = track_index_map[id(trk_obj)]      # index into self.trackers
                g_d = det_index_map[id(det_obj)]        # row in dets[N, 5]
                matched_pairs.append([g_d, g_t])

                # remove from unmatched sets
                unmatched_trks_set.discard(g_t)
                unmatched_dets_set.discard(g_d)

            # Unmatched locals remain in the sets already initialized
            # (No extra work needed; they were added up-front.)

        # Also handle clusters that exist only on the detection side (no prev tracks):
        for cid, det_list in cluster_matched_detections.items():
            if cid in prev_clustered_stracks:
                continue
            for det in det_list:
                unmatched_dets_set.add(det_index_map[id(det)])

        # ---- Final arrays in OC-SORT `associate` format ----
        matched = np.asarray(matched_pairs, dtype=int) if matched_pairs else np.empty((0, 2), dtype=int)
        unmatched_dets = np.array(sorted(unmatched_dets_set), dtype=int) if unmatched_dets_set else np.empty((0,), dtype=int)
        unmatched_trks = np.array(sorted(unmatched_trks_set), dtype=int) if unmatched_trks_set else np.empty((0,), dtype=int)

        return matched, unmatched_dets, unmatched_trks
