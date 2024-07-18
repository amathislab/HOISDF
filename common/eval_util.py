# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HFL-Net
# ------------------------------------------------------------------------------

import numpy as np
import open3d as o3d


class EvalUtil:
    """Util class for evaluation networks."""

    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible."""
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype("bool")

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """Returns pck for one keypoint for the given threshold."""
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck

    def _get_epe(self, kp_id):
        """Returns end point error for one keypoint."""
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """Outputs the average mean and median error as well as the pck score."""
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


def verts2pcd(verts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    if color is not None:
        if color == "r":
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == "g":
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == "b":
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    # d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr) # closest dist for each gt point
    # d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt) # closest dist for each pred point
    d1 = gt.compute_point_cloud_distance(pr)  # closest dist for each gt point
    d2 = pr.compute_point_cloud_distance(gt)  # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(
            len(d2)
        )  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(
            len(d1)
        )  # how many of gt points are matched?

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall
