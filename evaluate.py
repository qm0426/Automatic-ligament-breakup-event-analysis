import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import mask_treatment
from scipy.spatial import distance


def metrics_calculation(seg_res_root, gt_root, method_list):
    caselist = os.listdir(seg_res_root)
    metrics = {}
    for method in method_list:
        ps = []
        rs = []
        f1s = []
        for case in caselist:
            seg_res_path = os.path.join(seg_res_root, case, method)
            gt_path = os.path.join(gt_root, case, "mask")
            t_ps = []
            t_rs = []
            t_f1s = []
            for file in os.listdir(seg_res_path):
                seg_res = mask_treatment(
                    cv2.imread(os.path.join(seg_res_path, file), 0)
                )
                gt = mask_treatment(cv2.imread(os.path.join(gt_path, file), 0))
                p = precision_score(gt, seg_res, average="micro")
                r = recall_score(gt, seg_res, average="micro")
                f1 = f1_score(gt, seg_res, average="micro")
                t_ps.append(p)
                t_rs.append(r)
                t_f1s.append(f1)
            print(case + " " + method + " done")
            ps.append(sum(t_ps) / len(t_ps))
            rs.append(sum(t_rs) / len(t_rs))
            f1s.append(sum(t_f1s) / len(t_f1s))

        ps = np.array(ps)
        rs = np.array(rs)
        f1s = np.array(f1s)
        metrics[method] = [np.average(ps), np.average(rs), np.average(f1s)]
        print(method + " done")
    print(metrics)


def determine_breakup_posi_time(match_root, mask_root, save_root):
    '''
    calculation breakup length, ligament length and breakup time
    '''

    def read_txt(path):
        plist = []
        parents = []
        children = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]

            line = line.strip().split(" ")
            if line[0] == "p":
                plist.append(i)
        for i in range(len(plist) - 1):
            if plist[i + 1] - plist[i] > 2:

                par_line = lines[plist[i]].strip().split(" ")

                temp_child = []
                parents.append(
                    [
                        int(par_line[1]),
                        int(par_line[2]),
                        int(par_line[3]),
                        int(par_line[4]),
                        int(par_line[5]),
                    ]
                )
                for j in range(plist[i] + 1, plist[i + 1]):
                    child_line = lines[j].strip().split(" ")
                    temp_child.append(
                        [
                            int(child_line[1]),
                            int(child_line[2]),
                            int(child_line[3]),
                            int(child_line[4]),
                            int(child_line[5]),
                        ]
                    )
                children.append(temp_child)

        return parents, children

    match_list = os.listdir(match_root)
    all_breakup_information = {}
    for i in match_list:
        match_temp = []
        try:
            parent, children = read_txt(os.path.join(match_root, i))
            casename = i[:-9]
            maskpath = os.path.join(mask_root, casename)
            masklist = os.listdir(maskpath)

            for h, j in zip(parent, children):
                p_id = h[0]
                p_ca_id = h[1]
                p_mask = mask_treatment(
                    cv2.imread(os.path.join(maskpath, masklist[p_id]), 0)
                )
                revtalp, labelsp, statsp, centroidsp = cv2.connectedComponentsWithStats(
                    p_mask, connectivity=8
                )
                labelsp[labelsp != p_ca_id] = 0
                coorsp = np.argwhere(labelsp == p_ca_id)
                yp_min, xp_min = coorsp.min(axis=0)
                yp_max, xp_max = coorsp.max(axis=0)
                l_p = xp_max - xp_min
                for k in range(len(j) - 1):
                    temp_infor = []
                    temp_infor.append(j[0][0])
                    c1 = j[k]
                    c2 = j[k + 1]

                    mask1 = (
                        mask_treatment(
                            cv2.imread(os.path.join(maskpath, masklist[c1[0]]), 0)
                        )
                        * 255
                    )
                    revtal1, labels1, stats1, centroids1 = (
                        cv2.connectedComponentsWithStats(mask1, connectivity=8)
                    )
                    mask2 = (
                        mask_treatment(
                            cv2.imread(os.path.join(maskpath, masklist[c2[0]]), 0)
                        )
                        * 255
                    )
                    revtal2, labels2, stats2, centroids2 = (
                        cv2.connectedComponentsWithStats(mask2, connectivity=8)
                    )
                    labels1[labels1 != c1[1]] = 0
                    labels2[labels2 != c2[1]] = 0
                    coors1 = np.argwhere(labels1 == c1[1])
                    coors2 = np.argwhere(labels2 == c2[1])
                    dist_matrix = distance.cdist(coors1, coors2, "euclidean")
                    dist_matrix1 = distance.cdist(coors1, coors1, "euclidean")
                    dist_matrix2 = distance.cdist(coors2, coors2, "euclidean")
                    l1 = np.max(dist_matrix1)
                    l2 = np.max(dist_matrix2)

                    idx1, idx2 = np.unravel_index(
                        dist_matrix.argmin(), dist_matrix.shape
                    )
                    point1 = coors1[idx1]
                    point2 = coors2[idx2]
                    spatial_posi = [
                        0.5 * (point1[0] + point2[0]),
                        0.5 * (point1[1] + point2[1]),
                    ]
                    temp_infor.append(spatial_posi[0])
                    temp_infor.append(l1)
                    temp_infor.append(l2)
                match_temp.append(temp_infor)
            all_breakup_information[casename] = match_temp
        except:
            pass
    with open(save_root, "w") as f:
        for key, value in all_breakup_information.items():
            f.write(key + "\n")
            if len(value) > 0:
                for i in value:
                    f.write("%d %f %f %f\n" % (i[0], i[1], i[2], i[3]))
