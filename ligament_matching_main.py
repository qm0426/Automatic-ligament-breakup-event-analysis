import cv2
import numpy as np
import os

from utils import cal_light_flow, match, new_match_by_x, choose_area


class Ligament_Matcher:
    def __init__(
        self, mask_root, image_root, save_root, area_thre=0.2, distance_thre=30
    ):
        self.mask_root = mask_root
        self.image_root = image_root
        self.save_root = save_root
        self.area_thre = area_thre
        self.distance_thre = distance_thre
        os.makedirs(os.path.join(self.save_root, "trajectory"), exist_ok=True)
        os.makedirs(
            os.path.join(self.save_root, "parent_child_relationship"), exist_ok=True
        )

    def determine_ligament_matching(self):
        '''
        main matching function to get the trajectory of ligament
        '''
        masklist = os.listdir(self.mask_root)
        imglist = os.listdir(self.image_root)
        connect_areas_prop = []

        for h in range(len(masklist)):
            temp_area_prop = []
            mask = cv2.imread(os.path.join(self.mask_root, masklist[h]), 0)
            mask[mask > 127] = 255
            mask[mask <= 127] = 0

            mask = mask.astype(np.uint8)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            for i in range(1, retval):
                if stats[i, cv2.CC_STAT_AREA] > 10:

                    temp_area_prop.append(
                        [
                            h,
                            i,
                            stats[i, cv2.CC_STAT_AREA],
                            int(centroids[i, 0]),
                            int(centroids[i, 1]),
                        ]
                    )
            connect_areas_prop.append(temp_area_prop)
        pairs = {}
        ini_id = 0
        for i in connect_areas_prop[0]:
            pairs[str(ini_id)] = [i]
            ini_id += 1

        for i in range(len(imglist) - 1):
            img1 = cv2.imread(os.path.join(self.image_root, imglist[i]), 0)
            img2 = cv2.imread(os.path.join(self.image_root, imglist[i + 1]), 0)
            flow = cal_light_flow(img1, img2)
            candidate_area = connect_areas_prop[i + 1]
            pairs = match(
                flow, pairs, candidate_area, self.distance_thre, self.area_thre
            )
        self.ligament_pairs = pairs
        with open(os.path.join(self.save_root, 'trajectory', "pair.txt"), "w") as f:
            for keys, values in pairs.items():

                # if len(values) > 1:
                f.write("{} {}\n".format(keys, len(values)))
                for i in values:

                    id, labelid, area, x, y = i
                    f.write("{} {} {} {} {}\n".format(id, labelid, area, x, y))

    def determine_parchild(self):
        """
        get the relationship between parent ligament and child ligaments
        """
        masklist = os.listdir(self.mask_root)
        txtpath = os.path.join(self.save_root, 'trajectory', "pair.txt")
        bp_frame, infor = choose_area(txtpath)
        parent_droplets = {}
        child_droplets = {}
        for h in range(len(masklist)):
            if h in bp_frame and h != 0:
                data1 = infor[h - 1]
                data2 = infor[h]
                if len(data1) <= len(data2):
                    temp_par, temp_child = new_match_by_x(data1, data2)
                    parent_droplets[h - 1] = temp_par
                    child_droplets[h] = temp_child
        with open(
            os.path.join(
                self.save_root, 'parent_child_relationship', "parent_child.txt"
            ),
            "w",
        ) as f:

            for h in bp_frame:
                try:
                    par_drop = parent_droplets[h - 1]
                    child_drop = child_droplets[h]
                    for i in range(len(par_drop)):
                        f.write(
                            "p {} {} {} {} {}\n".format(
                                h - 1,
                                par_drop[i][0],
                                par_drop[i][1],
                                par_drop[i][2],
                                par_drop[i][3],
                            )
                        )
                        for j in child_drop[i]:
                            f.write(
                                "C {} {} {} {} {}\n".format(h, j[0], j[1], j[2], j[3])
                            )
                except:
                    pass


def main(mask_root, image_root, save_root, area_thre=0.2, distance_thre=30):
    ligament_matching = Ligament_Matcher(
        mask_root, image_root, save_root, area_thre, distance_thre
    )
    ligament_matching.determine_ligament_matching()
    ligament_matching.determine_parchild()
