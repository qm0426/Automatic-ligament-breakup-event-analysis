import os
import cv2
import numpy as np

from ultralytics import YOLO

from sklearn.cluster import KMeans
from sam2.build_sam import build_sam2_video_predictor

from utils import deal_with_single_frame, post_process_via_cluster, pro_propress_mask


class ligament_segmentor:
    def __init__(
        self,
        image_root,
        save_root,
        prompt_root,
        checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    ):
        self.image_root = image_root
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)
        self.prompt_root = prompt_root
        os.makedirs(self.prompt_root, exist_ok=True)
        self.predictor = build_sam2_video_predictor(checkpoint, model_cfg)

    def yolo_train(self, data_yaml_path, num_epochs=500, image_size=640, device='0'):
        model = YOLO(yaml="yolov8.yaml")
        model.train(
            data=data_yaml_path, epochs=num_epochs, imgsz=image_size, device=device
        )

    def yolo_main(self, yolo_weight_root, save=True, conf=0.5):
        '''
        using yolo to inference the image, and save the result
        '''

        def yolo_inference(weight_path, img_path, save=True, conf=0.5):
            model = YOLO(weight_path)
            model.predict(
                img_path,
                device="0",
                save=save,
                conf=conf,
                save_txt=True,
            )

        imglist = os.listdir(os.path.join(self.image_root))
        for img in imglist:
            yolo_inference(
                yolo_weight_root, os.path.join(self.image_root, img, "img"), save, conf
            )

    def post_process_yolo_main(self, detect_res_root, H, W):
        '''
        main function for dealing with the yolo detection results and save the processed results in the txt format.

        Due to the default name of yolo detection result is predcitXX, for the later processing, we need to change the name to the original case name from the real_case_name_list.
        '''
        txt_path = os.path.join(detect_res_root, 'labels')
        txt_list = os.listdir(txt_path)

        for i in range(0, len(txt_list)):
            boxes = deal_with_single_frame(os.path.join(txt_path, txt_list[i]), W, H)

            for j in range(len(boxes)):
                with open(
                    os.path.join(
                        self.prompt_root, txt_list[i][:-4] + "_" + str(j) + ".txt"
                    ),
                    "w",
                ) as f:

                    f.write(
                        f"{int(boxes[j][0])} {int(boxes[j][1])} {int(boxes[j][2])} {int(boxes[j][3])}\n"
                    )

    def giving_prompt_sam2(self):
        '''
        generate the prompts for sam2 model, the prompts are the boxes of the detected ligaments.
        '''
        boxes = []
        txtlist = os.listdir(self.prompt_root)
        prompts = []
        frames = []
        all_boxes = []
        for txt in txtlist:

            with open(os.path.join(self.prompt_root, txt), "r") as f:
                lines = f.readlines()
                # all_lines.append(lines)
            for line in lines:
                x1, y1, x2, y2 = line[:-1].split(" ")
                temp_box = [int(x1), int(y1), int(x2), int(y2)]
            boxes.append(temp_box)
        for b in range(len(boxes)):
            if prompts == []:
                prompts.append(boxes[b])
                frames.append(0)

            else:
                if boxes[b][2] - boxes[b][0] > (prompts[-1][2] - prompts[-1][0]) * 1.1:
                    prompts.append(boxes[b])
                    frames.append(b)
            all_boxes.append([b, boxes[b][0], boxes[b][1], boxes[b][2], boxes[b][3]])
        self.prompts = prompts
        self.prompt_frames = frames
        self.obj_ids = [0 for _ in range(len(prompts))]

    def sam2_seg(self):
        '''
        the main function for sam2 segmentation, including sam2 and sam2 with cluster post process.
        '''

        video_segments = {}
        img_root = os.path.join(self.image_root, "img")
        frame_names = os.listdir(img_root)

        save_path_cluster = os.path.join(self.save_root, "sam2_cluster")

        save_path = os.path.join(self.save_root, "sam2")

        os.makedirs(save_path, exist_ok=True)

        os.makedirs(save_path_cluster, exist_ok=True)

        inference_state = self.predictor.init_state(img_root)
        boxes, frame_idxs, obj_ids, _ = self.giving_prompt_sam2()
        for frame_idx, obj_id, box in zip(frame_idxs, obj_ids, boxes):

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx in range(frame_idxs[0], len(frame_names)):
            img = cv2.imread(os.path.join(img_root, frame_names[out_frame_idx]))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask = out_mask.squeeze(0)
                out_mask = out_mask.astype(np.uint8) * 255

                final_mask_cluster = post_process_via_cluster(img, out_mask)

                final_mask = out_mask

                cv2.imwrite(
                    os.path.join(save_path, frame_names[out_frame_idx]), final_mask
                )

                cv2.imwrite(
                    os.path.join(save_path_cluster, frame_names[out_frame_idx]),
                    final_mask_cluster,
                )

    def only_cluster_seg(self):
        def seg(img, box):
            kmeans = KMeans(n_clusters=2)
            img = img[box[1] : box[3], box[0] : box[2]]
            kmeans.fit(img.reshape(-1, 1))
            raw_mask = kmeans.labels_.reshape(img.shape)
            t_raw_mask = 1 - raw_mask
            if np.average(img * raw_mask) > np.average(img * t_raw_mask):
                return t_raw_mask
            else:
                return raw_mask

        def giving_prompt_cluster(txtpath):
            box = []
            with open(txtpath, "r") as f:
                lines = f.readlines()
            for line in lines:
                x1, y1, x2, y2 = line[:-1].split(" ")
                box += [int(x1), int(y1), int(x2), int(y2)]
            return box

        save_path = os.path.join(self.save_root, "cluster")

        os.makedirs(save_path)

        imglist = os.listdir(os.path.join(self.image_root, "img"))
        prompt_list = os.listdir(self.prompt_root)
        for i in imglist:
            name = i[:-4]
            img = cv2.imread(os.path.join(self.image_root, "img", i), 0)
            H, W = img.shape
            if name + "_0.txt" in prompt_list:
                box = giving_prompt_cluster(
                    os.path.join(self.prompt_root, name + "_0.txt")
                )
                x1, y1, x2, y2 = box
                mask = seg(img, box) * 255
                mask = pro_propress_mask(mask)
                mask = cv2.copyMakeBorder(
                    mask,
                    y1,
                    H - y2,
                    x1,
                    W - x2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=0,
                )
                cv2.imwrite(os.path.join(save_path, name + ".jpg"), mask)

    def seg_main(self, detect_res_root, H=480, W=640):
        self.post_process_yolo_main(detect_res_root, H, W)
        self.sam2_seg()
        self.only_cluster_seg()


def main(
    image_root,
    save_root,
    prompt_root,
    data_yaml_path=None,
    if_need_train=True,
    H=480,
    W=640,
    num_epochs=500,
    image_size=640,
    device='cuda',
):
    segmentor = ligament_segmentor(image_root, save_root, prompt_root)
    if not if_need_train:
        segmentor.seg_main()
    else:
        segmentor.yolo_train(data_yaml_path, num_epochs, image_size, device)
        yolo_weight_root = input("please input the yolo weight path:")
        segmentor.yolo_main(yolo_weight_root)
        detect_res_root = input("please input the yolo detect result path:")
        segmentor.seg_main(detect_res_root, H, W)
