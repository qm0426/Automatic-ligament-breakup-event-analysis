# Automated Ligament Analysis: Segmentation, Tracking, and Breakup Characterization

## Overview

This project provides a computational pipeline for the analysis of ligament dynamics captured in image sequences (videos). It is the implementation of the paper "Intelligent and quantitative ligament breakup event analysis in 65 kHz off-axis holographic video of swirl spray"

It automates several key tasks:
1.  **Segmentation:** Identifying and isolating ligaments from the background in each frame.
2.  **Tracking:** Following individual ligaments across consecutive frames to understand their trajectories.
3.  **Breakup Analysis:** Detecting ligament breakup events and identifying the relationship between parent ligaments and their resulting child droplets.
4.  **Evaluation:** Quantifying the performance of the segmentation and analyzing characteristics of the breakup events.

## Features

*   **Ligament Detection:** Utilizes YOLOv8 (via `ultralytics`) for initial detection of potential ligament regions. Includes functionality for training a custom YOLOv8 model.
*   **Video Segmentation:** Employs SAM2 (Segment Anything Model 2) for robust segmentation of detected ligaments throughout the video sequence, using initial detections as prompts.
*   **Segmentation Refinement:** Offers post-processing options for segmentation masks using K-Means clustering to improve boundary accuracy. An alternative segmentation method using only K-Means clustering on detected boxes is also provided.
*   **Temporal Tracking:** Tracks ligaments frame-by-frame using optical flow (DeepFlow) estimation and a matching algorithm based on spatial proximity, predicted movement, and area similarity.
*   **Parent-Child Relationship:** Identifies breakup events by analyzing changes in the number of detected ligaments between frames and establishes parent-child relationships using a matching algorithm optimized by relative X-coordinate positions.
*   **Quantitative Analysis:**
    *   Calculates standard segmentation evaluation metrics (Precision, Recall, F1-score) compared to ground truth masks.
    *   Determines breakup characteristics such as breakup time, spatial location, and parent/child ligament lengths.

## Workflow

The typical processing pipeline involves the following steps:

1.  **(Optional) Train YOLOv8:** Train a YOLOv8 model on a custom dataset of ligament images if a suitable pre-trained model is not available (`ligament_segmentation_main.py`).
2.  **Detect Ligaments:** Apply the trained (or pre-trained) YOLOv8 model to each frame of the input image sequence to obtain bounding boxes around potential ligaments (`ligament_segmentation_main.py`).
3.  **Process Detections:** Post-process the raw YOLO detection results to refine bounding boxes (e.g., handle overlaps, merge boxes) and prepare them as prompts for the segmentation model (`utils.py`, `ligament_segmentation_main.py`).
4.  **Segment Ligaments:** Feed the processed prompts and the image sequence into the SAM2 video predictor (or the K-Means clustering method) to generate binary segmentation masks for each ligament in each frame (`ligament_segmentation_main.py`). Optional clustering-based post-processing can be applied to SAM2 masks (`utils.py`).
5.  **Track Ligaments:** Analyze the sequence of generated segmentation masks. Calculate optical flow between consecutive frames and use it along with mask properties (centroid, area) to match and track individual ligaments over time, recording their trajectories (`ligament_matching_main.py`, `utils.py`).
6.  **Analyze Breakups:** Examine the tracked ligament data to identify frames where breakup occurs (i.e., the number of ligaments increases). Determine the parent-child relationships between ligaments immediately before and after the breakup (`ligament_matching_main.py`, `utils.py`).
7.  **Evaluate Results:**
    *   Compare the generated segmentation masks against ground truth masks (if available) to calculate precision, recall, and F1-score (`evaluate.py`).
    *   Use the parent-child relationship data and corresponding masks to calculate breakup position, time, and ligament lengths (`evaluate.py`).

## Core Components

*   `ligament_segmentation_main.py`: Main script for handling YOLO training/inference and SAM2/K-Means segmentation.
*   `ligament_matching_main.py`: Main script for tracking segmented ligaments and determining parent-child relationships during breakups.
*   `evaluate.py`: Script for calculating segmentation performance metrics and analyzing breakup event characteristics.
*   `utils.py`: Module containing various helper functions for image processing (mask handling, connected components), detection post-processing (IoU, box merging), segmentation post-processing (clustering), optical flow, matching algorithms, and file I/O.

## Key Dependencies

*   Python 3.10.0
*   OpenCV (`opencv-python`, `opencv-contrib-python` for `optflow`)
*   NumPy
*   Scikit-learn (`sklearn`)
*   SciPy
*   Ultralytics (`ultralytics`)
*   SAM2 (Requires specific checkpoint, e.g., `sam2.1_hiera_large.pt`, and model config, e.g., `sam2.1_hiera_l.yaml`)

## Usage (Conceptual)

1.  **Setup:** Ensure all dependencies are installed. Place input image sequences, YOLO weights, SAM2 checkpoints/configs, and (optionally) ground truth masks in appropriate directories.
2.  **Segmentation:** Run `ligament_segmentation_main.py`, providing paths to the image directory, output save directory, prompt save directory, and YOLO weights. Specify whether YOLO training is needed.
3.  **Tracking & Breakup Analysis:** Run `ligament_matching_main.py`, providing paths to the segmentation masks, original images, and the output save directory.
4.  **Evaluation:** Run `evaluate.py`, providing paths to the segmentation results, ground truth masks, and the parent-child relationship files generated in the previous step.

## Input / Output

*   **Inputs:**
    *   Directory containing image sequences (e.g., `image_root/case_name/img/frame_001.png`).
    *   YOLOv8 model weights (`.pt` file).
    *   SAM2 checkpoint (`.pt` file) and configuration (`.yaml` file).
    *   (Optional) Directory containing ground truth segmentation masks.
    *   (Optional) Data configuration file (`.yaml`) for YOLO training.
*   **Outputs:**
    *   YOLO detection results (bounding boxes in `.txt` files).
    *   Processed prompt files (`.txt`).
    *   Generated segmentation masks (image files, e.g., `.png` or `.jpg`) for different methods (SAM2, SAM2+Cluster, Cluster-only).
    *   Ligament trajectory data (`pair.txt`).
    *   Parent-child relationship data (`parent_child.txt`).
    *   Evaluation metrics (printed to console or saved to file).
    *   Breakup analysis results (saved to file).
