import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch.nn.functional as F  # For padding and image processing
import csv
import time

from segment_anything import sam_model_registry, SamPredictor

# ----------------------
# Define class properties
# ----------------------
CLASSES = {
    "kidney": {
         "target_color": np.array([255, 55, 0]),  # RGB for kidney-parenchyma
         "display_color": np.array([30/255, 144/255, 255/255, 0.6])
    },
    "small_intestine": {
         "target_color": np.array([124, 155, 5]),   # RGB for small-intestine
         "display_color": np.array([124/255, 155/255, 5/255, 0.6])
    },
    "instrument-shaft": {
         "target_color": np.array([0, 255, 0]),       # RGB for instrument-shaft
         "display_color": np.array([0/255, 255/255, 0/255, 0.6])
    },
    "instrument-clasper": {
         "target_color": np.array([0, 255, 255]),     # RGB for instrument-clasper
         "display_color": np.array([0/255, 255/255, 255/255, 0.6])
    },
    "instrument-wrist": {
         "target_color": np.array([125, 255, 12]),    # RGB for instrument-wrist
         "display_color": np.array([125/255, 255/255, 12/255, 0.6])
    },
    "clamps": {
         "target_color": np.array([0, 255, 125]),    
         "display_color": np.array([125/255, 255/255, 125/255, 0.6])
    },
}

# ----------------------
# Visualization helpers
# ----------------------
def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

def show_box(box, ax, edge_color=None):
    if edge_color is None:
        edge_color = 'green'
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                 edgecolor=edge_color, facecolor=(0, 0, 0, 0), lw=2))

# ----------------------
# Segmentation Metrics Helpers
# ----------------------
def dice_coefficient(pred, gt, smooth=1e-6):
    """Compute the Dice coefficient between binary masks."""
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

def iou_score(pred, gt, smooth=1e-6):
    """Compute the Intersection over Union (IoU) between binary masks."""
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    intersection = np.sum(pred * gt)
    union = np.sum(np.maximum(pred, gt))
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, gt, smooth=1e-6):
    """Compute precision between binary masks."""
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(pred, gt, smooth=1e-6):
    """Compute recall between binary masks."""
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    return (tp + smooth) / (tp + fn + smooth)

# ----------------------
# Bounding box computation functions
# ----------------------
def compute_bbox_from_mask_simple(mask_path, target_color_rgb):
    """
    Compute a bounding box for the target color region in the mask image.
    Returns [x_min, y_min, x_max, y_max].
    """
    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        print(f"Warning: Could not read mask image {mask_path}")
        return None
    mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    binary_mask = ((mask_rgb[:, :, 0] == target_color_rgb[0]) &
                   (mask_rgb[:, :, 1] == target_color_rgb[1]) &
                   (mask_rgb[:, :, 2] == target_color_rgb[2])).astype(np.uint8)
    ys, xs = np.where(binary_mask)
    if ys.size == 0 or xs.size == 0:
        print(f"No pixels match the target color {target_color_rgb}.")
        return None
    min_y, max_y = np.min(ys), np.max(ys)
    min_x, max_x = np.min(xs), np.max(xs)
    return [min_x, min_y, max_x, max_y]

# ----------------------
# New function for multiple bounding boxes
# ----------------------
def compute_bboxes_from_mask(mask_path, target_color_rgb, min_area=10):
    """
    Compute bounding boxes for every connected component in the mask that matches the target color.
    Only components with area >= min_area are returned.
    """
    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        print(f"Warning: Could not read mask image {mask_path}")
        return []
    mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    binary_mask = ((mask_rgb[:, :, 0] == target_color_rgb[0]) &
                   (mask_rgb[:, :, 1] == target_color_rgb[1]) &
                   (mask_rgb[:, :, 2] == target_color_rgb[2])).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    bboxes = []
    for label in range(1, num_labels):  # label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        bboxes.append([left, top, left + width, top + height])
    return bboxes

def compute_bbox_from_predicted_mask(mask, margin=10, image_shape=None):
    """
    Computes a bounding box that fully encloses the predicted mask.
    Expands the box by a margin.
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    min_x_exp = min_x - margin
    max_x_exp = max_x + margin
    min_y_exp = min_y - margin
    max_y_exp = max_y + margin
    if image_shape is not None:
        H, W = image_shape[:2]
        min_x_exp = max(0, min_x_exp)
        min_y_exp = max(0, min_y_exp)
        max_x_exp = min(W, max_x_exp)
        max_y_exp = min(H, max_y_exp)
    return [min_x_exp, min_y_exp, max_x_exp, max_y_exp]

# ----------------------
# Prompt computation functions
# ----------------------
def compute_prompt_from_mask(mask_path, target_color_rgb):
    """
    Compute the centroid of the largest connected component for the target class.
    """
    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        print(f"Warning: Could not read mask image {mask_path}")
        return None
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    binary_mask = ((mask_img[:, :, 0] == target_color_rgb[0]) &
                   (mask_img[:, :, 1] == target_color_rgb[1]) &
                   (mask_img[:, :, 2] == target_color_rgb[2])).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None
    max_area = 0
    max_label = 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label
    centroid = centroids[max_label]
    return [float(centroid[0]), float(centroid[1])]

def compute_negative_prompts_from_mask(mask_path, target_color_rgb, num_neg_points=1, kernel_size=5):
    """
    Compute negative prompt points from the border of the target region.
    """
    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        print(f"Warning: Could not read mask image {mask_path}")
        return None
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    binary_mask = ((mask_img[:, :, 0] == target_color_rgb[0]) &
                   (mask_img[:, :, 1] == target_color_rgb[1]) &
                   (mask_img[:, :, 2] == target_color_rgb[2])).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    border = dilated - binary_mask
    border_coords = np.argwhere(border > 0)
    if len(border_coords) == 0:
        return None
    if num_neg_points == 1:
        mean_row, mean_col = np.mean(border_coords, axis=0)
        return [[float(mean_col), float(mean_row)]]
    indices = np.random.choice(len(border_coords), size=num_neg_points, replace=False)
    neg_points = []
    for idx in indices:
        row, col = border_coords[idx]
        neg_points.append([float(col), float(row)])
    return neg_points

# ----------------------
# Main pipeline
# ----------------------
def main(args: argparse.Namespace) -> None:
    print("Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Determine segmentation output directory (for SAM results)
    if args.box is not None and args.point is None:
        seg_path = os.path.join(args.data, f"seg_box_{args.iteration}_multimask")
    elif args.box is None and args.point is not None:
        seg_path = os.path.join(args.data, f"seg_point_{args.iteration}_multimask")
    elif args.box is not None and args.point is not None:
        seg_path = os.path.join(args.data, f"seg_box_point_{args.iteration}_multimask")
    else:
        seg_path = os.path.join(args.data, f"seg_default_{args.iteration}_multimask")
    os.makedirs(seg_path, exist_ok=True)

    # Set up folder to save ground truth overlays.
    gt_overlay_path = os.path.join(os.path.dirname(seg_path), "gt_overlay_" + str(args.iteration))
    os.makedirs(gt_overlay_path, exist_ok=True)

    # Initialize global metrics container and exclusion list (for "video" dataset).
    all_metrics = {}  # {dataset: [ { "image_name": ..., "kidney": {..}, ... }, ... ]}
    exclude_images = {"video": ["00001", "00028"]} # these images are test images, we want to skip these

    # Process each dataset directory.
    for dataset in ["test", "video", "novel_views"]:
        image_dir = os.path.join(args.data, dataset, f"ours_{args.iteration}", "renders")
        if not os.path.exists(image_dir):
            print(f"Directory {image_dir} does not exist!")
            continue
        images = sorted([os.path.join(image_dir, f)
                         for f in os.listdir(image_dir)
                         if not os.path.isdir(os.path.join(image_dir, f))])

        # Precomputed embeddings.
        feature_dir = os.path.join(args.data, dataset, f"ours_{args.iteration}", "saved_feature")
        use_precomputed_embedding = (not args.image) and os.path.exists(feature_dir)
        if use_precomputed_embedding:
            features = sorted([os.path.join(feature_dir, f)
                               for f in os.listdir(feature_dir)
                               if not os.path.isdir(os.path.join(feature_dir, f))])
            if len(features) != len(images):
                print("Warning: Number of feature files does not match number of images.")
        else:
            if not args.image:
                print(f"Features directory '{feature_dir}' does not exist! Computing embeddings on the fly.")

        # Check for mask directory.
        mask_dir = os.path.join(args.data, dataset, f"ours_{args.iteration}", "all_masks")
        use_mask_prompt = os.path.exists(mask_dir)
        if use_mask_prompt:
            mask_files = sorted([os.path.join(mask_dir, f)
                                 for f in os.listdir(mask_dir)
                                 if not os.path.isdir(os.path.join(mask_dir, f))])
            if len(mask_files) != len(images):
                print("Warning: Number of mask files does not match number of images.")
        else:
            print(f"No mask directory found at {mask_dir}.")

        # Create output subfolders for this dataset.
        output_dir = os.path.join(seg_path, dataset)
        os.makedirs(output_dir, exist_ok=True)
        gt_dataset_dir = os.path.join(gt_overlay_path, dataset)
        os.makedirs(gt_dataset_dir, exist_ok=True)

        for i, image_path in enumerate(tqdm(images, desc=f"Processing {dataset} images")):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"Processing '{image_path}' ...")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            precomputed_features = None
            if use_precomputed_embedding:
                embedding = torch.load(features[i])
                if embedding.dim() == 3:
                    embedding = embedding.unsqueeze(0)
                _, _, fea_h, fea_w = embedding.shape
                if fea_w > fea_h:
                    embedding = F.pad(embedding, (0, 0, 0, fea_w - fea_h))
                predictor.image_embedding = embedding.to(args.device)
                precomputed_features = embedding.to(args.device)

            predictor.set_image(image, features=precomputed_features)

            # Process segmentation per class using mask prompts.
            class_results = {}
            if use_mask_prompt:
                mask_path = mask_files[i]
                for class_name, class_info in CLASSES.items():
                    target_color = class_info["target_color"]
                    if class_name in ["instrument-shaft", "instrument-clasper", "instrument-wrist", "clamps"]:
                        bboxes = compute_bboxes_from_mask(mask_path, target_color_rgb=target_color, min_area=10)
                        if not bboxes:
                            print(f"No {class_name} target color found in mask; skipping segmentation for this class.")
                            continue
                        inst_results = []
                        for bbox in bboxes:
                            box_prompt = np.array(bbox, dtype=np.float32).reshape(1, 4)
                            masks, scores, logits = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=box_prompt,
                                multimask_output=True,
                            )
                            best_idx = np.argmax(scores) if scores is not None and len(scores) > 0 else 0
                            best_mask = masks[best_idx]
                            computed_box = compute_bbox_from_predicted_mask(best_mask, margin=10, image_shape=image.shape)
                            inst_results.append({"mask": best_mask, "box": computed_box})
                        class_results[class_name] = {"instances": inst_results}
                    else:
                        if args.prompt_mode == 'bbox':
                            computed_prompt = compute_bbox_from_mask_simple(mask_path, target_color_rgb=target_color)
                            if computed_prompt is None:
                                print(f"No {class_name} target color found in mask; skipping segmentation for this class.")
                                continue
                            else:
                                box_prompt = np.array(computed_prompt, dtype=np.float32).reshape(1, 4)
                                print(f"Using {class_name} mask-derived prompt box: {computed_prompt}")
                                input_point, input_label = None, None
                        else:
                            computed_prompt = compute_prompt_from_mask(mask_path, target_color_rgb=target_color)
                            if computed_prompt is None:
                                print(f"No {class_name} target color found in mask; skipping segmentation for this class.")
                                continue
                            else:
                                input_point = [computed_prompt]
                                input_label = [1]
                                print(f"Using {class_name} mask-derived prompt point: {computed_prompt}")
                                neg_prompts = compute_negative_prompts_from_mask(mask_path, target_color_rgb=target_color,
                                                                                 num_neg_points=args.num_neg_points)
                                if neg_prompts is not None:
                                    for pt in neg_prompts:
                                        input_point.append(pt)
                                        input_label.append(0)
                                    print(f"Using {class_name} mask-derived negative prompt(s): {neg_prompts}")
                            box_prompt = None
                        
                        if input_point is not None:
                            input_point = np.array(input_point, dtype=np.float32)
                            input_label = np.array(input_label, dtype=np.int32)
                        
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            box=box_prompt,
                            multimask_output=True,
                        )
                        best_idx = np.argmax(scores) if scores is not None and len(scores) > 0 else 0
                        best_mask = masks[best_idx]
                        computed_box = compute_bbox_from_predicted_mask(best_mask, margin=10, image_shape=image.shape)
                        class_results[class_name] = {"mask": best_mask, "box": computed_box}
            else:
                # Fallback if no mask prompts (not detailed here).
                pass

            # Overlay segmentation results.
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for class_name, res in class_results.items():
                display_color = CLASSES[class_name]["display_color"]
                if class_name in ["instrument-shaft", "instrument-clasper", "instrument-wrist", "clamps"]:
                    for inst in res["instances"]:
                        show_mask(inst["mask"], plt.gca(), color=display_color)
                        if inst["box"] is not None:
                            show_box(inst["box"], plt.gca(), edge_color=display_color)
                else:
                    show_mask(res["mask"], plt.gca(), color=display_color)
                    if res["box"] is not None:
                        show_box(res["box"], plt.gca(), edge_color=display_color)
            plt.axis('off')
            seg_save_path = os.path.join(output_dir, image_name + '.png')
            plt.savefig(seg_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved segmentation result to '{seg_save_path}'")

            # ---------------------------
            # Compute and Store Metrics for this Image
            # ---------------------------
            # Only compute metrics if a GT mask is available.
            image_metrics = {}
            # If the dataset is "video", skip images in the exclusion list.
            if dataset == "video" and image_name in exclude_images.get("video", []):
                print(f"Excluding image {image_name} from metric computation.")
            else:
                gt_mask = cv2.imread(mask_files[i])
                if gt_mask is not None:
                    gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
                    for class_name, res in class_results.items():
                        target_color = CLASSES[class_name]["target_color"]
                        gt_binary = ((gt_mask_rgb[:, :, 0] == target_color[0]) &
                                     (gt_mask_rgb[:, :, 1] == target_color[1]) &
                                     (gt_mask_rgb[:, :, 2] == target_color[2])).astype(np.uint8)
                        
                        if class_name in ["instrument-shaft", "instrument-clasper", "instrument-wrist", "clamps"]:
                            instance_metrics = []
                            for inst in res.get("instances", []):
                                pred_mask = (inst["mask"] > 0.5).astype(np.uint8)
                                d = dice_coefficient(pred_mask, gt_binary)
                                iou = iou_score(pred_mask, gt_binary)
                                prec = precision_score(pred_mask, gt_binary)
                                rec = recall_score(pred_mask, gt_binary)
                                instance_metrics.append({"dice": d, "iou": iou, "precision": prec, "recall": rec})
                            if instance_metrics:
                                avg_dice = np.mean([m["dice"] for m in instance_metrics])
                                avg_iou = np.mean([m["iou"] for m in instance_metrics])
                                avg_prec = np.mean([m["precision"] for m in instance_metrics])
                                avg_rec = np.mean([m["recall"] for m in instance_metrics])
                                image_metrics[class_name] = {"dice": avg_dice, "iou": avg_iou, 
                                                             "precision": avg_prec, "recall": avg_rec}
                        else:
                            pred_mask = (res["mask"] > 0.5).astype(np.uint8)
                            d = dice_coefficient(pred_mask, gt_binary)
                            iou = iou_score(pred_mask, gt_binary)
                            prec = precision_score(pred_mask, gt_binary)
                            rec = recall_score(pred_mask, gt_binary)
                            image_metrics[class_name] = {"dice": d, "iou": iou, 
                                                         "precision": prec, "recall": rec}
                    
                    # Aggregated binary metrics:
                    binary_pred = np.zeros(image.shape[:2], dtype=np.uint8)
                    for class_name, res in class_results.items():
                        if class_name in ["instrument-shaft", "instrument-clasper", "instrument-wrist", "clamps"]:
                            for inst in res.get("instances", []):
                                binary_pred = np.logical_or(binary_pred, (inst["mask"] > 0.5))
                        else:
                            binary_pred = np.logical_or(binary_pred, (res["mask"] > 0.5))
                    binary_gt = np.zeros(image.shape[:2], dtype=np.uint8)
                    for class_name, class_info in CLASSES.items():
                        target_color = class_info["target_color"]
                        gt_class = ((gt_mask_rgb[:, :, 0] == target_color[0]) &
                                    (gt_mask_rgb[:, :, 1] == target_color[1]) &
                                    (gt_mask_rgb[:, :, 2] == target_color[2])).astype(np.uint8)
                        binary_gt = np.logical_or(binary_gt, gt_class)
                    
                    agg_dice = dice_coefficient(binary_pred.astype(np.uint8), binary_gt.astype(np.uint8))
                    agg_iou = iou_score(binary_pred.astype(np.uint8), binary_gt.astype(np.uint8))
                    agg_prec = precision_score(binary_pred.astype(np.uint8), binary_gt.astype(np.uint8))
                    agg_rec = recall_score(binary_pred.astype(np.uint8), binary_gt.astype(np.uint8))
                    image_metrics["aggregated"] = {"dice": agg_dice, "iou": agg_iou, 
                                                   "precision": agg_prec, "recall": agg_rec}
                    print(f"Metrics for image {image_name}: {image_metrics}")
            # Store metrics per image.
            if dataset not in all_metrics:
                all_metrics[dataset] = []
            all_metrics[dataset].append({"image_name": image_name, **image_metrics})

            # ---------------------------
            # Save Ground Truth Overlay
            # ---------------------------
            if use_mask_prompt:
                gt_mask_img = cv2.imread(mask_files[i])
                if gt_mask_img is not None:
                    gt_mask_img = cv2.cvtColor(gt_mask_img, cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(image, 0.5, gt_mask_img, 0.5, 0)
                    gt_save_path = os.path.join(gt_dataset_dir, image_name + '.png')
                    plt.imsave(gt_save_path, overlay)
                    print(f"Saved GT overlay to '{gt_save_path}'")
                else:
                    print(f"Warning: Could not load GT mask from {mask_files[i]}")

    # ---------------------------
    # Save Metrics to CSV and Compute Averages
    # ---------------------------
    with open("segmentation_metrics.csv", "w", newline="") as csvfile:
        fieldnames = ["dataset", "image_name", "class", "dice", "iou", "precision", "recall"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, images_metrics in all_metrics.items():
            for im in images_metrics:
                image_name = im["image_name"]
                for cls, metrics in im.items():
                    if cls == "image_name":
                        continue
                    writer.writerow({
                        "dataset": dataset,
                        "image_name": image_name,
                        "class": cls,
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"]
                    })

    # Compute and print average metrics per dataset.
    print("\nAverage Metrics per Dataset:")
    for dataset, images_metrics in all_metrics.items():
        sum_metrics = {}
        for im in images_metrics:
            for cls, metrics in im.items():
                if cls == "image_name":
                    continue
                if cls not in sum_metrics:
                    sum_metrics[cls] = {"dice": 0, "iou": 0, "precision": 0, "recall": 0, "count": 0}
                sum_metrics[cls]["dice"] += metrics["dice"]
                sum_metrics[cls]["iou"] += metrics["iou"]
                sum_metrics[cls]["precision"] += metrics["precision"]
                sum_metrics[cls]["recall"] += metrics["recall"]
                sum_metrics[cls]["count"] += 1
        print(f"Dataset: {dataset}")
        for cls, vals in sum_metrics.items():
            avg_dice = vals["dice"] / vals["count"]
            avg_iou = vals["iou"] / vals["count"]
            avg_prec = vals["precision"] / vals["count"]
            avg_rec = vals["recall"] / vals["count"]
            print(f"  Class {cls}: Dice = {avg_dice:.4f}, IoU = {avg_iou:.4f}, Precision = {avg_prec:.4f}, Recall = {avg_rec:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment images with SAM using dynamic prompts derived from mask information. "
                    "Compute segmentation metrics and save results to a CSV file. "
                    "Excludes specified images from the 'video' dataset and computes aggregated binary metrics."
    )
    parser.add_argument("--model-type", type=str, required=True,
                        help="Type of model to load, e.g., 'vit_h', 'vit_b'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the SAM checkpoint (e.g., sam_vit_b_01ec64.pth)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path including input images and (optionally) saved features and all_masks folders")
    parser.add_argument("--iteration", type=int, required=True,
                        help="Chosen iteration number")
    parser.add_argument("--image", action="store_true",
                        help="If set, compute image embeddings on the fly. Otherwise, load precomputed .pt embeddings if available.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--prompt_mode", type=str, default="bbox", choices=["point", "bbox"],
                        help="Type of prompt to use: 'point' or 'bbox'")
    parser.add_argument("--num_neg_points", type=int, default=0,
                        help="Number of negative prompt points to sample (only used in point mode)")
    parser.add_argument('--point', type=int, nargs='+',
                        help='Two values x y as a fixed positive prompt (fallback if mask prompt is unavailable)')
    parser.add_argument('--box', type=int, nargs='+',
                        help='Four values x1 y1 x2 y2 as a fixed bounding box prompt (fallback if mask prompt is unavailable)')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)


#sample usage:

#VIT-H
# python segment_masks.py --checkpoint /mnt/newhome/kai/EndoGaussian_CUHK-AIM_emb/encoders/sam_encoder/checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --data /mnt/newhome/kai/FEG/output/endovis/pulling --iteration 6000

#VIT-B
# python segment_masks.py --checkpoint /mnt/newhome/kai/EndoGaussian_CUHK-AIM_emb/encoders/sam_encoder/checkpoints/sam_vit_b_01ec64.pth --model-type vit_b --data /mnt/newhome/kai/FEG/output/endovis/pulling --iteration 6000

# python segment_masks.py --checkpoint /mnt/newhome/kai/EndoGaussian_CUHK-AIM_emb/encoders/sam_encoder/checkpoints/sam_vit_b_01ec64.pth --model-type vit_b --data /mnt/newhome/kai/EndoGaussian_CUHK-AIM_emb/output/endovis/pulling --iteration 3000

# DONT FORGET to add all_masks folder into the video directory since this isnt automatically generated during rendering, you can find an example if this in: /mnt/newhome/kai/EndoGaussian_CUHK-AIM_emb/output/endovis/pulling/video_128/ours_3000/all_masks