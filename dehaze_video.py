#!/usr/bin/env python
# dehaze_video.py
import argparse
import os
import cv2
import mmcv
from mmcv import Config
from mmedit.models import build_model
from mmcv.runner import load_checkpoint
from mmedit.apis import restoration_video_inference
from mmedit.core import tensor2img
from tqdm import tqdm
import numpy as np
import time
import logging
import copy  # For deepcopy

# ロガー設定
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Dehaze a video using MAP-Net")
    parser.add_argument("-i", "--video_path", required=True, help="Input video file path")
    parser.add_argument(
        "--mode",
        type=str,
        default="preview",
        choices=["save", "preview"],
        help="Operation mode: 'save' to save the output video, 'preview' to show side-by-side frames (default: preview)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output.mp4",
        help="Output video file path (used only if mode is 'save', e.g., output.mp4)",
    )
    parser.add_argument(
        "--config",
        default="configs/dehazers/mapnet/mapnet_hazeworld.py",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/mapnet_hazeworld_40k.pth", help="Path to the model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Sliding window size for video processing (match num_input_frames in config)",
    )
    parser.add_argument("--device", default="cuda:0", help='Device to use (e.g., "cpu", "cuda:0")')
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for processing (mainly for recurrent framework)",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="FPS for the output video (used only if mode is 'save'). If None, use the input video FPS.",
    )
    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=1280,
        help="Maximum width for the preview window (used only if mode is 'preview'). Aspect ratio is maintained.",
    )

    args = parser.parse_args()
    return args


def init_model_mapnet(config_path, checkpoint_path, device="cuda:0"):
    """Initialize the MAP-Net model."""
    logger.info(f"Initializing model from {config_path} and {checkpoint_path} on {device}")
    cfg = Config.fromfile(config_path)
    model_test_cfg = cfg.get("test_cfg")  # Get test_cfg from config

    # Adjust test_cfg for inference-only mode before building the model
    if model_test_cfg is not None:
        logger.info(f"Original test_cfg from file: {model_test_cfg}")
        if "metrics" in model_test_cfg:
            logger.info("Disabling metrics in test_cfg for inference-only mode.")
            model_test_cfg.metrics = None  # Modify the cfg dict directly
        if "save_image" in model_test_cfg:
            logger.info("Disabling save_image in test_cfg for inference-only mode.")
            model_test_cfg.save_image = False
    else:
        logger.info("No test_cfg found in config file. Proceeding without test_cfg adjustments.")
        model_test_cfg = {}  # Pass an empty dict if None, or build_model might complain

    model = build_model(cfg.model, test_cfg=model_test_cfg)  # Pass adjusted or empty test_cfg

    load_checkpoint(model, checkpoint_path, map_location=device)
    model.cfg = cfg  # Assign full cfg for pipeline adjustment later
    model.eval()
    model.to(device)
    logger.info("Model initialized successfully.")
    return model


def adjust_pipeline_for_video_inference(original_pipeline_cfg_list):
    """
    Adjusts the MMEditing pipeline configuration list for direct video inference.
    """
    logger.debug(f"Original pipeline before adjustment: {original_pipeline_cfg_list}")
    adjusted_pipeline_cfg = []
    known_meta_keys_from_api = ["key", "lq_path"]

    for p_cfg_orig in original_pipeline_cfg_list:
        p_cfg = copy.deepcopy(p_cfg_orig)
        step_type = p_cfg.get("type", "UnknownStep")
        skip_this_step = False

        if step_type in ["GenerateFileIndices", "LoadImageFromFileList", "GenerateSegmentIndices"]:
            logger.info(f"Skipping data loading/indexing pipeline step for inference: {step_type}")
            continue

        if "keys" in p_cfg:
            current_keys = p_cfg["keys"]
            modified_keys = [k for k in current_keys if k != "gt"]
            p_cfg["keys"] = modified_keys
            if current_keys != modified_keys:
                logger.debug(f"Adjusted 'keys' for step '{step_type}': from {current_keys} to {modified_keys}")
            if not modified_keys and "gt" in current_keys:
                logger.info(
                    f"Skipping GT-only pipeline step '{step_type}' as 'keys' became empty. Original keys: {current_keys}"
                )
                skip_this_step = True
        if skip_this_step:
            continue

        if "meta_keys" in p_cfg:
            current_meta_keys = p_cfg["meta_keys"]
            modified_meta_keys = []
            if step_type == "Collect":
                for k in current_meta_keys:
                    if k in known_meta_keys_from_api:
                        modified_meta_keys.append(k)
                    # elif "gt" not in k.lower(): # Optionally keep other non-GT keys
                    #     pass
            else:
                modified_meta_keys = [k for k in current_meta_keys if "gt" not in k.lower()]
            p_cfg["meta_keys"] = modified_meta_keys
            if current_meta_keys != modified_meta_keys:
                logger.debug(
                    f"Adjusted 'meta_keys' for step '{step_type}': from {current_meta_keys} to {modified_meta_keys}"
                )
        adjusted_pipeline_cfg.append(p_cfg)

    logger.debug(f"Pipeline after adjustment: {adjusted_pipeline_cfg}")
    return adjusted_pipeline_cfg


def main():
    args = parse_args()

    if not os.path.isfile(args.video_path):
        logger.error(f"Error: Input video path not found or not a file: {args.video_path}")
        return

    try:
        model = init_model_mapnet(args.config, args.checkpoint, args.device)
    except Exception as e:
        logger.exception(f"Error initializing model: {e}")
        return

    pipeline_source_attr = None
    if hasattr(model.cfg, "demo_pipeline"):  # Prefer demo_pipeline if it exists
        pipeline_source_attr = "demo_pipeline"
        original_pipeline = model.cfg.demo_pipeline
    elif hasattr(model.cfg, "test_pipeline"):
        pipeline_source_attr = "test_pipeline"
        original_pipeline = model.cfg.test_pipeline
    elif hasattr(model.cfg, "val_pipeline"):
        pipeline_source_attr = "val_pipeline"
        original_pipeline = model.cfg.val_pipeline
    else:
        logger.error("No suitable pipeline (demo_pipeline, test_pipeline, or val_pipeline) found in model.cfg.")
        return

    if not original_pipeline:
        logger.error(f"The selected pipeline '{pipeline_source_attr}' from model.cfg is empty or None.")
        return

    logger.info(f"Using '{pipeline_source_attr}' from model.cfg for inference setup.")
    adjusted_pipeline = adjust_pipeline_for_video_inference(original_pipeline)

    # Update the chosen pipeline in model.cfg
    setattr(model.cfg, pipeline_source_attr, adjusted_pipeline)

    logger.info(f"Adjusted inference pipeline ('{pipeline_source_attr}') consists of {len(adjusted_pipeline)} steps.")
    for i, step_cfg in enumerate(adjusted_pipeline):
        logger.info(
            f"  Step {i}: Type='{step_cfg.get('type', 'Unknown')}', Keys='{step_cfg.get('keys', 'N/A')}', MetaKeys='{step_cfg.get('meta_keys', 'N/A')}'"
        )

    input_fps = 30.0
    input_width, input_height, total_frames = 0, 0, 0
    try:
        video_reader_info = mmcv.VideoReader(args.video_path)
        input_fps = video_reader_info.fps if video_reader_info.fps > 0 else 30.0
        input_width = video_reader_info.width
        input_height = video_reader_info.height
        total_frames = len(video_reader_info)
        logger.info(
            f"Input video: {args.video_path} ({input_width}x{input_height} @ {input_fps:.2f} fps, {total_frames} frames)"
        )
        del video_reader_info  # Close it as restoration_video_inference will open it again
    except Exception as e:
        logger.warning(f"Could not read video metadata accurately: {e}. Using defaults.")
        # If video_reader fails, we might not be able to proceed with preview mode if it needs frame-by-frame original
        if args.mode == "preview" and (input_width == 0 or input_height == 0):
            logger.error("Cannot get input video dimensions for preview mode. Exiting.")
            return

    logger.info(f"Starting dehazing process with window_size={args.window_size}...")
    start_time = time.time()
    try:
        # restoration_video_inference is expected to return a tensor of shape (1, T, C, H, W)
        dehazed_output_tensor = restoration_video_inference(
            model,
            args.video_path,
            window_size=args.window_size,
            start_idx=0,
            filename_tmpl="",
            max_seq_len=args.max_seq_len,
            progress_bar=True,
        )
    except Exception as e:
        logger.exception(f"Error during inference: {e}")
        return
    end_time = time.time()
    logger.info(f"Inference finished in {end_time - start_time:.2f} seconds.")

    if dehazed_output_tensor is None or dehazed_output_tensor.numel() == 0:
        logger.error("Inference resulted in an empty or None output tensor.")
        return

    logger.info(f"Shape of dehazed_output_tensor before unpacking: {dehazed_output_tensor.shape}")
    logger.info(f"Number of dimensions: {dehazed_output_tensor.ndim}")

    _, num_frames_out, _, out_h, out_w = dehazed_output_tensor.shape
    logger.info(f"Output tensor shape: {dehazed_output_tensor.shape}")

    if args.mode == "save":
        output_video_fps = args.output_fps if args.output_fps is not None else input_fps
        logger.info(f"Output video will be encoded at {output_video_fps:.2f} fps.")
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if args.output_path.lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        elif not args.output_path.lower().endswith((".mp4", ".mov", ".mkv")):
            logger.warning(f"Output path {args.output_path} has an unknown extension. Using mp4v codec.")

        video_writer = cv2.VideoWriter(args.output_path, fourcc, output_video_fps, (out_w, out_h))
        if not video_writer.isOpened():
            logger.error(f"Error: Could not open video writer for {args.output_path}")
            return

        logger.info(f"Writing output video to {args.output_path}...")
        try:
            for i in tqdm(range(num_frames_out), desc="Writing frames"):
                frame_tensor = dehazed_output_tensor[:, i, :, :, :]
                img_np = cv2.cvtColor(tensor2img(frame_tensor, min_max=(0, 1), out_type=np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(img_np)
        except Exception as e:
            logger.exception(f"Error during writing video frames: {e}")
        finally:
            video_writer.release()
        logger.info(f"Dehazed video saved to {args.output_path}")

    elif args.mode == "preview":
        logger.info("Starting preview mode. Press 'q' to quit, 'p' to pause/resume.")

        # Need to read original frames again for side-by-side display
        try:
            original_video_reader = mmcv.VideoReader(args.video_path)
        except Exception as e:
            logger.error(f"Failed to open video for preview: {e}")
            return

        paused = False
        frame_idx = 0

        # Determine preview window size
        preview_h, preview_w = out_h, out_w  # dehazed frame size
        if input_height != preview_h or input_width != preview_w:
            logger.warning("Original and dehazed frame sizes differ. Preview might look misaligned if not handled.")
            # For simplicity, we'll resize original to match dehazed if necessary,
            # or a more sophisticated layout might be needed.
            # Here, we assume they are close enough or the user is aware.

        combined_width = input_width + preview_w
        combined_height = max(input_height, preview_h)

        # Scale combined image if it exceeds max_width
        display_scale_factor = 1.0
        if combined_width > args.preview_max_width:
            display_scale_factor = args.preview_max_width / combined_width
            display_width = args.preview_max_width
            display_height = int(combined_height * display_scale_factor)
        else:
            display_width = combined_width
            display_height = combined_height

        logger.info(
            f"Preview window target size: {display_width}x{display_height} (scaled by {display_scale_factor:.2f})"
        )

        try:
            for i in tqdm(range(min(len(original_video_reader), num_frames_out)), desc="Previewing frames"):
                if not paused:
                    original_frame_bgr = original_video_reader[i]  # Reads as BGR
                    if original_frame_bgr is None:
                        logger.warning(f"Could not read original frame {i}. Skipping.")
                        continue

                    dehazed_frame_tensor = dehazed_output_tensor[:, i, :, :, :]
                    dehazed_frame_bgr = tensor2img(dehazed_frame_tensor, min_max=(0, 1), out_type=np.uint8)  # BGR

                    # Resize original if its dimensions don't match dehazed (for simple stacking)
                    if original_frame_bgr.shape[:2] != dehazed_frame_bgr.shape[:2]:
                        original_frame_bgr = cv2.resize(
                            original_frame_bgr, (dehazed_frame_bgr.shape[1], dehazed_frame_bgr.shape[0])
                        )

                    # Combine frames side by side
                    combined_frame = np.concatenate((original_frame_bgr, dehazed_frame_bgr), axis=1)

                    # Scale for display
                    if display_scale_factor != 1.0:
                        display_frame = cv2.resize(
                            combined_frame, (display_width, display_height), interpolation=cv2.INTER_AREA
                        )
                    else:
                        display_frame = combined_frame

                    cv2.imshow(
                        f"Preview: Original (Left) vs Dehazed (Right) - Frame {i + 1}/{num_frames_out}", display_frame
                    )
                    frame_idx += 1

                key = cv2.waitKey(int(1000 / input_fps)) & 0xFF  # Delay based on input FPS
                if key == ord("q"):
                    logger.info("Quit key ('q') pressed. Exiting preview.")
                    break
                elif key == ord("p"):
                    paused = not paused
                    logger.info("Pause key ('p') pressed. Preview " + ("paused." if paused else "resumed."))
                elif key == ord(" "):  # Space bar also for pause/resume
                    paused = not paused
                    logger.info("Space bar pressed. Preview " + ("paused." if paused else "resumed."))

        except Exception as e:
            logger.exception(f"Error during preview: {e}")
        finally:
            cv2.destroyAllWindows()
            if "original_video_reader" in locals() and original_video_reader is not None:
                # mmcv.VideoReader might not have an explicit close, relies on GC
                pass
        logger.info("Preview finished.")

    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
