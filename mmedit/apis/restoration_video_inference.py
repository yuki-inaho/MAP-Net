# mmedit/apis/restoration_video_inference.py
import glob
import os.path as osp

import mmcv
import numpy as np
import torch
from tqdm import tqdm  # tqdm をインポート

from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = (".mp4", ".mov")


def pad_sequence(data, window_size):
    padding = window_size // 2
    # Ensure data is 5D: (N, T, C, H, W)
    # Original data from video_reader is (T, H, W, C), then flipped to (T, H, W, C)
    # pipeline turns it into (T, C, H, W), then unsqueezed to (1, T, C, H, W)
    # So, data should be (N, T, C, H, W) here.

    # padding calculation needs to be careful with T dimension
    # data shape: (N, T_orig, C, H, W)
    # We are padding along the T dimension (dim=1)

    # Example: T_orig=5, padding=2. We need 2 frames from start and 2 from end.
    # start_pad_source: data[:, 1:1+padding] but flipped. Indices: 1, 2 (0-indexed from original T)
    # end_pad_source: data[:, -1-padding:-1] but flipped. Indices: T-1-padding, ..., T-2 (0-indexed from original T)

    # Corrected padding logic assuming T is large enough:
    # If T_orig < padding, this logic might fail or give unexpected results.
    # Consider a case where T_orig = 3, window_size = 5, padding = 2.
    # We need to pad 2 frames at the beginning and 2 at the end.
    # A robust way is to replicate border frames if T_orig is too small.

    num_frames_original = data.size(1)

    if num_frames_original == 0:  # Handle empty sequence
        return data

    # Start padding
    # Replicate the first frame 'padding' times if sequence is too short
    start_padding_frames = []
    for i in range(padding):
        # Use frame at index min(i, num_frames_original - 1) and flip if i > 0, or just replicate first frame
        # A simpler approach for border replication:
        idx_to_replicate = 0
        start_padding_frames.append(data[:, idx_to_replicate : idx_to_replicate + 1, ...])
    if start_padding_frames:
        start_padding_tensor = torch.cat(list(reversed(start_padding_frames)), dim=1)  # Reversed to mimic flip effect
    else:  # if padding is 0
        start_padding_tensor = torch.empty_like(data[:, 0:0, ...])

    # End padding
    end_padding_frames = []
    for i in range(padding):
        idx_to_replicate = num_frames_original - 1
        end_padding_frames.append(data[:, idx_to_replicate : idx_to_replicate + 1, ...])
    if end_padding_frames:
        end_padding_tensor = torch.cat(list(reversed(end_padding_frames)), dim=1)  # Reversed to mimic flip effect
    else:  # if padding is 0
        end_padding_tensor = torch.empty_like(data[:, 0:0, ...])

    # Original flip logic (might be better if sequence is long enough)
    # This assumes num_frames_original > padding
    # start_pad_orig = data[:, 1 : 1 + padding, ...].flip(1) if num_frames_original > padding else start_padding_tensor
    # end_pad_orig   = data[:, -1 - padding : -1, ...].flip(1) if num_frames_original > padding else end_padding_tensor

    # Using the replication padding for robustness
    data = torch.cat([start_padding_tensor, data, end_padding_tensor], dim=1)
    return data


def restoration_video_inference(
    model, img_dir, window_size, start_idx, filename_tmpl, max_seq_len=None, progress_bar=False
):  # 新しい引数 progress_bar
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video or image sequence.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework (not fully handled here for progress).
        start_idx (int): The index corresponds to the first frame in the
            sequence (for image sequence folder).
        filename_tmpl (str): Template for file name (for image sequence folder).
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.
        progress_bar (bool): Whether to display a tqdm progress bar.
            Default: False.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device

    # build the data pipeline
    # (dehaze_video.py 側で調整済みのパイプラインが model.cfg に設定されている前提)
    if model.cfg.get("demo_pipeline", None):
        pipeline_cfg = model.cfg.demo_pipeline
    elif model.cfg.get("test_pipeline", None):
        pipeline_cfg = model.cfg.test_pipeline
    elif model.cfg.get("val_pipeline", None):
        pipeline_cfg = model.cfg.val_pipeline
    else:
        raise ValueError("No suitable pipeline found in model.cfg (demo_pipeline, test_pipeline, or val_pipeline)")

    # Check if input is a video file or image sequence directory
    file_extension = osp.splitext(img_dir)[1]
    is_video_file = file_extension.lower() in VIDEO_EXTENSIONS

    num_total_frames_for_progress = 0  # tqdm の total 用

    if is_video_file:
        video_reader = mmcv.VideoReader(img_dir)
        num_total_frames_for_progress = len(video_reader)
        data_dict = dict(lq=[], lq_path=img_dir, key=osp.basename(img_dir))  # lq_path for consistency

        # tqdm で動画フレームの読み込み進捗を表示 (オプション)
        frames_iterator = (
            tqdm(video_reader, total=num_total_frames_for_progress, desc="Reading video frames")
            if progress_bar
            else video_reader
        )
        for frame in frames_iterator:
            # MMEditingパイプラインはRGBを期待することが多い。OpenCVはBGRで読む。
            # パイプライン内で to_rgb=True があればそこで変換される。
            # ここでは、元コードに合わせて flip(axis=2) を行う (RGB <-> BGR)
            data_dict["lq"].append(np.flip(frame, axis=2))
        del video_reader  # Close video reader

        # パイプラインからデータローディング関連のステップを削除する処理は
        # dehaze_video.py 側で事前に行っているため、ここでは不要。
        # pipeline_cfg は調整済みのものを使う。

    else:  # Image sequence directory
        # This part assumes GenerateSegmentIndices or similar is the first step
        # and it populates lq_path, key, sequence_length.
        # The progress for image sequence loading is harder to track here without
        # deeper modification of how GenerateSegmentIndices works with tqdm.
        # For simplicity, progress bar for image sequence is mainly for model inference part.

        # 既存のロジックで sequence_length を取得して tqdm の total に使う
        if pipeline_cfg and pipeline_cfg[0]["type"] == "GenerateSegmentIndices":
            # これはあくまで推定。実際のフレーム数はLoadImageFromFileListで決まる。
            # しかし、GenerateSegmentIndicesがないとフォルダ処理が難しい。
            # dehaze_video.py側でpipeline_cfg[0]がこれであることを保証する必要がある。
            try:
                # このパスの画像数を数える (これは単なる推定)
                # 正確には、GenerateSegmentIndicesが生成するリストの長さ
                num_total_frames_for_progress = len(
                    glob.glob(osp.join(img_dir, filename_tmpl.format(start_idx)))
                )  # filename_tmplに依存
                if num_total_frames_for_progress == 0:  # ワイルドカードの場合もある
                    num_total_frames_for_progress = len(glob.glob(osp.join(img_dir, "*")))

            except Exception:
                num_total_frames_for_progress = 0  # 不明な場合は0
        else:
            # dehaze_video.py 側で調整済みのはずなので、ここは通らない想定
            # ただ、直接このAPIがフォルダに対して呼ばれる場合を考慮すると...
            logger = mmcv.utils.get_logger("mmedit")
            logger.warning(
                "First pipeline step is not GenerateSegmentIndices. Progress for image sequence might be inaccurate."
            )
            num_total_frames_for_progress = 0  # 不明

        lq_folder = osp.dirname(img_dir)  # img_dir がフォルダそのものを指す場合
        key = osp.basename(img_dir)
        # この data_dict は Compose(pipeline_cfg) に渡される
        data_dict = dict(
            lq_path=img_dir,  # GenerateSegmentIndices が使う
            gt_path="",  # パイプライン調整で不要になっているはず
            key=key,
            # sequence_length は GenerateSegmentIndices が内部で設定することを期待
            # start_idx と filename_tmpl も GenerateSegmentIndices が使う
        )
        # Note: GenerateSegmentIndices should be adjusted or made aware of start_idx/filename_tmpl
        # if it's the first step. The original code modifies pipeline_cfg[0] directly.
        if pipeline_cfg and pipeline_cfg[0]["type"] == "GenerateSegmentIndices":
            pipeline_cfg[0]["start_idx"] = start_idx
            pipeline_cfg[0]["filename_tmpl"] = filename_tmpl

    # Compose the pipeline
    pipeline_processor = Compose(pipeline_cfg)
    processed_data = pipeline_processor(data_dict)  # ここでlqがnumpy list -> tensor listに変わる

    # Ensure 'lq' is a tensor and on CPU before padding and model inference
    # The pipeline (FramesToTensor) should produce a tensor for 'lq'.
    # It's typically (T, C, H, W) at this point.
    if "lq" not in processed_data or not isinstance(processed_data["lq"], torch.Tensor):
        raise ValueError("Pipeline did not produce a tensor for 'lq' key.")

    data_tensor = processed_data["lq"].cpu().unsqueeze(0)  # Add batch dim: (1, T, C, H, W)

    # --- Model Inference ---
    with torch.no_grad():
        if window_size > 0:  # Sliding window framework
            if data_tensor.size(1) == 0:  # Handle empty sequence after processing
                return torch.empty_like(data_tensor)

            data_padded = pad_sequence(data_tensor, window_size)  # pad_sequence expects (N,T,C,H,W)
            result_frames = []

            # スライディングウィンドウのループ範囲を計算
            # 出力されるフレーム数は、パディング前の元のフレーム数と同じはず
            # ループ回数は num_total_frames_for_progress (元のフレーム数)
            num_output_frames = data_tensor.size(1)  # パディング前のフレーム数

            # tqdm を設定 (スライディングウィンドウの反復回数)
            iterations = range(num_output_frames)
            if progress_bar:
                iterations = tqdm(iterations, total=num_output_frames, desc="Dehazing (sliding window)")

            for i in iterations:  # i は出力フレームのインデックス (0 to T_orig-1)
                # data_padded から現在のウィンドウを取得
                # ウィンドウの開始インデックスは i (パディングを考慮したインデックス)
                # ウィンドウの終了インデックスは i + window_size
                current_window_data = data_padded[:, i : i + window_size, ...].to(device)

                # モデル推論 (結果は辞書 {'output': tensor} を期待)
                model_output_dict = model(lq=current_window_data, test_mode=True)

                if "output" not in model_output_dict or not torch.is_tensor(model_output_dict["output"]):
                    raise ValueError("Model output does not contain 'output' tensor or it's not a tensor.")

                # スライディングウィンドウでは、通常ウィンドウの中央フレームを出力として使う
                # window_size が奇数の場合、中央は padding = window_size // 2 番目のフレーム
                # 例: window_size=5, padding=2. 入力は frame0,1,2,3,4. 中央は frame2 (インデックス2)
                center_frame_in_window = model_output_dict["output"][:, window_size // 2, ...].cpu()  # (N,C,H,W)
                result_frames.append(center_frame_in_window)

            if not result_frames:  # もし空なら空のテンソルを返す
                return torch.empty_like(data_tensor)

            result_tensor = torch.stack(result_frames, dim=1)  # (N, T_orig, C, H, W)

        else:  # Recurrent framework (or process whole sequence at once if window_size <= 0)
            if max_seq_len is None:
                # tqdm を設定 (1回の処理なのであまり意味はないが形式として)
                iterations = range(1)
                if progress_bar:
                    iterations = tqdm(iterations, total=1, desc="Dehazing (full sequence)")

                for _ in iterations:  # 1回だけループ
                    model_output_dict = model(lq=data_tensor.to(device), test_mode=True)
                    if "output" not in model_output_dict or not torch.is_tensor(model_output_dict["output"]):
                        raise ValueError("Model output does not contain 'output' tensor or it's not a tensor.")
                    result_tensor = model_output_dict["output"].cpu()
            else:  # max_seq_len で分割処理
                result_segments = []
                num_segments = (data_tensor.size(1) + max_seq_len - 1) // max_seq_len

                iterations = range(0, data_tensor.size(1), max_seq_len)
                if progress_bar:
                    iterations = tqdm(iterations, total=num_segments, desc=f"Dehazing (segments of {max_seq_len})")

                for i in iterations:
                    segment_data = data_tensor[:, i : i + max_seq_len, ...].to(device)
                    model_output_dict = model(lq=segment_data, test_mode=True)
                    if "output" not in model_output_dict or not torch.is_tensor(model_output_dict["output"]):
                        raise ValueError("Model output does not contain 'output' tensor or it's not a tensor.")
                    result_segments.append(model_output_dict["output"].cpu())
                result_tensor = torch.cat(result_segments, dim=1)  # Concatenate along time dimension T

    return result_tensor
