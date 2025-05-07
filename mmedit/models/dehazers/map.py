import torch
import torch.nn.functional as F
import einops

from ..registry import MODELS
from .basic_dehazer import BasicDehazer

from mmedit.models.backbones.map_backbones.map_utils import flow_warp_5d
from mmedit.models.losses.pixelwise_loss import l1_loss
from mmedit.models.backbones.map_backbones.map_utils import resize

import os
import mmcv
from torch.nn import functional as F
from mmedit.utils import get_root_logger

from mmedit.core import tensor2img

logger = get_root_logger()


def flow_loss(grid, ref, img0, img1, level):
    """
    see map_utils get_flow_from_grid
    """
    b, T, h, w, p = grid.shape
    assert p == 3, "Implementation for space-time flow warping"
    sf = 1.0 / 2 ** (level + 2)

    flow = (grid - ref).reshape(b * T, h, w, p)
    flow[:, :, :, 0] *= h
    flow[:, :, :, 1] *= w
    d = img0.shape[2]
    flow[:, :, :, 2] *= d
    assert flow.requires_grad

    # downsample and warp
    img0_lr = einops.rearrange(img0, "bT c d h w -> (bT d) c h w")
    img0_lr = F.interpolate(img0_lr, scale_factor=sf, mode="bicubic")
    img0_lr = einops.rearrange(img0_lr, "(bT d) c h w -> bT c d h w", d=d)
    img0_lr_warp = flow_warp_5d(img0_lr, flow.unsqueeze(1))
    img0_lr_warp = img0_lr_warp.squeeze(2)
    img1_lr = F.interpolate(img1, scale_factor=sf, mode="bicubic")

    return l1_loss(img0_lr_warp, img1_lr)


@MODELS.register_module()
class MAP(BasicDehazer):
    """MAP model for video dehazing.

    Paper:
        Video Dehazing via a Multi-Range Temporal Alignment Network
        with Physical Prior, CVPR, 2023

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self, generator, pixel_loss, train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg, pretrained)

        num_kv_frames = generator.get("num_kv_frames", 1)
        self.num_kv_frames = sorted(num_kv_frames) if isinstance(num_kv_frames, (list, tuple)) else [num_kv_frames]

    @staticmethod
    def _get_output_from_dict(x):
        if isinstance(x, dict):
            return x["out"]
        return x

    def forward_train(self, lq, gt):
        """Training forward function.
        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            Tensor: Output tensor.
        """
        assert lq.ndim == 5 and lq.shape[1] > 1, f"Video dehazing methods should have input t > 1 but get: {lq.shape}"
        losses = dict()
        output = self.generator(lq)
        loss_name = None
        if isinstance(output, dict):
            for key in output.keys():
                if key == "out":
                    loss_key = self.pixel_loss(output[key], gt)
                elif key == "img_01":
                    continue
                elif key in ("aux_j", "aux_i"):
                    loss_name = key.replace("aux_", "phy-")
                    loss_key, lambda_phy = 0.0, 0.2
                    num_stages = len(output[key])
                    gt_key = gt if key == "aux_j" else output["img_01"]
                    for s in range(num_stages):
                        loss_weight = lambda_phy / (2**s)  # be careful about the stage notation
                        loss_key += loss_weight * self.pixel_loss(output[key][s], gt_key)
                elif key.startswith("pos"):
                    # flow loss
                    assert len(output[key]) <= 4, (
                        f"pos should be less than or equal to 4 stages but get {len(output[key])}."
                    )
                    loss_name = "flow"
                    loss_key, lambda_flow = 0.0, 0.04
                    num_stages = len(output[key])
                    for s in range(num_stages):
                        assert output[key][s].shape[-1] == 3
                        loss_weight = lambda_flow / 2**s  # be careful about the stage notation
                        b, T, c, h, w = gt.shape
                        num_groups = output[key][s].size(3)
                        for g in range(num_groups):
                            # pos[s] is in shape (b, T, nr, g, h, w, 3)
                            num_kv_frames = self.num_kv_frames  # assume kv_frames to be [1, 2, 3, etc...]
                            img0s = []
                            for step in range(max(num_kv_frames)):
                                indices = torch.clip(torch.arange(T) - (step + 1), 0).to(gt.device)
                                img0 = torch.index_select(gt, dim=1, index=indices)
                                img0 = img0.reshape(b * T, 3, h, w)
                                img0s.append(img0)
                            img0s = torch.stack(img0s, dim=2)
                            for r, kv_frames in enumerate(num_kv_frames):
                                grid = output[key][s][:, :, r, g, :, :, :].clone()
                                ref = output[key.replace("pos", "ref")][s].clone()
                                assert not ref.requires_grad
                                img0 = img0s[:, :, :kv_frames].clone()
                                img1 = gt.clone().reshape(b * T, 3, h, w)
                                loss_key += loss_weight * flow_loss(grid, ref, img0, img1, s)
                elif key.startswith("ref"):
                    continue
                loss_name = loss_name or key
                losses[f"loss_{loss_name}"] = loss_key
            output = self._get_output_from_dict(output)
        else:
            loss_pix = self.pixel_loss(output, gt)
            losses["loss_pix"] = loss_pix
        outputs = dict(
            losses=losses, num_samples=len(gt.data), results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu())
        )
        return outputs

    def forward_test(
        self,
        lq,
        gt=None,
        meta=None,
        save_image=False,  # APIからの直接指定を優先 (デフォルトFalse)
        save_path=None,  # APIからの直接指定を優先
        iteration=None,
    ):  # APIからの直接指定を優先
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor, optional): GT image. Default: None.
            meta (list[dict], optional): Meta data. Default: None.
                Expected to be a list containing one dict: [{'key': 'sequence_name', ...}].
            save_image (bool): Whether to save image. Default: False.
            save_path (str, optional): Path to save image. Default: None.
            iteration (int, optional): Iteration for the filename. Default: None.

        Returns:
            dict: Output results.
        """
        # --- デバッグ用ログ (必要に応じて有効化) ---
        # logger.debug(f"[MAP.forward_test] CALLED. save_image (arg): {save_image}, iteration (arg): {iteration}, save_path (arg): {save_path}")
        # if self.test_cfg is not None:
        #     logger.debug(f"[MAP.forward_test] self.test_cfg: {self.test_cfg}")
        # else:
        #     logger.debug("[MAP.forward_test] self.test_cfg is None")
        # if meta is not None:
        #     logger.debug(f"[MAP.forward_test] meta (arg): {meta}")
        # else:
        #     logger.debug("[MAP.forward_test] meta (arg) is None")
        # --- ここまでデバッグ用 ---

        # 1. GTリサイズ処理 (gtが存在し、形状がlqと異なる場合)
        # この部分は dehaze_video.py からの呼び出しでは gt=None なので実行されない想定。
        # データセット評価などで gt が渡された場合のためのロジック。
        if gt is not None and lq.shape[-2:] != gt.shape[-2:]:  # H, W のみが異なる場合を主に想定
            logger.warning(
                f"GT shape {gt.shape} and LQ shape {lq.shape} spatial dimensions mismatch during testing. "
                f"Attempting to resize GT to LQ's H, W: {lq.shape[-2:]}."
            )
            # lq, gt ともに N(T)CHW を想定。T次元の扱いは元のコードが曖昧なので注意。
            # ここでは単純に空間次元のみを合わせる試み。
            # F.interpolate は入力が4D (NCHW) または 5D (NCDHW) を期待。
            # MAP-Netはビデオ処理なので、lq, gt は N, T, C, H, W (5D) の可能性がある。
            # PyTorch の F.interpolate は (N, C, D_in, H_in, W_in) -> (N, C, D_out, H_out, W_out) または
            # (N, C, H_in, W_in) -> (N, C, H_out, W_out)
            # T を D (Depth) とみなすか、各フレームごとに処理するか。
            # ここでは、各フレームごとにリサイズするのが安全か。元のコードは squeeze(0) していたので N=1 を想定。

            if lq.ndim == 5 and gt.ndim == 5:  # NTCHW
                if lq.shape[0] == 1 and gt.shape[0] == 1:  # Batch size 1
                    # (T, C, H, W) を (C, T*H, W) のように見せかけてリサイズし、戻すのは難しい。
                    # 各フレームを処理するのが妥当。
                    resized_frames_gt = []
                    for t_idx in range(gt.shape[1]):
                        frame_gt = gt[:, t_idx, ...]  # (N,C,H,W)
                        frame_gt_resized = F.interpolate(
                            frame_gt, size=lq.shape[-2:], mode="bilinear", align_corners=False
                        )
                        resized_frames_gt.append(frame_gt_resized)
                    gt = torch.stack(resized_frames_gt, dim=1)  # (N,T,C,H_new,W_new)
                else:
                    logger.warning(
                        "GT/LQ shape mismatch with batch size > 1 for 5D tensors. Resize logic might be inexact."
                    )
                    # ここでは単純に最後の2次元だけ合わせる試み（F.interpolateはよしなにやってくれるが注意）
                    if gt.numel() > 0:  # 空のテンソルでないことを確認
                        gt_shape_prefix = gt.shape[:-2]
                        gt = F.interpolate(
                            gt.view(-1, gt.shape[-3], gt.shape[-2], gt.shape[-1]),
                            size=lq.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        gt = gt.view(*gt_shape_prefix, gt.shape[-2], gt.shape[-1])
                    else:
                        logger.warning("GT is an empty tensor, skipping resize.")

            elif lq.ndim == 4 and gt.ndim == 4:  # NCHW
                if gt.numel() > 0:
                    gt = F.interpolate(gt, size=lq.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    logger.warning("GT is an empty tensor, skipping resize.")
            else:
                logger.warning(
                    f"Cannot reliably resize GT due to differing dimensions (LQ: {lq.ndim}D, GT: {gt.ndim}D) or unhandled batching."
                )

        # 2. モデルによる推論実行
        generator_output = self.generator(lq)

        # generator_output (pdbで確認した結果 {'out': tensor(...)} ) から 'out' キーでテンソルを取り出す
        if isinstance(generator_output, dict):
            if "out" in generator_output:
                output = generator_output["out"]
            elif "output" in generator_output:  # フォールバック
                logger.warning("Generator output dict using 'output' key instead of 'out'.")
                output = generator_output["output"]
            else:
                available_keys = list(generator_output.keys())
                logger.error(
                    f"Generator output is a dict, but key 'out' (or 'output') not found. Available keys: {available_keys}"
                )
                raise KeyError(f"Key 'out' not found in generator_output dict. Available keys: {available_keys}")
        elif torch.is_tensor(generator_output):
            output = generator_output
        else:
            logger.error(f"Generator output type is not a Tensor or dict: {type(generator_output)}")
            raise TypeError(f"Generator output type is not a Tensor or dict: {type(generator_output)}")

        # output がテンソルであることを最終確認
        if not torch.is_tensor(output):
            logger.error(f"Processed 'output' (from generator) is not a Tensor. Type: {type(output)}")
            raise TypeError(f"Processed 'output' (from generator) is not a Tensor. Type: {type(output)}")

        # 3. 結果辞書の作成
        # metrics計算は model.test_cfg.metrics が存在し、かつ gt が None でない場合のみ
        # dehaze_video.py 側で model.test_cfg.metrics = None に設定しているので、通常この分岐には入らない
        perform_evaluation = self.test_cfg is not None and self.test_cfg.get("metrics", None) and gt is not None

        if perform_evaluation:
            # logger.debug(f"[MAP.forward_test] Performing evaluation with metrics: {self.test_cfg['metrics']}")
            eval_result = self.evaluate(output, gt)  # BasicDehazer.evaluate を想定
            results = dict(eval_result=eval_result)
        else:
            # logger.debug("[MAP.forward_test] Not performing evaluation. Returning lq, output.")
            if not torch.is_tensor(lq):
                logger.error(f"Input 'lq' is not a Tensor in forward_test before .cpu(). Type: {type(lq)}")
                raise TypeError(f"Input 'lq' is not a Tensor in forward_test. Type: {type(lq)}")
            # output は上でテンソルであることを確認済み
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:  # gt があれば、評価はしなくても結果に含める (テンソルであると期待)
                if torch.is_tensor(gt):
                    results["gt"] = gt.cpu()
                else:
                    logger.warning(f"gt was provided but is not a tensor, type: {type(gt)}. Not adding to results.")

        # 4. 画像保存処理 (forward_testの引数 save_image が True の場合のみ)
        # dehaze_video.py からは save_image=False (デフォルト) で呼ばれるので、通常このブロックは実行されない。
        if save_image:
            # logger.debug(f"[MAP.forward_test] ENTERING save_image block. meta: {meta}, iteration: {iteration}, save_path: {save_path}")
            if save_path is None:
                logger.error("save_path must be provided if save_image is True.")
                raise ValueError("save_path must be provided if save_image is True.")

            _iteration = iteration if iteration is not None else 0

            folder_name = "unknown_sequence"
            if meta is not None and isinstance(meta, list) and len(meta) > 0:
                if isinstance(meta[0], dict) and "key" in meta[0]:
                    folder_name = meta[0]["key"]
                else:
                    logger.warning(
                        f"meta[0] is not a dict or does not contain 'key'. Using default folder_name. meta[0]: {meta[0]}"
                    )
            else:
                logger.warning(f"meta is None or empty for save_image. Using default folder_name. meta: {meta}")

            if output.ndim == 5:  # N, T, C, H, W
                num_batch, num_frames, _, _, _ = output.shape
                for batch_idx in range(num_batch):
                    current_folder_name_base = folder_name
                    if num_batch > 1:
                        if (
                            meta is not None
                            and len(meta) > batch_idx
                            and isinstance(meta[batch_idx], dict)
                            and "key" in meta[batch_idx]
                        ):
                            current_folder_name_base = meta[batch_idx]["key"]
                        else:
                            current_folder_name_base = f"{folder_name}_batch{batch_idx}"

                    final_save_folder = os.path.join(save_path, current_folder_name_base)
                    mmcv.mkdir_or_exist(final_save_folder)

                    for frame_idx in range(num_frames):
                        output_frame_tensor = output[batch_idx : batch_idx + 1, frame_idx, ...]
                        filename = f"{current_folder_name_base}_{_iteration + frame_idx:08d}.png"
                        save_img_full_path = os.path.join(final_save_folder, filename)
                        img_np = tensor2img(output_frame_tensor)
                        mmcv.imwrite(img_np, save_img_full_path)

                        if self.test_cfg and self.test_cfg.get("save_lq", False) and lq is not None:
                            if lq.ndim == 5 and lq.shape[0] > batch_idx and lq.shape[1] > frame_idx:
                                lq_frame_tensor = lq[batch_idx : batch_idx + 1, frame_idx, ...]
                                lq_filename = f"{current_folder_name_base}_{_iteration + frame_idx:08d}_lq.png"
                                save_lq_full_path = os.path.join(final_save_folder, lq_filename)
                                img_lq_np = tensor2img(lq_frame_tensor)
                                mmcv.imwrite(img_lq_np, save_lq_full_path)
            elif output.ndim == 4:  # N, C, H, W
                # (静止画保存ロジック - 前回提示のものを参照)
                num_batch = output.shape[0]
                for batch_idx in range(num_batch):
                    current_name_base = folder_name
                    if (
                        num_batch > 1
                        and meta is not None
                        and len(meta) > batch_idx
                        and isinstance(meta[batch_idx], dict)
                        and "key" in meta[batch_idx]
                    ):
                        current_name_base = meta[batch_idx]["key"]
                    elif num_batch > 1:
                        current_name_base = f"{folder_name}_batch{batch_idx}"

                    filename = f"{current_name_base}_{_iteration:08d}.png"
                    save_img_full_path = os.path.join(save_path, filename)
                    mmcv.mkdir_or_exist(save_path)

                    img_np = tensor2img(output[batch_idx : batch_idx + 1, ...])
                    mmcv.imwrite(img_np, save_img_full_path)
                    # (LQ保存ロジックも同様に)
            else:
                logger.error(f"Unsupported output tensor dimension for saving: {output.ndim}")

        return results
