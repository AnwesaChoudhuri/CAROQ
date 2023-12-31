"""
CAROQ Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import time

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluators,
    verify_results,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)

from caroq_video import VPSEvaluator
from caroq_video import YTVISDatasetMapper, YTVISEvaluator
from caroq_video import get_detection_dataset_dicts as video_detection_dataset_dicts
from caroq_video import build_detection_train_loader as video_train_loader
from caroq_video import build_detection_test_loader as video_test_loader

# from cityscapes_vps.mmdet.datasets import build_dataset as build_vps_dataset

from detectron2.evaluation.evaluator import DatasetEvaluator

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

import pdb

# from caroq.data.datasets.register_mots import register_all_mots

from caroq.mots_evaluation import MOTSEvaluator
from caroq import add_maskformer2_config


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):

            data_loader = cls.build_test_loader(cfg)

            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        return results

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if cfg.DATASETS.TRAIN[0].find("mots")>-1:
            gt_dir = cfg.TEST.GT_DIR
            output_dir = cfg.OUTPUT_DIR+"/"+cfg.TEST.OUTPUT_DIR
            eval_mode = cfg.TEST.EVAL_MODE
            seqmap_filename=cfg.TEST.SEQMAP

            evaluator = MOTSEvaluator(gt_dir = gt_dir,
                     output_dir = output_dir,
                     eval_mode=eval_mode,
                     seqmap_filename=seqmap_filename
                     )

        elif cfg.INPUT.DATASET_MAPPER_NAME == "youtube_vis":
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference/")
                os.makedirs(output_folder, exist_ok=True)
            evaluator = YTVISEvaluator(dataset_name, cfg, True, output_folder)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "cityscapes_vps":
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "panoptic_pred/")
                os.makedirs(output_folder, exist_ok=True)
                truth_dir = "data/Cityscapes/cityscapes_vps/val/panoptic_video/"
                gt_json = "data/Cityscapes/cityscapes_vps/panoptic_gt_val_city_vps.json"
            evaluator = VPSEvaluator(output_folder, truth_dir, gt_json)

        return evaluator

    @classmethod
    def build_test_loader(cls, cfg):

        if cfg.INPUT.DATASET_MAPPER_NAME in ("youtube_vis", "cityscapes_vps"):
            mapper = YTVISDatasetMapper(cfg, is_train=False)
            dataset_name = cfg.DATASETS.TEST[0]
            return video_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):

        if cfg.INPUT.DATASET_MAPPER_NAME == "youtube_vis":
            mapper = YTVISDatasetMapper(cfg, is_train=True)
            dataset_name = cfg.DATASETS.TRAIN[0]
            dataset_dict = video_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subset=cfg.INPUT.SUBSET,
                train=True,
            )
            return video_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "cityscapes_vps":
            mapper = YTVISDatasetMapper(cfg, is_train=True)
            dataset_name = cfg.DATASETS.TRAIN[0]
            dataset_dict = video_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subset=cfg.INPUT.SUBSET,
                train=True,
            )
            return video_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former"
    )
    return cfg


def main(args):
    cfg = setup(args)

    if cfg.INPUT.DATASET_MAPPER_NAME == "cityscape_vps":
        time1 = time.time()
        register_all_vps(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
