

from functools import partial
from multiprocessing import Pool
import sys
import pdb
sys.path.append("../")
import external.TrackEval.trackeval as trackeval
from external.mots_tools.mots_vis.visualize_mots import load_seqmap, process_sequence
from detectron2.evaluation import DatasetEvaluator

class MOTSEvaluator(DatasetEvaluator):
    def __init__(self, gt_dir=".", output_dir = ".", eval_mode="train", seqmap_filename="val.seqmap"):
        self.gt_dir=gt_dir
        self.output_dir=output_dir
        self.eval_mode=eval_mode
        self.seqmap_filename=seqmap_filename


    def evaluate(self):

        print("Evaluating...")
        results=eval(output_dir=self.output_dir, seqmap=self.seqmap_filename, gt_dir=self.gt_dir)

        print("Saving first 50 frame visualization...")
        try:
            vis(self.output_dir+"/Instances_txt/", self.gt_dir+"/images/", self.output_dir+"/Superimposed/", self.seqmap_filename)
        except:
            print("error")
        for cls in results: 
            for met in results[cls]:
                results[cls][met]=float(results[cls][met])

        return results


def eval(output_dir="", seqmap=".", gt_dir="."):
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    if gt_dir.find("KITTI")==-1:
        default_dataset_config = trackeval.datasets.MOTSChallenge.get_default_dataset_config()

    else:
        default_dataset_config = trackeval.datasets.KittiMOTS.get_default_dataset_config()
    
    default_dataset_config["TRACKERS_FOLDER"]=output_dir[:output_dir[:-1].find("/")]
    default_dataset_config["GT_FOLDER"]=gt_dir
    default_dataset_config['SEQMAP_FILE']=seqmap
    default_dataset_config['TRACKERS_TO_EVAL'] =[output_dir[output_dir[:-1].find("/")+1:]]
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    if gt_dir.find("KITTI")==-1:
        dataset_list = [trackeval.datasets.MOTSChallenge(dataset_config)]
    else:
        dataset_list = [trackeval.datasets.KittiMOTS(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    results=evaluator.evaluate(dataset_list, metrics_list)
    return results


def vis(tracks_folder, img_folder, output_folder, seqmap_filename, maxf=50):
    seqmap, max_frames = load_seqmap(seqmap_filename)
    for k in max_frames.keys():
        max_frames[k]=maxf
    process_sequence_part = partial(process_sequence, max_frames=max_frames,tracks_folder=tracks_folder, img_folder=img_folder, output_folder=output_folder)
    with Pool(10) as pool:
        pool.map(process_sequence_part, seqmap)

# eval(output_dir="../saved_outputs/kittimots_0/new/", seqmap="../external/mots_tools/mots_eval/val_KITTIMOTS.seqmap", gt_dir="../data/KITTI_MOTS/val")
# vis("../saved_outputs/kittimots_0/new/Instances_txt/", "../data/KITTI_MOTS/val/images/", "../saved_outputs/kittimots_0/new/Superimposed/","../external/mots_tools/mots_eval/val_KITTIMOTS.seqmap")
