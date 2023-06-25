

import json
from detectron2.evaluation import DatasetEvaluator

import os

class VPSEvaluator(DatasetEvaluator):
    def __init__(self,output_folder):
        self.output_folder=output_folder.replace("inference","panoptic_pred/")


    def evaluate(self):

        annotations_all=[]
        json_files=os.listdir(self.output_folder)
        json_files=[j for j in json_files if j.startswith("pred_")]
        numbers=[int(j.replace("pred_","").replace(".json","")) for j in json_files]
        numbers.sort()
        json_files=["pred_"+str(j)+".json" for j in numbers]

        for json_file in json_files:
            if json_file.startswith("pred_"):
                data=json.load((open(self.output_folder+json_file)))
                annotations_all=annotations_all+data

        json.dump({"annotations": annotations_all},
                open(self.output_folder+"/pred.json",'w'))
        return
