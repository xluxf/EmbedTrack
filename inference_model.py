import os
from datetime import datetime
from pathlib import Path
from time import time
import shutil
from embedtrack.ctc_metrics.eval_ctc import calc_ctc_scores
from embedtrack.infer.infer_ctc_data import inference

DATASETS = ["BF-C2DL-HSC"]
PROJECT_PATH = Path.cwd()

RAW_DATA_PATHS = [os.path.join(PROJECT_PATH, "ctc_raw_data/train")]
                  
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
RES_PATH = os.path.join(PROJECT_PATH, "results10")

# Adam optimizer; normalize images; OneCycle LR sheduler; N epochs
MODEL_NAME = "adam_norm_onecycle_allcrops_15"
MODEL_NAME = "."

BATCH_SIZE = 128

SEQS = ["09"]

# test setting
#SEQS = ["01", "02"]
#RAW_DATA_PATHS = [os.path.join(PROJECT_PATH, "ctc_raw_data/train"),
#                  os.path.join(PROJECT_PATH, "ctc_raw_data/challenge")]


for data_set in DATASETS:
    for raw_data_path in RAW_DATA_PATHS:
        for data_id in SEQS:
            img_path = os.path.join(raw_data_path, data_set, data_id)

            model_dir = os.path.join(MODEL_PATH, data_set, MODEL_NAME)
            if not os.path.exists(model_dir):
                print(f"no trained model for data set {data_set}")
                continue

            # time stamps
            timestamps_trained_models = [
                datetime.strptime(time_stamp, "%Y-%m-%d---%H-%M-%S")
                for time_stamp in os.listdir(model_dir) if '20' in time_stamp
            ]
            timestamps_trained_models.sort()
            #last_model = timestamps_trained_models[-1].strftime("%Y-%m-%d---%H-%M-%S")
            last_model = '.'
            model_path = os.path.join(model_dir, last_model, "best_iou_model.pth")
            config_file = os.path.join(model_dir, last_model, "config.json")
            t_start = time()
            inference(img_path, model_path, config_file, batch_size=BATCH_SIZE)
            t_end = time()

            run_time = t_end - t_start
            print(f"Image sequence: {img_path}")
            print(f"Inference Time {img_path}: {run_time}s")



            res_path = os.path.join(RES_PATH, data_set, MODEL_NAME, os.path.basename(raw_data_path), data_id+"_RES")

            # keep all in res_path
            print('res_path', res_path)
            continue
            
            if not os.path.exists(os.path.dirname(res_path)):
                os.makedirs(os.path.dirname(res_path))
                shutil.move(img_path + "_RES", res_path)


        '''
        if False and os.path.basename(raw_data_path) == "train":
        metrics = calc_ctc_scores(Path(res_path), Path(img_path+"_GT"))
        print(metrics)
        '''

