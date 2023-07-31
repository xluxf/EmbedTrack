import os
import sys
from embedtrack.infer.infer_ctc_data import inference
import getopt


def read_args(argv):
    arg_help = f"{argv[0]} -d <data_path> -s <sequence> -m <model_path>"

    try:
        opts, args = getopt.getopt(argv[1:], "hi:u:o:", ["help", "data_path=",
                                                         "sequence=", "model_path="])
    except:
        print(arg_help)
        sys.exit(2)

    args_dict = {}
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-d", "--data_path"):
            args_dict['data_path'] = arg
        elif opt in ("-s", "--sequence"):
            args_dict['sequence'] = arg
        elif opt in ("-m", "--model_path"):
            args_dict['model_path'] = arg

    return args_dict


def main(data_path, seq, model_dir, batch_size=128):
    """
    load model
    infer model in a given data folder
    stores EmbedTrack outputs into the folder <data_path>/<seq>_DATA

    Parameters:
    -----------
    data_path : str
        path to data
    seq : str
        sequence to be tracked, e.g. '01', '02'
    model_path : str
        path to the model
    batch_size : int
        size of the batch
    """

    img_path = os.path.join(data_path, seq)
    assert os.path.isdir(img_path), img_path

    if not os.path.exists(model_dir):
        print(f"no trained model in {model_dir}")
        return

    model_path = os.path.join(model_dir, "best_iou_model.pth")
    config_file = os.path.join(model_dir, "config.json")
    inference(img_path, model_path, config_file, batch_size=batch_size)


if __name__ == "__main__":
    args = read_args(sys.argv)

    print('running embedtrack with the following arguments:', args)
    main(args['data_path'], args['sequence'], args['model_path'])

