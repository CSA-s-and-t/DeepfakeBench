import argparse
import os
import shutil
import sys


detector_config_map = {
    'xception': "./training/config/detector/xception.yaml",
    'ucf': "./training/config/detector/ucf.yaml",
    'capsule_net': "./training/config/detector/capsule_net.yaml",
    'meso4Inception': "./training/config/detector/meso4Inception.yaml"
}

detector_weights_map = {
    'xception': "./training/weights/xception_best.pth",
    'ucf': "./training/weights/ucf_best.pth",
    'capsule_net': "./training/weights/capsule_best.pth",
    'meso4Inception': "./training/weights/meso4Incep_best.pth"
}

def generate_model_list():
    return ', '.join(detector_config_map.keys())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='process_testset.py',
                    description='Takes all videos from folder (src_dir), runs preprocess, rearrange and finally inference')
    
    parser.add_argument('folder_name', help='Name of folder')
    parser.add_argument('model', help=f'Choose from options: {generate_model_list()}')
    parser.add_argument('--skip_processing', action='store_true', help='skip preprocessing and rearrange steps (if dataset is already preprocessed)')

    args = parser.parse_args()

    folder_name = args.folder_name
    model = args.model
    skip_processing = args.skip_processing

    if model not in detector_config_map:
        print(f"No model found! Pick from {generate_model_list()}")

    src_dir = os.path.join(os.getcwd(), f'../deepfakebench-frontend/uploads/{folder_name}') #can change!!
    dest_dir = os.path.join(os.getcwd(), '../datasets/TestSet/fake') # must be same directory structure as deepfakebench

    from preprocessing.preprocess import main as preprocess
    from preprocessing.rearrange import main as rearrange
    from training.test import main as tests

    detector_config = detector_config_map[model]
    detector_weights = detector_weights_map[model]

    if not skip_processing:
        shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

        os.chdir(os.path.join(os.getcwd(), './preprocessing'))
        preprocess()
        print("Stage 1: Frames and Landmarks Generated!")
        sys.stdout.flush()
        
        rearrange()
        print("Stage 2: JSON File Generated!")
        sys.stdout.flush()
        
        os.chdir(os.path.join(os.getcwd(), '../'))
    else:
        print("Skipping Stages 1 and 2 of preprocessing and rearranging!")
    
    tests(["--detector_path", detector_config, "--test_dataset", "TestSet", "--weights_path", detector_weights])
    shutil.move(os.path.join(os.getcwd(), f'./results/{model}/TestSet_results.csv'), os.path.join(src_dir, f'frame_{model}.csv'))
    shutil.move(os.path.join(os.getcwd(), f'./results/{model}/TestSet_video_results.csv'), os.path.join(src_dir, f'video_{model}.csv'))
    print("Stage 3: Results Generated!")
    sys.stdout.flush()
