import os
from tqdm import tqdm
import argparse
import numpy as np
import cv2

from MTCNN_Portable.mtcnn import MTCNN

def mkdir(name):
    '''创建文件夹'''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def del_dirs(path):
    '''删除文件夹'''
    try:
        shutil.rmtree(path)
    except:
        pass

def main(args):
    input_fold = args.input
    if input_fold[-1]!= '/':
        output_fold = input_fold+'_list'
    else:
        output_fold = input_fold[:-1]+'_list'
    del_dirs(output_fold)
    mkdir(output_fold)
    # MTCNN Detector
    detector = MTCNN()

    img_list = tqdm(os.listdir(input_fold))
    for img_path in img_list:
        img_list.set_description("Detect landmarks")
        save_landmark(os.path.join(input_fold, img_path), output_fold, detector)

# load input images and corresponding 5 landmarks
def save_landmark(img_path, output_fold, detector):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #Detect 5 key point
    try:
        face = detector.detect_faces(img)[0]
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        nose = face["keypoints"]["nose"]
        mouth_left = face["keypoints"]["mouth_left"]
        mouth_right = face["keypoints"]["mouth_right"]
        lm = np.array([[left_eye[0], left_eye[1]],
                    [right_eye[0], right_eye[1]],
                    [nose[0], nose[1]],
                    [mouth_left[0], mouth_left[1]],
                    [mouth_right[0], mouth_right[1]]])
        save_name = img_path.split('/')[-1].replace(".jpg",".txt")
        np.savetxt(os.path.join(output_fold,save_name), lm)
    except:
        print(img_path)
    return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, default='./test_images/n000002/',help='Path of input image')
	arguments = parser.parse_args()
	main(arguments)