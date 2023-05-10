import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


path = "crewcamp/samsstag/"
#path = "../../../Data/Insta_CHI/Gyro_Test/"
name = "android.sensor.accelerometer.csv"

def plot_data(min=0, max=67000):
    plt.plot(sum3D_1)
    plt.plot(sum3D_2)
    plt.ylabel('gyrometer data')
    plt.xlabel('sample')
    plt.xlim(min, max)
    plt.show()

def create_video(image_dir, start_offset=0, endframe=None):
    out_path = image_dir[:-len(image_dir.replace("\\","/").split('/')[-2])-1]
    video = cv2.VideoWriter(out_path+'new_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(2160/2), int(3840/2)))  #mp4 codec: cv2.VideoWriter_fourcc(*'h265')
    files = os.listdir(image_dir)

    num_frames_plot = 30 * 5
    size_x = 800
    size_y = 200 / 4
    if endframe is None:
        endframe = len(files)
    for j, image_name in enumerate(files):
        if image_name.endswith('.jpg') and j < endframe:
            image = cv2.imread(image_dir + image_name)
            #img_new = cv2.resize(image, (int(2160/2), int(3840/2)))
            for k in range(num_frames_plot):
                y1_1 = 0; y1_2 = 0; y2_1 = 0; y2_2 = 0
                index1 = int(j + start_offset - num_frames_plot + k)
                index2 = int(j + start_offset - num_frames_plot + k - 1)
                if j > num_frames_plot - k and index1 < sum3D_1.shape[0]:
                    y1_1 = sum3D_1[index1] * size_y
                    y1_2 = sum3D_1[index2] * size_y
                    y2_1 = sum3D_2[index1] * size_y
                    y2_2 = sum3D_2[index2] * size_y
                cv2.line(image, (int(100+k*(size_x/num_frames_plot)), int(1800-y1_2)), (int(100+(k+1)*(size_x/num_frames_plot)), int(1800-y1_1)), (70,180,240), 5)
                cv2.line(image, (int(100+k*(size_x/num_frames_plot)), int(1800-y2_2)), (int(100+(k+1)*(size_x/num_frames_plot)), int(1800-y2_1)), (250,120,120), 5)
            video.write(image)
    video.release()

def data_to_file(filepath, cut_intervals=None, use_frames=False, fps=30, a_little_bit_more=False):
    if cut_intervals is None:
        np.savetxt(filepath + "gyro_data"+"_("+str(steps_per_frame)+")_all.txt", np.swapaxes((sum3D_1, sum3D_2),0,1))
        #np.savetxt(filepath + "angular_velocity.txt", sum3D_2)
    else:
        for j in range(len(cut_intervals)):
            if use_frames:
                idx1 = int(cut_intervals[j, 0])
                idx2 = int(cut_intervals[j, 1])
            else:
                idx1 = int(cut_intervals[j, 0]*fps)
                idx2 = int(cut_intervals[j, 1]*fps)
            if a_little_bit_more:
                idx2 += 10*fps
            print("frame:", idx1, idx2)
            np.savetxt(filepath + "gyro_data"+"_("+str(steps_per_frame)+")_"+str(j)+".txt", np.swapaxes((sum3D_1[idx1:idx2], sum3D_2[idx1:idx2]), 0, 1))


data = np.fromfile(path + name)
#data = data.reshape((int(data.size/8), 8))

steps_per_frame = 1/30 / ((data[-1,1] - data[0,1]) / data.shape[0])

offset = 6 * 30 * 11.07    # equals 6sec offset


sum3D_1 = np.zeros(int((data.shape[0] - offset) / steps_per_frame))
sum3D_2 = np.zeros(int((data.shape[0] - offset) / steps_per_frame))
for i in range(sum3D_1.shape[0]):
    idx = int(offset+i*steps_per_frame)
    sum3D_1[i] = np.sqrt(np.average(data[idx:int(idx+steps_per_frame),2])**2 + np.average(data[idx:int(idx+steps_per_frame),3])**2 + np.average(data[idx:int(idx+steps_per_frame),4])**2)
    sum3D_2[i] = abs(np.sqrt(np.average(data[idx:int(idx+steps_per_frame),5])**2 + np.average(data[idx:int(idx+steps_per_frame),6])**2 + np.average(data[idx:int(idx+steps_per_frame),7])**2)-1)

#sum3D_1 = np.zeros(data.shape[0])
#sum3D_2 = np.zeros(data.shape[0])
#for i, sample in enumerate(data):
#    sum3D_1[i] = np.sqrt(sample[2]**2 + sample[3]**2 + sample[4]**2)
#    sum3D_2[i] = np.sqrt(sample[2+3]**2 + sample[3+3]**2 + sample[4+3]**2)


#plot_data(max=len(sum3D_1))

#create_video(path + 'extr_fall6/', 445*30)

#data_to_file(path, cut_intervals=np.array([(95.0, 118), (153, 180), (190, 205), (250, 267), (320, 350), (385, 420), (445, 465), (534, 547)]))
#data_to_file(path, cut_intervals=np.array([(2749, 3499), (4749, 5499), (5749, 6249), (7499, 8129), (9629, 10630), (11629, 12629), (13379, 14129), (15879, 16629)]), use_frames=True, a_little_bit_more=True)

# 1: 2749 / 0:01:31:633 - 3499 / 0:01:56:633
# 2: Frame 4749 (0:02:38.300) [K] - Frame 5499 (0:03:03.300) [K]
# 3: Frame 5749 (0:03:11.633) [K] - Frame 6249 (0:03:28.300) [K]
# 4: Frame 7499 (0:04:09.967) [K] - Frame 8129 (0:04:30.967) [K]
# 5: Frame 9629 (0:05:20.967) [K] - Frame 10630 (0:05:54.333) [K]
# 6: Frame 11629 (0:06:27.633) [K] - Frame 12629 (0:07:00.967) [K]
# 7: Frame 13379 (0:07:25.967) [K] - Frame 14129 (0:07:50.967) [K]
# 8: Frame 15879 (0:08:49.300) [K] - Frame 16629 (0:09:14.300) [K]
