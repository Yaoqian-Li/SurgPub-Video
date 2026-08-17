import os
import cv2

def convert_video_to_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 初始化帧计数器
    frame_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，退出循环
        if not ret:
            break
        
        # 每隔一秒保存一帧
        if frame_count % int(fps) == 0:
            # 生成保存路径
            frame_path = os.path.join(output_folder, f"frame_{frame_count // int(fps):05d}.png")
            # 保存帧为PNG图像
            cv2.imwrite(frame_path, frame)
        
        # 增加帧计数器
        frame_count += 1
    
    # 释放视频对象
    cap.release()
    print(f"视频 {video_path} 转换完成，保存到 {output_folder}")

def process_directory(directory):
    # 遍历目录下的所有文件夹
    i=0
    for root, dirs, files in os.walk(directory):
       
        for dir_name in dirs:
            i=i+1
            print(f'process file {i}/{len(dirs)}')
            # 获取文件夹路径
            folder_path = os.path.join(root, dir_name)
            
            # 查找文件夹中的mp4文件
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".mp4"):
                    # 获取视频文件路径
                    video_path = os.path.join(folder_path, file_name)
                    if 'frames' not in os.listdir(folder_path):
                    # 创建保存帧的文件夹
                        output_folder = os.path.join(folder_path, "frames")
                        os.makedirs(output_folder, exist_ok=True)
                        
                        # 转换视频为帧图像
                        convert_video_to_frames(video_path, output_folder)
                        txt_file = os.path.join(folder_path, "done.txt")
                        with open(txt_file, "w", encoding="utf-8"):
                            pass


# 指定要处理的目录
# directory = "/research/d1/gds/yqli/SurV/collect/cardiac_video"
# #740
# 处理目录
# process_directory(directory)
directory = "/research/d1/gds/yqli/SurV/collect/else_video"
process_directory(directory)
directory = "/research/d1/gds/yqli/SurV/collect/else2_video"
process_directory(directory)
directory = "/research/d1/gds/yqli/SurV/collect/lung_video"
process_directory(directory)
