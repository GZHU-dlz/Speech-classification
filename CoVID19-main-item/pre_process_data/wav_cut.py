'''
切割音频的脚本
'''
import os
import wave
import numpy as np

audio_class = 'vowel_o'
cut_time_def = 5  # 以5秒截断

filename = '/private/Coswara-Data/positive_quality12_id_scp/positive_quality12_{}_path.scp'.format(audio_class)
try:
    with open(filename, 'r') as file:
        files = [line.strip() for line in file.readlines()]
except FileNotFoundError:
    print(f"Error: The file {filename} does not exist.")
    files = []  # 或者你可以退出程序，取决于你的需求

def cut_file():
    for file_path in files:
        try:
            print(f"Processing file: {file_path}")
            with wave.open(file_path, 'rb') as f:
                params = f.getparams()
                nchannels, sampwidth, framerate, nframes = params[:4]
                cut_frame_num = int(framerate * cut_time_def)
                # 读取整个音频文件
                str_data = f.readframes(nframes)
                wave_data = np.frombuffer(str_data, dtype=np.int16)  # 注意这里使用 frombuffer 而不是 fromstring
                # 假设是单声道，如果是多声道需要额外处理
                # 如果 wave_data 是二维的（多声道），则需要按声道处理
                if wave_data.ndim == 2:
                    wave_data = wave_data.flatten()  # 将多声道数据展平为单声道处理（简单示例，可能不适用所有情况）
                # 分割音频
                step_num = cut_frame_num
                haha = 0
                output_files = []
                while step_num * (haha + 1) <= nframes:
                    start_frame = step_num * haha
                    end_frame = step_num * (haha + 1)
                    temp_data_temp = wave_data[start_frame:end_frame]
                    # 获取文件名和目录信息
                    fn = os.path.basename(file_path)
                    parts = file_path.split('/')
                    target_part = parts[-2]
                    output_dir = '/private/Coswara-Data/cut_5s_data/positive_data/{}'.format(audio_class)
                    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
                    output_filename = os.path.join(output_dir, target_part + "-" + fn.split(".")[0] + "-" + str(haha + 1) + ".wav")
                    print(f"Writing to file: {output_filename}")
                    with wave.open(output_filename, 'wb') as output_f:
                        output_f.setnchannels(nchannels)
                        output_f.setsampwidth(sampwidth)
                        output_f.setframerate(framerate)
                        output_f.writeframes(temp_data_temp.tobytes())
                    output_files.append(output_filename)
                    haha += 1
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist and will be skipped.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

if __name__ == '__main__':
    cut_file()
    print("Run Over!")