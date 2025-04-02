
import os
import pandas as pd

audio_dir = '/root/autodl-tmp/DCASE2023_data/dcase22_reassembled_test/TAU-urban-acoustic-scenes-2022-mobile-development/audio'
csv_file = 'audio.csv'

data = []
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            scene_label = file.split('-')[0]
            identifier = '-'.join(file.split('-')[1:-1])
            source_label = file.split('-')[-1][:-4]
            data.append([filename, scene_label, identifier, source_label])

df = pd.DataFrame(data, columns=['filename', 'scene_label', 'identifier', 'source_label'])
df.to_csv(csv_file, sep=' ', index=False)