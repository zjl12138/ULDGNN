import os
import sys
rootdir="/media/sda1/ljz-workspace/dataset/ui_dataset/"
for i in range(20):
    os.system(f"cp {rootdir}/{i}.png /media/sda1/ljz-workspace/dataset/graph_dataset/{i}/{i}.png")
    os.system(f"cp {rootdir}/{i}-assets.png /media/sda1/ljz-workspace/dataset/graph_dataset/{i}/{i}-assets.png")
    
    
