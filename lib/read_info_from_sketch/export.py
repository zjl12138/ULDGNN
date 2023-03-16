import json
from logging import root
import os
from asyncio.subprocess import create_subprocess_shell, Process, PIPE
import glob

rootDir = "/Users/clq2021/Desktop/ljz-workspace/test_sketch_file"
sketch_name = "Alibaba_Sketch_132"
artboard_id="层级对比"
info = json.load(open(os.path.join(rootDir,sketch_name,artboard_id+".json"),"r"))
layers = info['layers']
id_list = [layer['id']  for layer in layers if layer['class']!='Text']

export_dir = os.path.join(rootDir, sketch_name, artboard_id)
os.makedirs(export_dir,exist_ok=True)

def bool_to_cmd(value: bool):
    return "YES" if value else "NO"

def cmd(
        document: str,
        output:str,
        formats= None,
        items=None,
        item= None,
        scales=  None,
        save_for_web= None,
        overwriting=  None,
        trimmed= None,
        background= None,
        group_contents_only=  None,
        use_id_for_name= None,
        suffixing= None,
        ) -> Process:
            cmd = f"export layers \"{document}\""
           
            args = [
                ("output", output, lambda x: f"\"{x}\""),
                ("formats", formats, lambda x: ",".join([f.value for f in x])),
                ("items", items, lambda x: ",".join(x)),
                ("item", item, lambda x: x),
                ("scales", scales, lambda x: ",".join([str(s) for s in x])),
                ("save-for-web", save_for_web, bool_to_cmd),
                ("overwriting", overwriting, bool_to_cmd),
                ("trimmed", trimmed, bool_to_cmd),
                ("background", background, str),
                ("group-contents-only", group_contents_only, bool_to_cmd),
                ("use-id-for-name", use_id_for_name, bool_to_cmd),
                ("suffixing", suffixing, bool_to_cmd),
            ]
            for arg in args:
                if arg[1] is not None:
                    cmd += f" --{arg[0]}={arg[2](arg[1])}"
            return cmd

sketch_file = os.path.join(rootDir, sketch_name+".sketch")

command = cmd(sketch_file,
                output=export_dir,
                items=id_list)
os.system(f"{'/Applications/Sketch.app/Contents/MacOS/sketchtool'} {command}")
os.system(f"{'/Applications/Sketch.app/Contents/MacOS/sketchtool'} export artboards {sketch_file} --output={os.path.join(rootDir,sketch_name)} --scales=4")
print(f"{'/Applications/Sketch.app/Contents/MacOS/sketchtool'} export artboards {sketch_file} --output={os.path.join(rootDir,sketch_name)}")
print(sketch_file)
print(command)

print(len(id_list))
img_list = glob.glob(export_dir+"/*.png")
print(len(img_list))
print("EBBC4F3F-89E5-4AC6-B24F-C6E45ABB778F" in id_list)