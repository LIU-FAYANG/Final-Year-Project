import os
import shutil

previous = None 
teacher = "test"
for f in os.listdir(teacher):
    source_path = os.path.join(teacher, f)
    destination_path = previous 
    if destination_path:
        shutil.rmtree(destination_path)

    # Overwrite files in distill_teacher2 with files from distill_teacher1
    if previous:
        shutil.copytree(source_path, destination_path)
        print(source_path + " copy to " + destination_path)
    previous = os.path.join(teacher, f)

print("over")