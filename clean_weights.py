import os
import glob

# Set the directory path where the files are located
dir_path = "output_hyak"

# Use glob to get a list of files that end with "00007999.pth"
remove_postfix = ["0000999.pth", "0001999.pth", "0002999.pth", "0003999.pth", 
                  "0004999.pth", "0005999.pth", "0006999.pth", "0007999.pth",
                  "0008999.pth", "0009999.pth", "0010999.pth", "0011999.pth",
                  "0012999.pth", "0013999.pth",                "0015999.pth",]
for postfix in remove_postfix:
    file_list = glob.glob(os.path.join(dir_path, "*", f"*{postfix}"))

    # Loop through the list and remove the files
    for file_path in file_list:
        print(file_path)
        os.remove(file_path)
