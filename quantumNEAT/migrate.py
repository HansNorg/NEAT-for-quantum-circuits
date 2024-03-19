import os
from pathlib import Path

def main():
    folders = ["results", "logs", "figures"]
    for folder in folders:
        # files = Path(".\\data_change_try\\"+folder+"\\").glob("**/*")
        files = Path(".\\"+folder+"\\").glob("*.png")
        for file in files:
            if file.is_dir():
                continue
            predir=""
            # if file.parent.name != folder:
            #     predir = file.parent.name
            print(file.name)
            filedir, filename = file.name.rsplit("_", maxsplit=1)
            filedir = predir+"\\"+filedir
            filedir = filedir.replace("_", "\\")
            os.makedirs("new_results\\"+filedir, exist_ok=True)
            os.rename(file, "new_results\\"+filedir+"\\"+filename)

if __name__ == "__main__":
    # Idea for if time to restructure result files. For now just a little too much manual labour
    # main()
    print("disabled to prevent accidental migration")