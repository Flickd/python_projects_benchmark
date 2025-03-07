import glob
import os
from pathlib import Path

import pandas as pd


class Utility:
    @staticmethod
    def ensure_dir(root_dir, directory):
        directory_path = Path(f"{root_dir}/{directory}").absolute().resolve()
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return directory_path

    @staticmethod
    def get_translation(txt, en_txt):
        de_en = None
        for script in en_txt:
            s_script = script.strip("\n")
            txt_seg = txt.split(";")
            if txt_seg[0] in script:
                de_en = txt.strip("\n") + ";" + s_script.split(";")[1].strip(" ") + "\n"
                break
        if de_en is None:
            print("This must not happened")
        return de_en

    @staticmethod
    def load_embeddings():
        path = r"yourPathToEmbeddings"
        embeddings = pd.read_pickle(path)
        for embed in embeddings:
            key, text_embed, answer_embed, bridge_embed, vid_embed, en_de_txt = embed
            print(key)
        print("End")
