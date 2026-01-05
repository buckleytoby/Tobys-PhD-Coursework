import pygame

import nodes

from pathlib import Path

def load_files_from_folder(folder_path, extension="*"):
    """
    Loads file paths from a specific subfolder.
    :param folder_path: String or Path to the subfolder
    :param extension: Filter by extension (e.g., ".png" or ".txt"). Default is all files.
    """
    path = Path(folder_path)
    
    # Use .glob() to find files. 
    # "*" means all files, "*.png" means only PNGs.
    files = list(path.glob(f"*{extension}"))
    
    return files

class Assets:
    def __init__(self) -> None:

        # my members
        self.assets = {}

        # load
        self.load_all_pngs()
        
    def load_all_pngs(self):
        folder = "assets"

        filenames = load_files_from_folder(folder, "png")

        for filename in filenames:
            asset = pygame.image.load(filename).convert_alpha()

            self.assets[filename.stem] = asset

    def __getitem__(self, key):
        return self.assets[key]