# Load modules
import tqdm
import czifile
import tifffile
import sys
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
from IPython.display import Image as IPImage, display
from matplotlib.animation import FuncAnimation
from skimage.io import imread
from itertools import combinations
import cv2

class ParallelLines:

    def __init__(self):
        pass

    def __str__():
        pass

    def __repr__(self):
        pass
    
    def visualize_annotated_cut(self, save_figure: bool = False):
        """
        Shows the cut on top of the frame according to the annotated metadata related to
        the experiment. If save_figure is set to True, the image will be saved to the
        specified output path.
        """
        
        # Get points for experiment
        cell_id_to_keep = self.experiment

        # Load the points data
        points_file_loc = os.path.join(self.data_dir, "ablation-lineage/", f"{cell_id_to_keep}.lineage")
                
        # Check that file exists
        if not os.path.isfile(points_file_loc):
            raise FileNotFoundError(f"The file {points_file_loc} does not exist.")
        # If file exists, extract cut coordinates
        else:
            df = pd.read_csv(points_file_loc, sep='\t')
            # Extract cuts
            df_cuts = df[df.iloc[:, 0].str.contains("cut", case=False, na=False)]

        # Draw the cut
        # Extract the points for the cuts (first two rows in this case)
        x1, y1 = df_cuts.iloc[0]['x'], df.iloc[0]['y']
        x2, y2 = df_cuts.iloc[1]['x'], df.iloc[1]['y']

        # Get the image axes from visualize_frame (without showing)
        ax = self.visualize_frame(show=False)

        # Draw the annotated cut
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)

        # Save or show
        if save_figure:
            path = os.path.join(self.output_path, f"annotated_cut_{self.experiment}.jpg")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Figure was savaed to {path}.")
        else:
            ax.set_title('Annotated Cut')
            ax.axis('off')
            plt.show()
    
    def find_lines(self):
        pass

    def draw_lines(self):
        pass
    

if __name__ == '__main__':

    pass