"""
04/24/2025

Peixoto Lab script used to analyze Suite2P-based data with txt containg ROI label of E2-Tom+ cells and store as hdf5 files

Version used to analyze data in Shih et al 2025 paper "Early Postnatal Dysfunction of ACC PV Interneurons in Shank3B-/- mice"

"""
#%% import packages
import os
import tkinter as tk
from tkinter import Label, Entry, Button, filedialog
from pathlib import Path, PurePath
import plotly.tools as tls
import numpy as np
import pandas as pd
from oasis.functions import deconvolve
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

#%% Choose directory containing Suite2P data
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Modify Global Variables")
        
        """Set global variables, 
        rec: number of recording used for saving file
        SR: sampling rate (Hz)
        movement_thr: threshold for detecting movement artifact
        movement_win: window for detecting movement artifact
        e2source: file type that contain Suite2P ROI of E2+ cells
        hemisphere: side of brain
        head_orientation: orientation of mouse head in 2P imaging
        ml_dist: mediolateral distance from central sinus
        cellpose: set 1 if you have a secondary suite2P folder with ROI identified by anatomical method (CellPose)
        """
        self.rec = 0
        self.SR = 15
        self.movement_thr = 1.5
        self.movement_win = 45
        self.e2source = 'txt'
        self.hemisphere = 'L'
        self.head_orientation = 'L'
        self.ml_dist = 500
        self.cellpose = 1
        
        self.path = None
        
        # Launch the directory selection dialog
        self.select_directory()

    def select_directory(self):
        self.withdraw()
        path = filedialog.askdirectory(title="Select the parent folder (e.g., TSeries-WT-001)")
        if path:
            self.path = path
            self.deiconify()
            self.init_ui()
            # Automatically load data
            self.load_data()
        else:
            print("No directory was selected. Closing...")
            self.withdraw()
            self.destroy()

    def load_data(self):
        if self.path:
            # Load suite2p data
            suite2p_path = os.path.join(self.path, 'suite2p')
            self.p2 = TwoP(suite2p_path, self.e2source)
            print(f"Selected suite2p folder: {suite2p_path}")

            # Load suite2p_CP data if cellpose is set to 1
            if self.cellpose == 1:
                suite2p_cp_path = os.path.join(self.path, 'suite2p_CP')
                self.p2_cp = TwoP(suite2p_cp_path, self.e2source)
                print(f"Selected suite2p_CP folder: {suite2p_cp_path}")
            else:
                self.p2_cp = None

            # You can add additional data processing here

    def save_changes(self):
        global rec, SR, movement_thr, movement_win, e2source, hemisphere, head_orientation, ml_dist, cellpose

        try:
            rec = int(self.rec_entry.get())
            SR = int(self.sr_entry.get())
            movement_thr = float(self.movement_thr_entry.get())
            movement_win = int(self.movement_win_entry.get())
            e2source = self.e2source_entry.get().strip().lower()
            hemisphere = self.hemisphere_entry.get().strip().upper()
            head_orientation = self.head_orientation_entry.get().strip().upper()
            ml_dist = int(self.ml_dist_entry.get())
            cellpose = int(self.cellpose_entry.get())

            if e2source not in ["txt", "excel", ""]:
                raise ValueError("E2 'txt', 'excel', or empty")

            print(f"Updated rec: {rec}, SR: {SR}, movement_thr: {movement_thr}, movement_win: {movement_win}, e2source: {e2source}, hemisphere: {hemisphere}, head_orientation: {head_orientation}, ml_dist: {ml_dist}, cellpose: {cellpose}")
            self.withdraw()
            self.quit()  # Quit the mainloop to proceed

        except ValueError as e:
            print("Error updating variables:", e)

    def init_ui(self):
        Label(self, text="Recording Folder Index (rec):").grid(row=0, column=0, sticky="w")
        self.rec_entry = Entry(self)
        self.rec_entry.insert(0, str(self.rec))
        self.rec_entry.grid(row=0, column=1)

        Label(self, text="Sampling Rate (SR):").grid(row=1, column=0, sticky="w")
        self.sr_entry = Entry(self)
        self.sr_entry.insert(0, str(self.SR))
        self.sr_entry.grid(row=1, column=1)

        Label(self, text="Movement Threshold (movement_thr):").grid(row=2, column=0, sticky="w")
        self.movement_thr_entry = Entry(self)
        self.movement_thr_entry.insert(0, str(self.movement_thr))
        self.movement_thr_entry.grid(row=2, column=1)

        Label(self, text="Movement window (movement_win):").grid(row=3, column=0, sticky="w")
        self.movement_win_entry = Entry(self)
        self.movement_win_entry.insert(0, str(self.movement_win))
        self.movement_win_entry.grid(row=3, column=1)
        
        Label(self, text="E2 Source (txt/excel/empty):").grid(row=4, column=0, sticky="w")
        self.e2source_entry = Entry(self)
        self.e2source_entry.insert(0, self.e2source)
        self.e2source_entry.grid(row=4, column=1)
        
        Label(self, text="Hemisphere (L/R):").grid(row=5, column=0, sticky="w")
        self.hemisphere_entry = Entry(self)
        self.hemisphere_entry.insert(0, self.hemisphere)
        self.hemisphere_entry.grid(row=5, column=1)
        
        Label(self, text="Head orientation (L/UL/LL):").grid(row=6, column=0, sticky="w") # L:left / UL:upper left / LL:lower left
        self.head_orientation_entry = Entry(self)
        self.head_orientation_entry.insert(0, self.head_orientation)
        self.head_orientation_entry.grid(row=6, column=1)
        
        Label(self, text="Mediolateral distance (ml_dist):").grid(row=7, column=0, sticky="w")
        self.ml_dist_entry = Entry(self)
        self.ml_dist_entry.insert(0, self.ml_dist)
        self.ml_dist_entry.grid(row=7, column=1)
        
        Label(self, text="CellPose (0 or 1):").grid(row=8, column=0, sticky="w")
        self.cellpose_entry = Entry(self)
        self.cellpose_entry.insert(0, self.cellpose)
        self.cellpose_entry.grid(row=8, column=1)
        
        Label(self, text="").grid(row=9, column=0, sticky="w")

        Button(self, text="Save Changes", command=self.save_changes).grid(row=10, column=0, columnspan=2)


class TwoP:
    
    '''searches and loads npy, txt and excel files in source directory'''
    
    def __init__(self, path, e2source):
        self.path = path
        self.name = PurePath(path).stem
        self.e2source = e2source  # Use the e2source passed from App
        self.load_npy()
        
        if self.e2source == "txt":
            self.load_txt()
        elif self.e2source == "excel":
            self.load_excel()
        elif self.e2source == "":
            self.handle_no_e2source()

    def load_txt(self): # txt file should only contain numbers as ROI of E2-Tom+ cells
        try:
            txt_files = list(Path(self.path).rglob("E2*.txt"))
            if not txt_files:
                raise FileNotFoundError("No .txt files found in the specified directory.")
            txt_path = txt_files[0]
            self.txt = np.asarray(np.loadtxt(txt_path), dtype=np.int64)
        except FileNotFoundError as e:
            print(f"Error loading .txt file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading .txt file: {e}")

    def load_npy(self):
        try:
            npy_files = list(Path(self.path).rglob("*.npy"))
            if not npy_files:
                raise FileNotFoundError("No .npy files found in the specified directory.")
            for i in npy_files:
                if i.stem in {"F", "Fneu", "iscell", "spks", "ops", "stat"}:
                    x = np.load(i, allow_pickle=True)
                    setattr(self, i.stem, x)
        except FileNotFoundError as e:
            print(f"Error loading .npy file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading .npy files: {e}")

    def load_excel(self):
        try:
            excel_files = [file for file in Path(self.path).rglob("E2*.xlsx")]
            if not excel_files:
                raise FileNotFoundError("No Excel files starting with 'E2' found in the specified directory.")
            excel_path = excel_files[0]  # Assuming you want to load the first matching file
            self.excel_data = pd.read_excel(excel_path)
            print(f"Excel data loaded from {excel_path}")
        except FileNotFoundError as e:
            print(f"Error loading Excel file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading Excel file: {e}")

    def handle_no_e2source(self):
        # Handle case where no is_e2 data is to be loaded
        print("No source for is_e2 specified.")

class PlotWindow:
    
    '''This is used to plot the movement artifact dxdy function. 
    OK leaves data as is, Crop removes time segments with mov artifacts > thr'''
    
    def __init__(self, master, F_subt, F, Fneu, movement_ranges):
        self.master = master
        self.F_subt = F_subt
        self.F = F 
        self.Fneu = Fneu
        self.movement_ranges = movement_ranges  # Store as an instance attribute
        self.master.title("Movement Artifacts")
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.plot_movement_artifacts()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ok_button = tk.Button(master=self.master, text="OK", font=('Helvetica', 16), command=self.close_window)
        self.ok_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.crop_button = tk.Button(master=self.master, text="Crop", font=('Helvetica', 16), command=self.crop)
        self.crop_button.pack(side=tk.LEFT, padx=5, pady=5)    


    def crop(self):
        global F_corrected, F_crop, Fneu_crop
        F_corrected, F_crop, Fneu_crop = crop_and_filter(self.F_subt, self.F, self.Fneu, self.movement_ranges)
        print("Movement periods cropped.")
        self.master.withdraw()  
        self.master.quit()      
        self.master.destroy()
        
    def close_window(self):
        self.master.withdraw()  
        self.master.quit()      
        self.master.destroy()          
    
    def plot_movement_artifacts(self):
        ax = self.fig.add_subplot(111)
        ax.plot(np.arange(len(movement)), movement, color='blue', label='d_xy offset')
    
        # Access movement_ranges from the instance attribute
        if self.movement_ranges:
            for sublist in self.movement_ranges:
                ax.plot(np.arange(len(movement))[list(sublist)], movement[list(sublist)], color='gray', linewidth=2)
    
        ax.scatter(np.arange(len(movement))[movement_frames], movement[movement_frames], color='red', zorder=5, label=f'Above Thr= {movement_thr}')
        ax.set_xlabel('Frames')
        ax.set_ylabel('dxdy offset')
        ax.set_title('Movement artifacts')
        ax.legend()


def viewer_dataset(data=None, mode='line', label="Data", x=None, y=None): 
    
    '''aux function to prepare datasets to be plotted in the functionviewer. It expects dictionaries in which each key is one function - 3 functions max'''
    
    return {'data': data, 'mode': mode, 'label': label, 'x': x, 'y': y} # for line plots add data, mode and label. For scatter plots (peaks) add mode, label, x and y (list of arrays)


class FunctionViewer:
    
    '''Plots and scrolls ROI data. It can plot 3 functions (line or scatter) per ROI and display parameter values.
    You can view ROI and flag them. The function returns a list of flagged ROIs (index)''' 
    
    
    def __init__(self, master, datasets, constants_df=None):
        self.master = master
        self.datasets = datasets  # This should be a list of datasets; data1 is mandatory for operations
        if constants_df is not None:
            self.constants = constants_df.to_dict(orient='records')  # Convert DataFrame to list of dicts
        else:
            self.constants = []        
        #self.constants = constants if constants is not None else []  # List of dictionaries with constants to display
        self.index = 0
        self.flagged = []
        
        self.fig = Figure(figsize=(16, 8), dpi=100)
        self.fig.subplots_adjust(left=0.05, right=0.83, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()

        self.nav_frame = tk.Frame(self.master)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_navigation_buttons()
        self.update_plot()

    def create_navigation_buttons(self):        
        self.prev_button = tk.Button(self.nav_frame, text="Previous", font=('Helvetica', 16), command=self.previous, padx=12, pady=6)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.flag_button = tk.Button(self.nav_frame, text="Flag", font=('Helvetica', 16), command=self.flag, padx=12, pady=6)
        self.flag_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(self.nav_frame, text="Next", font=('Helvetica', 16), command=self.next, padx=12, pady=6)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.quit_button = tk.Button(self.nav_frame, text="Quit", font=('Helvetica', 16), command=self.quit, padx=12, pady=6)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.master.bind('<Left>', self.previous)
        self.master.bind('<Right>', self.next)
        self.master.bind('<Up>', self.flag)
        self.master.bind('<Down>', self.quit)          
   
    def update_plot(self):
        if hasattr(self, 'ax'):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
    
        self.fig.clear()
        ax = self.fig.add_subplot(111)
    
        colors = ['green', 'red', 'blue']  # colors of data1, 2 and 3

        for i, dataset in enumerate(self.datasets):
            color = colors[i] if i < len(colors) else 'black'

            if dataset and dataset.get('data') is not None and self.index < len(dataset.get('data')):
                if dataset.get('mode') == 'line':
                    data = dataset['data'][self.index]
                if isinstance(data, pd.Series):  # Check if data is a pandas Series and convert to numpy
                    data = data.to_numpy()
                x_values = np.arange(len(data))  # Generate x-values as integers
                ax.plot(x_values, data, label=dataset.get('label'), color=color)
            
            elif dataset.get('mode') == 'scatter':
                # Directly use 'x' and 'y' from the dataset for scatter plots
                if 'x' in dataset and 'y' in dataset and self.index < len(dataset['x']) and self.index < len(dataset['y']):
                    x_values = dataset['x'][self.index]
                    y_values = dataset['y'][self.index]
                    if isinstance(x_values, pd.Series):  # Convert pandas Series to numpy if necessary
                        x_values = x_values.to_numpy()
                    if isinstance(y_values, pd.Series):
                        y_values = y_values.to_numpy()
                    ax.scatter(x_values, y_values, label=dataset.get('label'), color=color)

        if self.constants:
            text_str = "\n".join([
                f"{key}: {float(val):.2f}" if isinstance(val, (int, float)) else f"{key}: {val}"
                for key, val in self.constants[self.index].items()])
            ax.text(1.01, 1, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top')
        

        ax.legend(loc='best')
        ax.set_title(f"ROI {self.index}")
        self.canvas.draw()
        
        if 'xlim' in locals() and 'ylim' in locals():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            self.canvas.draw_idle()  # Update the canvas with the new limits
    
        self.check_flag_status()



    def next(self, event=None):
        # Navigate to next plot based on data1 length
        data1 = self.datasets[0]['data']
        self.index = min(self.index + 1, len(data1) - 1)
        self.update_plot()

    def previous(self, event=None):
        # Navigate to previous plot
        self.index = max(self.index - 1, 0)
        self.update_plot()

    def flag(self, event=None):
        # Flag or unflag the current index
        if self.index not in self.flagged:
            self.flagged.append(self.index)
        else:
            self.flagged.remove(self.index)
        self.check_flag_status()

    def check_flag_status(self):
        # Update flag button text based on flag status
        if self.index in self.flagged:
            self.flag_button.config(text="Unflag")
        else:
            self.flag_button.config(text="Flag")

    def quit(self, event=None):
        # Quit the application
        self.master.withdraw()
        self.master.quit()
        self.master.destroy()

def launch_function_viewer(datasets, constants_df=None):
    root = tk.Tk()
    viewer = FunctionViewer(root, datasets, constants_df)
    root.mainloop()
    return viewer.flagged


class MultiFunctionViewer:
    def __init__(self, master, datasets1, datasets2, constants_df=None):
        self.master = master
        self.datasets1 = datasets1
        self.datasets2 = datasets2
        self.constants = constants_df.to_dict(orient='records') if constants_df is not None else []
        self.index = 0
        self.flagged = []

        self.fig = Figure(figsize=(16, 7), dpi=100)
        self.fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()

        self.nav_frame = tk.Frame(self.master)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.create_navigation_buttons()
        self.update_plot()

    def create_navigation_buttons(self):
        self.prev_button = tk.Button(self.nav_frame, text="Previous", font=('Helvetica', 16), command=self.previous)
        self.prev_button.pack(side=tk.LEFT)

        self.flag_button = tk.Button(self.nav_frame, text="Flag", font=('Helvetica', 16), command=self.flag)
        self.flag_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.nav_frame, text="Next", font=('Helvetica', 16), command=self.next)
        self.next_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(self.nav_frame, text="Quit", font=('Helvetica', 16), command=self.quit)
        self.quit_button.pack(side=tk.RIGHT)

        self.master.bind('<Left>', self.previous)
        self.master.bind('<Right>', self.next)
        self.master.bind('<Up>', self.flag)
        self.master.bind('<Down>', self.quit)

    def update_plot(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        self.plot_data(ax1, self.datasets1, plot_constants=True)
        ax1.set_title(f"ROI {self.index}")
        self.plot_data(ax2, self.datasets2)

        self.canvas.draw()
        self.update_flag_status()  # Call to update the flag status when plot updates

    def plot_data(self, ax, datasets, plot_constants=False):
        colors = ['green', 'red', 'blue','magenta']
        for i, dataset in enumerate(datasets):
            if dataset['mode'] == 'line' and dataset['data'] is not None and self.index < len(dataset['data']):
                data = dataset['data'][self.index]
                if isinstance(data, pd.Series):
                    data = data.to_numpy()
                x_values = np.arange(len(data))
                ax.plot(x_values, data, label=dataset['label'], color=colors[i % len(colors)])
            elif dataset['mode'] == 'scatter' and 'x' in dataset and 'y' in dataset and self.index < len(dataset['x']) and self.index < len(dataset['y']):
                x_values = dataset['x'][self.index]
                y_values = dataset['y'][self.index]
                if isinstance(x_values, pd.Series):
                    x_values = x_values.to_numpy()
                if isinstance(y_values, list) or isinstance(y_values, np.ndarray):
                    y_values = np.array(y_values)
                ax.scatter(x_values, y_values, label=dataset['label'], color=colors[i % len(colors)])
            ax.legend()

        if plot_constants and self.constants:
            const_text = "\n".join(f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}" for key, val in self.constants[self.index].items())
            ax.text(1.01, 1, const_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def next(self, event=None):
        data_length = len(self.datasets1[0]['data']) if self.datasets1 else len(self.datasets2[0]['data'])
        self.index = min(self.index + 1, data_length - 1)
        self.update_plot()

    def previous(self, event=None):
        self.index = max(self.index - 1, 0)
        self.update_plot()

    def flag(self, event=None):
        if self.index not in self.flagged:
            self.flagged.append(self.index)
        else:
            self.flagged.remove(self.index)
        self.update_flag_status()

    def update_flag_status(self):
        # Update flag button text based on flag status
        if self.index in self.flagged:
            self.flag_button.config(text="Unflag")
        else:
            self.flag_button.config(text="Flag")

    def quit(self, event=None):
        self.master.withdraw()
        self.master.quit()
        self.master.destroy()

def launch_multifunction_viewer(datasets1, datasets2, constants_df=None):
    root = tk.Tk()
    viewer = MultiFunctionViewer(root, datasets1, datasets2, constants_df)
    root.mainloop()
    return viewer.flagged

if __name__ == "__main__":
    app = App()
    app.mainloop()

    '''Chooses data folder and loads files'''
    if app.path:
        # Instantiate TwoP with the selected directory
        p2 = app.p2
        p2_cp = app.p2_cp
        print(f"Selected folder: {app.path}")

        #Files loaded
        F = p2.F # Suite2P raw F
        Fneu = p2.Fneu # Suite2P F neuropil
        spks = p2.spks # Suite2P deconvolved spikes
        is_cell_Suite2P = p2.iscell # is_cell_list with all ROIs, and ROIs considered as cells based on confidence factor 
        ROIstat = p2.stat # stat.npy file
        ops = p2.ops.item() # ops.npy file
        
        if hasattr(p2, 'excel_data'): # if it has excel file load it
            excel_data = p2.excel_data
            print("Excel data loaded successfully.")
            #print(excel_data.head())
        else:
            print("Excel data was not loaded.")

        if cellpose == 1 and p2_cp:
            F_cp = p2_cp.F
            Fneu_cp = p2_cp.Fneu
            spks_cp = p2_cp.spks
            is_cell_Suite2P_cp = p2_cp.iscell
            ROIstat_cp = p2_cp.stat
            ops_cp = p2_cp.ops.item()

            
#%% process is_cell
'''initializes is_cell, and is_e2 lists. 
    If E2 file is txt, loads txt numbers as e2 indices
    If E2 file is excel, runs function to automatically detect overalapping tdTom regions detected in qupath'''

#rec_size = ops['nframes']  
  
is_cell = np.int64(np.hsplit(is_cell_Suite2P,2)[0]) #gets the 0/1 column of is_cell
idx_is_cell = np.where(is_cell[:,0]==1)[0] #gets ROI index where is_cell == 1

is_e2 = np.zeros((len(F), 1), dtype=int) # initializes is_e2
is_pyr = np.copy(is_cell) #initializes is_pyr as a copy of is_cell that where e2 will be deleted from 


e2_loaded = False               

if e2source == 'txt': #import e2 list from text, simple array where each number in txt file is ROi index
    e2_loaded = True
    tdT_positive = p2.txt  
    tdT_positive_idx = np.copy(tdT_positive)  # Use np.copy to avoid modifying the original list
    condition = (is_cell[tdT_positive_idx] == 0) # exclude tdT+ cell that is not a cell
    tdT_positive_idx = np.delete(tdT_positive_idx, np.where(condition)[0])
   
    if tdT_positive.size > 0: # if tdT_positive is not empty
        is_e2[tdT_positive_idx] = 1 
        is_pyr[tdT_positive_idx] = 0  # Exclude E2+ cells from is_cell list
        #idx_is_cell = np.where(is_cell[:,0]==1)[0] # updates idx_is_cell after exlusion of e2
        print(f"There are {tdT_positive.size} E2+ cells in functional ROI.")
    else:
        print("The tdT_positive txt list is empty. The is_e2 list remains all zeros.")
   
elif e2source == 'excel': # choose excel file with tdTom Qupath regions. name should start with E2
    
    e2_loaded = True
    def generate_approx_circle_points_based_on_area(x_centroid, y_centroid, area, resolution):
        '''this generates a circle out of centroid coordinates from qupath. Area of each circle is 1/2 area of qupath ROI'''
        
        np.random.seed(99)
        
        radius = np.sqrt((area / 1.5) / np.pi) # Calculate radius for a circle with half the area
        circumference = 2 * np.pi * radius # Estimate the circumference of the circle
        num_points = int(np.round(circumference)) # approximate num_points to match the circle's circumference
    
        points = []
        checked_pixels = set()
    
        x_min, x_max = max(0, int(x_centroid - radius)), min(resolution[0], int(x_centroid + radius + 1))
        y_min, y_max = max(0, int(y_centroid - radius)), min(resolution[1], int(y_centroid + radius + 1))
    
        while len(points) < num_points and len(checked_pixels) < (x_max - x_min) * (y_max - y_min):
            x, y = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)
            if (x, y) not in checked_pixels:
                checked_pixels.add((x, y))
                if (x - x_centroid) ** 2 + (y - y_centroid) ** 2 <= radius ** 2:
                    points.append((x, y))

        return np.array(points)

    def process_row(row): # Function to wrap generate_circle_points to work with rows
        x_centroid = row['Centroid X px']
        y_centroid = row['Centroid Y px']
        area = row['Nucleus: Area']
        
        return generate_approx_circle_points_based_on_area(x_centroid, y_centroid, area, resolution)

    resolution = (ops['Lx'], ops['Ly'])  
    x_centroid = excel_data['Centroid X px'] # qupath region centroid x
    y_centroid = excel_data['Centroid Y px'] # qupath region centroid y
    area = excel_data['Nucleus: Area'] # qupath region area  
    qupath_regions = excel_data.apply(process_row, axis=1).tolist() #runs the function and generates points list
    
    ROIxy = [] 
    
    for idx, d in enumerate(ROIstat):  
        xpix = d['xpix']  # Suite2P ROI x coordinates (all, not just centroid)
        ypix = d['ypix']  # Suite2P ROI y coordinates (all, not just centroid)
        # Build tuples from xpix and ypix and ROI index.
        roi_pixels_with_ref = [(x, y, idx) for x, y in zip(xpix, ypix)]  # Adding idx as a reference to keep track of which suite2P ROI corresponds to which pixel
        ROIxy.append(roi_pixels_with_ref)

    def find_ROI_matches(ROIxy, qupath_regions): # this goes through all xy ROIs 
        is_e2_img = [] # this is the is_e2 list that is generated with indices of is_cell 
        matched_points_list_indices = set()
        ROI_with_no_qupath = [] #this is the list of qupath regions in which no suite2P ROI was found nearby

        # Convert each array in points_list to a set of tuples for efficient comparison
        qupath_regions_sets = [set(map(tuple, points)) for points in qupath_regions]

        for roi_idx, roi_with_ref in enumerate(ROIxy):
            roi_set = set((x, y) for x, y, _ in roi_with_ref)  # Convert the ROI pixels to a set for efficient overlap check
            overlap_detected = False

            for points_idx, points_set in enumerate(qupath_regions_sets):
                # Calculate overlap
                overlap = roi_set.intersection(points_set)
                if len(overlap) / len(roi_set) >= 0.25:  # At least qupath_overlap overlap tdTom and Suite2P ROIs
                    is_e2_img.append(roi_idx)
                    matched_points_list_indices.add(points_idx)
                    overlap_detected = True
                    break  # Stop checking this ROI against other points_list arrays if overlap found

            if not overlap_detected:
                # ROI did not match any points_list array by at least qupath_overlap
                original_idx = roi_with_ref[0][2]  # this takes the third value of the ROIxy tuple that is the index of the ROI in Suite2p stat file
                ROI_with_no_qupath.append(original_idx)

        # Identifying points_list arrays with no matches at all
        all_indices = set(range(len(qupath_regions)))
        tdTom_with_no_ROI = list(all_indices - matched_points_list_indices)

        return is_e2_img, ROI_with_no_qupath, tdTom_with_no_ROI

    ROIxy_is_cell = [ROIxy[i] for i in idx_is_cell]
    # Find matches only once 
    is_e2_img, _ , tdTom_with_no_ROI = find_ROI_matches(ROIxy, qupath_regions)

    # Convert is_e2_img to a numpy array for further processing
    is_e2_img_np = np.array(is_e2_img)

    # Check if there are any ROIs identified as e2
    if is_e2_img_np.size > 0:
        is_e2[is_e2_img_np] = 1
        is_pyr[is_e2_img_np] = 0  # Exclude these ROIs from is_pyr
        # Update idx_is_cell based on the latest is_cell status
        print(f"Qupath identifies {len(is_e2_img)} E2+ cells. Note that some might not have valid ROIs")
        
        """plots the overlap of Suite2P ROIs with qupath reconstructed circle regions"""
        
        image_array = np.zeros((ops['Lx'], ops['Ly']), dtype=np.uint8)  # Initialize a 1024x1024 black image

        # Update the image array with points from points_list
        for points in qupath_regions:
            for x, y in points:
                image_array[y, x] = 1  # Set the points to white

        # Plot setup
        plt.figure(figsize=(20, 20))

        # Plot points from points_list in red
        for points in qupath_regions:
            if len(points) > 0:  # Check if the array of points is not empty
                plt.scatter(points[:, 0], points[:, 1], color='red', s=5)

        # Plotting all points from ROIxy_is_cell in white and larger
        for sublist in ROIxy_is_cell:
            x_coords, y_coords = zip(*[(x, y) for x, y, _ in sublist])  # This unpacks your sublist of (x, y) tuples into x and y coordinates
            plt.scatter(x_coords, y_coords, color='gray', s=1)  # Increased size for visibility

        # Adjust plot aesthetics to match image coordinates and set background
        plt.gca().invert_yaxis()  # Match the y-axis orientation to image coordinates
        plt.gca().set_facecolor('black')  # Black background to resemble the image array visualization
        plt.gca().set_aspect('equal', 'box')  # Ensure equal aspect ratio
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        plt.show()

    else:
        print("No ROIs in qupath excel file meet the overlap criteria. The is_e2 list remains all zeros.")

if not e2_loaded:
    
    print("No E2 txt or excel files in this recording. The is_e2 list remains all zeros.")
        
#Create dataframes with original index of e2 and non e2 so we can separate L/R based on stat.npy file

idx_pyr = np.where((is_pyr[:,0]==1) & (is_cell[:,0]==1))[0]
idx_e2 = np.where((is_e2[:,0]==1) & (is_cell[:,0]==1))[0]

print(f'Functional ROI: {len(idx_pyr)} non-E2 cells')
print(f'Functional ROI: {len(idx_e2)} E2 cells')


def classify_LR (ROIstat, idx_list, head_orientation):
    """
    Classify ROIs as Left or Right based on their Suite2P y-coordinate.
    
    Returns:
        LR_list (list): List of tuples with (medxy, label) for each ROI in idx_list.
        Note the med coordinates are YX and not XY
    """
    
    LR_list = []
    LR_labels = []
    for idx in idx_list:
        d = ROIstat[idx]
        medxy = d['med']  # Suite2P ROI median coordinates
        # Note: med is in y, x format; the first value of the tuple is the y-coordinate.
        # 2x view: 512x512 pixel = 550x550 um
        if head_orientation == 'UL':
            ml_dist_diff = ml_dist*512//550
            if hemisphere == 'L':
                label = "Left"
                medxy_adjusted = ((medxy[1]+medxy[0])/2**0.5, ml_dist_diff - (medxy[1]-medxy[0])/2**0.5)
            else:
                label = "Right"
                medxy_adjusted = ((medxy[1]+medxy[0])/2**0.5, ml_dist_diff + (medxy[1]-medxy[0])/2**0.5)
                
        elif head_orientation == 'LL':
            ml_dist_diff = ml_dist*512//550
            if hemisphere == 'L':
                label = "Left"
                medxy_adjusted = ((medxy[1]+medxy[0])/2**0.5, ml_dist_diff + (medxy[1]+medxy[0]-512)/2**0.5)
            else:
                label = "Right"
                medxy_adjusted = ((medxy[1]+medxy[0])/2**0.5, ml_dist_diff - (medxy[1]+medxy[0]-512)/2**0.5)
        else:
            ml_dist_diff = ml_dist*512//550-256
            if hemisphere == 'L':
                label = "Left"
                medxy_adjusted = (medxy[1], ml_dist_diff + medxy[0]) 
                # adjust y coordinate by  coordinate of the view related to midline
                # the left side of the image is snout
            else:
                label = "Right"
                medxy_adjusted = (medxy[1], ml_dist_diff + 512 - medxy[0])
        
        LR_list.append((medxy_adjusted, label))
        LR_labels.append(label) 
        """Be aware the ouput coordinate is pixcel here!"""
    return LR_list, LR_labels


"""this section measures movement artifacts and determines time ranges with significant movement"""

movement = np.gradient(np.hypot(ops['xoff'], ops['yoff'])) #calculates the first derivative of the xy vector of suite2P registration
movement_frames = np.argwhere(movement > movement_thr) #all frames in which movement > thr
movement_ranges = [] #creates new list of ranges based on movement frames with movement

cp_crop_switch = 0 # Set a switch to control whether crop CellPose ROI or not

if np.max(movement) > movement_thr:
    cp_crop_switch = 1
    print("significant movement artifacts around frame", 
          np.argmax(movement), 
          " with ", 
          (movement > movement_thr).sum(), 
          " frames above thr")


    def find_ranges(x_values, y_values, threshold, initial_range, offset):
        
        """Find frames around values of dxdy that exceed threshold, excluding an offset before the exceeding value,
        returns the list of indices for each range. you can have multiple ranges per recording"""
      
        range_indices = []
        i = 0
        while i < len(y_values):
            if y_values[i] > threshold:
                start_index = i + offset  # Start the range 'offset' points after the found value
                end_index = min(i + initial_range, len(y_values) - 1)
                
                # Ensure the start_index does not go below 0
                start_index = max(start_index, 0)
                
                # Extend the range if new frame with mov artifacts is found within initial_range
                while True:
                    extended = False
                    if end_index < len(y_values) - 1 and np.any(y_values[end_index + 1:min(end_index + initial_range, len(y_values))] > threshold):
                        end_index = min(end_index + initial_range, len(y_values) - 1)
                        extended = True
                    
                    if not extended:
                        break
                
                # Generate list of indices for the current range
                current_range_indices = list(range(start_index, end_index + 1))
                range_indices.append(current_range_indices)
                
                i = end_index + 1  # Skip ahead to the end of the current range.
            else:
                i += 1
        return range_indices #this has all the idx of frames before offset(-10) and after initial_window (150) around movement artifacts. If new movement artifact is found within window, it keeps extending the range

    movement_ranges = find_ranges(np.arange(len(movement)), movement, movement_thr, movement_win, -2)
    #mov_intervals = [(sublist[0], sublist[-1]) for sublist in movement_ranges if sublist]


"""creates dataframes with Suite2P F data""" 

F_subt = F - 0.7*Fneu #F_subt is the array with full data. 
F_corrected = np.copy(F_subt) #F_corrected will be either F_subt copy or with cropped movement artifact
Fneu_crop = np.copy(Fneu) # In case movement artifacts are cropped we generate Fneu and F cropped files to calculate avergare Fneu and F per ROI to show in Functionviewer
F_crop = np.copy(F)

def crop_and_filter(F_subt, F, Fneu, movement_ranges):

    """this function crops the time sections with movement artifacts detected in find_ranges. I also filters the joining edges around the cropped segments using a savgol filter
    
    IMPORTANT!!! right now it's being called from the tkinter GUI when you click Crop, but you can uncomment the line below and run it directly"""
    
    F_crop = np.delete(F, np.concatenate(movement_ranges), axis=1)
    Fneu_crop = np.delete(Fneu, np.concatenate(movement_ranges), axis=1)
    F_corrected = np.delete(F_subt, np.concatenate(movement_ranges), axis=1)
    crop_points = np.array([sublist[-1] for sublist in movement_ranges])
    cropped_ranges = np.cumsum([sublist[-1]-sublist[0] for sublist in movement_ranges])
    adjusted_crop_points = crop_points - cropped_ranges
    window_size = 15
    poly_order = 5

    for row in F_corrected:
        for index in adjusted_crop_points:
            start = max(index - window_size // 2, 0)
            end = min(index + window_size // 2 + 1, len(row))
            actual_window_size = min(end - start, window_size)
            if actual_window_size % 2 == 0:
                actual_window_size -= 1
            if actual_window_size > poly_order:
                row[start:end] = savgol_filter(row[start:end], actual_window_size, poly_order)
    
    return F_corrected, F_crop, Fneu_crop

#Run this line if you want to remove mov ranges without using the tkinter GUI
#F_corrected, F_crop, Fneu_crop = crop_and_filter(F_subt, movement_ranges)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotWindow(root, F_subt, F, Fneu, movement_ranges)
    root.mainloop()

#%% Load data from CellPose ROI if cellpose set as "1"
if cellpose == 1:     
    is_cell_cp = np.int64(np.hsplit(is_cell_Suite2P_cp,2)[0]) #gets the 0/1 column of is_cell
    idx_is_cell_cp = np.where(is_cell_cp[:,0]==1)[0] #gets ROI index where is_cell == 1
    
    is_e2_cp = np.zeros((len(F_cp), 1), dtype=int) # initializes is_e2
    is_pyr_cp = np.copy(is_cell_cp) #initializes is_pyr as a copy of is_cell that where e2 will be deleted from 
    
    
    e2_loaded_cp = False               
    
    if e2source == 'txt': #import e2 list from text, simple array where each number in txt file is ROi index
        e2_loaded_cp = True
        tdT_positive_cp = p2_cp.txt  
        tdT_positive_idx_cp = np.copy(tdT_positive_cp)  # Use np.copy to avoid modifying the original list
        condition = (is_cell_cp[tdT_positive_idx_cp] == 0) # exclude tdT+ cell that is not a cell
        tdT_positive_idx_cp = np.delete(tdT_positive_idx_cp, np.where(condition)[0])
       
        if tdT_positive_cp.size > 0: # if tdT_positive is not empty
            is_e2_cp[tdT_positive_idx_cp] = 1 
            is_pyr_cp[tdT_positive_idx_cp] = 0  # Exclude E2+ cells from is_cell list
            #idx_is_cell = np.where(is_cell[:,0]==1)[0] # updates idx_is_cell after exlusion of e2
            print(f"CellPose identified {tdT_positive_cp.size} E2+ cells.")
        else:
            print("CellPose identified 0 E2+ cells.")
    
    if not e2_loaded_cp:
        
        print("No E2 txt in this recording.")
            
    #Create dataframes with original index of e2 and non e2 so we can separate L/R based on stat.npy file
    
    idx_pyr_cp = np.where((is_pyr_cp[:,0]==1) & (is_cell_cp[:,0]==1))[0]
    idx_e2_cp = np.where((is_e2_cp[:,0]==1) & (is_cell_cp[:,0]==1))[0]
    
    print(f'CellPose ROI: {len(idx_pyr_cp)} non-E2 cells')
    print(f'CellPose ROI: {len(idx_e2_cp)} E2 cells')
    
    
    F_subt_cp = F_cp - 0.7*Fneu_cp #F_subt is the array with full data. 
    F_corrected_cp = np.copy(F_subt_cp) #F_corrected will be either F_subt copy or with cropped movement artifact
    Fneu_crop_cp = np.copy(Fneu_cp) # In case movement artifacts are cropped we generate Fneu and F cropped files to calculate avergare Fneu and F per ROI to show in Functionviewer
    F_crop_cp = np.copy(F_cp)
    
    if cp_crop_switch == 1:
        F_corrected_cp, F_crop_cp, Fneu_crop_cp = crop_and_filter(F_subt_cp, F_cp, Fneu_cp, movement_ranges)

#%% pre-exclude bad cells that averaged F is smaller than Fneu+20
"""filter out cell with F_crop < Fneu_crop"""
F_avg = np.average(F_crop,axis=1)
Fneu_avg = np.average(Fneu_crop,axis=1)
F_avg2 = F_avg - Fneu_avg

F_filter = np.zeros(len(F_crop),int)
for i in range(len(F_crop)):
    if F_avg2[i] >= 20:
        F_filter[i] = 1
        
for i in range(len(F_crop)):
    if F_filter[i] == 0:
        is_cell[i] = 0
        is_pyr[i] = 0
        is_e2[i] = 0

idx_pyr = np.where(is_pyr[:,0]==1)[0]
idx_e2 = np.where(is_e2[:,0]==1)[0]
Pyr_LR , LR_labels_pyr = classify_LR(ROIstat, idx_pyr, head_orientation)
E2_LR, LR_labels_e2 = classify_LR(ROIstat, idx_e2, head_orientation)

"""same for CellPose ROI"""
if cellpose == 1:
    F_avg_cp = np.average(F_crop_cp,axis=1)
    Fneu_avg_cp = np.average(Fneu_crop_cp,axis=1)
    F_avg2_cp = F_avg_cp - Fneu_avg_cp
    
    F_filter_cp = np.zeros(len(F_crop_cp),int)
    for i in range(len(F_crop_cp)):
        if F_avg2_cp[i] >= 20:
            F_filter_cp[i] = 1
            
    for i in range(len(F_crop_cp)):
        if F_filter_cp[i] == 0:
            is_cell_cp[i] = 0
            is_pyr_cp[i] = 0
            is_e2_cp[i] = 0
    
    idx_pyr_cp = np.where(is_pyr_cp[:,0]==1)[0]
    idx_e2_cp = np.where(is_e2_cp[:,0]==1)[0]
    Pyr_LR_cp , LR_labels_pyr_cp = classify_LR(ROIstat_cp, idx_pyr_cp, head_orientation)
    E2_LR_cp, LR_labels_e2_cp = classify_LR(ROIstat_cp, idx_e2_cp, head_orientation)


if cellpose == 1:
    F_corrected_all = np.concatenate((F_corrected, F_corrected_cp), axis=0)
    Pyr_LR_all = Pyr_LR + Pyr_LR_cp
    LR_labels_pyr_all = LR_labels_pyr + LR_labels_pyr_cp
    E2_LR_all = E2_LR + E2_LR_cp
    LR_labels_e2_all = LR_labels_e2 + LR_labels_e2_cp
else:
    F_corrected_all = F_corrected
    Pyr_LR_all = Pyr_LR
    LR_labels_pyr_all = LR_labels_pyr
    E2_LR_all = E2_LR
    LR_labels_e2_all = LR_labels_e2

#%% split PYR and E2 and measure df/f
df = pd.DataFrame(F_corrected)
df.columns = [f't{i}' for i in range(df.shape[1])]
df['meta_index'] = df.index
df['meta_cellpose'] = 0
df['meta_is_cell'] = pd.Series(is_cell[:,0], index=df.index)
df['meta_is_pyr'] = pd.Series(is_pyr[:,0], index=df.index)
df['meta_is_e2']= pd.Series(is_e2[:,0], index=df.index)  

if cellpose == 1:
    df_cp = pd.DataFrame(F_corrected_cp)
    df_cp.columns = [f't{i}' for i in range(df_cp.shape[1])]
    df_cp['meta_index'] = df_cp.index
    df_cp['meta_cellpose'] = 1
    df_cp['meta_is_cell'] = pd.Series(is_cell_cp[:,0], index=df_cp.index)
    df_cp['meta_is_pyr'] = pd.Series(is_pyr_cp[:,0], index=df_cp.index)
    df_cp['meta_is_e2']= pd.Series(is_e2_cp[:,0], index=df_cp.index) 
    
    """concat functional ROI and CellPose ROI"""
    df_all = pd.concat([df,df_cp], axis=0)
    df_pyr = df_all[(df_all['meta_is_pyr'] == 1)]
    df_e2 = df_all[(df_all['meta_is_cell'] == 1) & (df_all['meta_is_e2'] == 1)]
    
else:
    df_pyr = df[(df['meta_is_pyr'] == 1)]
    df_e2 = df[(df['meta_is_cell'] == 1) & (df['meta_is_e2'] == 1)]
           
             
def calculate_dff(df, window_size, percentile): 
    
    '''F0 calculated as rolling 8th percentile for 120 sec as Runyan'''
    # Apply the rolling window and calculate the percentile for each frame
    baseline = df.apply(lambda row: row.rolling(window=window_size, min_periods=1, center=False).quantile(percentile / 100.0), axis=1)
    
    # Calculate the normalized fluorescence signal
    dff = (df - baseline) / abs(baseline)
    
    return dff    

meta_mask_pyr = df_pyr.columns.str.startswith('meta')
meta_data_pyr = df_pyr.loc[:, meta_mask_pyr].reset_index(drop=True)
pyr_traces_df = df_pyr.loc[:, ~meta_mask_pyr]
dff_win_size = SR * 120 # need to be adjust based on personal needed 
dff_percentile = 8 # need to be adjust based on personal needed
dff_df = calculate_dff(pyr_traces_df, dff_win_size, dff_percentile)
dff_df.rename(columns=lambda x: int(x[1:]), inplace=True)
dff_df.reset_index(drop=True, inplace=True)

meta_mask_e2 = df_e2.columns.str.startswith('meta')
meta_data_e2 = df_e2.loc[:, meta_mask_e2].reset_index(drop=True)
e2_traces_df = df_e2.loc[:, ~meta_mask_e2]
dff_win_size_e2 = SR * 120 # need to be adjust based on personal needed  
dff_percentile_e2 = 8 # need to be adjust based on personal needed
dff_e2 = calculate_dff(e2_traces_df, dff_win_size_e2, dff_percentile_e2)
dff_e2.rename(columns=lambda x: int(x[1:]), inplace=True)
dff_e2.reset_index(drop=True, inplace=True)

#%% rename columns to int
df_raw = df_pyr.iloc[:,:-5]
df_raw_e2 = df_e2.iloc[:,:-5]
df_raw.rename(columns=lambda x: int(x[1:]), inplace=True)
df_raw_e2.rename(columns=lambda x: int(x[1:]), inplace=True)

#%% denoise 
import pywt

def denoise(trace, wavelet='db4', level=2):
    coeffs = pywt.wavedec(trace, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(i, value=4*np.median(np.abs(i - np.median(i)))/0.6745, mode='soft') for i in coeffs[1:]]
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    
    # Ensure the reconstructed signal matches the original length
    if len(reconstructed_signal) > len(trace):
        reconstructed_signal = reconstructed_signal[:len(trace)]  # Trim to original length
    elif len(reconstructed_signal) < len(trace):
        reconstructed_signal = np.pad(reconstructed_signal, (0, len(trace) - len(reconstructed_signal)), 'constant')  # Pad to original length
    
    return reconstructed_signal

def wavelet_filter_df(df, wavelet='db4', level=2):
    if df.empty:
        print("Input DataFrame is empty. Returning empty DataFrame and empty noise list.")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrame and empty list if input is empty

    # Apply the denoise function to each row
    denoised_data = np.apply_along_axis(denoise, axis=1, arr=df.to_numpy(), wavelet=wavelet, level=level)
    denoised_df = pd.DataFrame(denoised_data, index=df.index, columns=df.columns)
    
    # Calculate the difference between the original and denoised signals
    diff_df = df.subtract(denoised_df)

    return denoised_df, diff_df

# smooth dF/F
sm_dff_df, dff_noise_df = wavelet_filter_df(dff_df, wavelet='db4', level=2)
dff_noise_rms = dff_noise_df.apply(lambda row: np.sqrt(np.mean(np.square(row))), axis=1).tolist()

sm_dff_e2, dff_noise_e2 = wavelet_filter_df(dff_e2, wavelet='db4', level=2)
dff_noise_rms_e2 = dff_noise_e2.apply(lambda row: np.sqrt(np.mean(np.square(row))), axis=1).tolist()

"""Measure noise size of each df. We measure the difference b/w raw df/f and smoothed df/f and take values within 10% and 90% quantile of rolling window as noise. The noise size is defined as the average of noise accross the trace."""
def noise_size_measure(noise_df):
    list_noise_size = []
    for i in range(len(noise_df)):
        x = noise_df.iloc[i,:]
        dt = x.rolling(window=SR*10, min_periods=1).quantile(0.9)
        db = x.rolling(window=SR*10, min_periods=1).quantile(0.1)
        ns = (dt-db).mean()
        list_noise_size.append(ns)
    return(list_noise_size)

noise_size_dff_df = noise_size_measure(dff_noise_df)
noise_size_dff_e2 = noise_size_measure(dff_noise_e2)

#%% calculate z_score trace based on df/f

def z_score_trace(df, df_noise_size, window_size):
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd

    z_scored_rows = []
    z_baseline_idx = []
    zs_baseline_mean = []

    for index, row in df.iterrows():
        # Determine the threshold as 5% quantile + noise size*1.1
        fiveth_percentile_value = row.quantile(0.05)
        noise_size = df_noise_size[index]
        threshold = fiveth_percentile_value + noise_size*1.1
        
        # Calculate the rolling mean and standard deviation for the entire row
        rolling_mean = row.rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = row.rolling(window=window_size, center=True, min_periods=1).std()
        
        # Indices of values in the row that are below the threshold
        below_threshold_mask = rolling_mean <= threshold
        below_threshold_indices = np.where(below_threshold_mask)[0]  # Convert boolean mask to positional indices
        
        if below_threshold_indices.size == 0:
            ten_percentile_value = row.quantile(0.1)
            threshold = ten_percentile_value + noise_size*1.1
            below_threshold_mask = rolling_mean <= threshold
            below_threshold_indices = np.where(below_threshold_mask)[0]

        if len(below_threshold_mask) > 0:
            # Calculate the mean and std at the indices below the 10th percentile
            min_std_index = below_threshold_indices[np.argmin(rolling_std.iloc[below_threshold_indices])]
            z_baseline_std = rolling_std.iloc[min_std_index]
            z_baseline_mean = rolling_mean.iloc[min_std_index]  # Use positional index to retrieve mean

            # Compute z-score for the entire row using the calculated mean and std
            z_scored_row = (row - z_baseline_mean) / z_baseline_std
        else:
            # If no values are below the 10th percentile, fill the row with zeros
            z_scored_row = pd.Series(np.zeros(len(row)), index=row.index)

        z_scored_rows.append(z_scored_row)
        zs_baseline_mean.append(z_baseline_mean)
        z_baseline_idx.append(below_threshold_indices.tolist())  # Store positional indices

    # Convert list of Series to DataFrame; this will include rows with all zeros where applicable
    z_scored_data = pd.DataFrame(z_scored_rows, index=df.index).reset_index(drop=True)

    return z_scored_data, zs_baseline_mean


zs_df, z_baseline_df = z_score_trace(dff_df, noise_size_dff_df, window_size=10*SR) #zs_df has the zcored data without metadata 
zs_df.reset_index(drop=True, inplace=True)# reset column index tp facilitate operations. original is maintained in the 'meta_index' column of meta_data 
z_scored_df = pd.concat([zs_df, meta_data_pyr], axis=1) #this has the z_scored data with meta data appended as last columns

zs_e2, z_baseline_e2 = z_score_trace(dff_e2, noise_size_dff_e2, window_size=10*SR)
zs_e2.reset_index(drop=True, inplace=True)
z_scored_e2 = pd.concat([zs_e2, meta_data_e2], axis=1)

#filter z-scored dF/F
sm_z_scored_df, noise_df = wavelet_filter_df(zs_df, wavelet='db4', level=2)
noise_rms = noise_df.apply(lambda row: np.sqrt(np.mean(np.square(row))), axis=1).tolist()

sm_z_scored_e2, noise_e2 = wavelet_filter_df(zs_e2, wavelet='db4', level=2)
noise_rms_e2 = noise_e2.apply(lambda row: np.sqrt(np.mean(np.square(row))), axis=1).tolist()

#noise size of z-scored dF/F
noise_size_zs_df = noise_size_measure(noise_df)
noise_size_zs_e2 = noise_size_measure(noise_e2)

#%% peaks, onsets, and tails finding
"""
First we indentify the peaks based on smoothed df/f, then define the onset point based on the peaks. 
Onset point is found in a chunk of smoothed df/f before the peak. The min value and the slope of that chunk is used to identify the onset. Since smoothing shifts the trace, both peaks and onsets indices need to be adjusted for the df/f by finding the local max and min. Also, here we identify another potential onset point by finding the min in the chunk of df/f before peak. We use slope to decide whether the adj_onset by local min or min of chunk is a better onset point.
We use y_vlaue of peaks as peak size to define 37% of peak size as peak tail, and find the tail in local smoothed trace.
"""

def find_peaks_in_array(data, list_noise_size, distance=None, prominence=None, width=None):
    peaks_indices = []
    for i, row in data.iterrows() if hasattr(data, 'iterrows') else enumerate(data):
        x = row.values if hasattr(row, 'values') else row
        cutoff = list_noise_size[i] * 1.2 # 1.2x noise size as cutoff
        peaks, _ = find_peaks(x, height=cutoff, distance=distance, prominence=prominence, width=width)
        peaks_indices.append(peaks)

    return peaks_indices

def find_onset (data, peaks_indices, upperlimit=15):
    onset_indice_list = []
    for i, row in data.iterrows() if hasattr(data, 'iterrows') else enumerate(data):
        x = row.values if hasattr(row, 'values') else row
        onset_indices = []
        for j, peaks in enumerate(peaks_indices[i]):
            if j == 0:
                if x[peaks] <= 1.5:
                    chunk_start = peaks_indices[i][j] - upperlimit
                    chunk_start = 0 if chunk_start < 0 else chunk_start
                else: # if peak value is > 1.5, we set bigger window for searching the onset
                    chunk_start = peaks_indices[i][j] - upperlimit - 5
                    chunk_start = 0 if chunk_start < 0 else chunk_start
                chunk_y = x[chunk_start:peaks_indices[i][j]+1] 
            else:
                interval = peaks_indices[i][j] - peaks_indices[i][j-1]
                if x[peaks] <= 1.5:
                    chunk_start = peaks_indices[i][j] - upperlimit
                    chunk_start = peaks_indices[i][j-1] if interval < upperlimit else chunk_start
                else:
                    chunk_start = peaks_indices[i][j] - upperlimit - 5
                    chunk_start = 0 if chunk_start < 0 else chunk_start
                chunk_end = peaks_indices[i][j]
                chunk_y = x[chunk_start:chunk_end] # chunk is the range b/w two peaks        
            
            min_index = np.argmin(chunk_y) # find the min value in the chunk
            values_after_min = chunk_y[min_index:]
            valid_indices = np.where((values_after_min >= chunk_y[min_index]) & (values_after_min - chunk_y[min_index] < 0.15))[0] # find all similar values afte the min
            slope_list = [] 
            for k in valid_indices: 
                # measure the slope of all potential onset relative to the peak
                slope = (x[peaks] - x[peaks-len(chunk_y)+k]) / (len(chunk_y)-k)
                slope_list.append(slope)
            slope_list = np.array(slope_list)
            max_slope = np.max(slope_list)
            valid_indices2 = np.where(slope_list > max_slope-0.03)[0]
            y_value_list = []
            for k in valid_indices2:
                y_value = x[chunk_start + min_index + valid_indices[k]]
                y_value_list.append(y_value)
            y_value_list = np.array(y_value_list)
            min_y_value_ind = np.argmin(y_value_list)
            # the true onset is defined as the frame with slope bigger than the threshold (max_slope-0.03) that is the nearest to the peak
            onset_index = valid_indices[valid_indices2[min_y_value_ind]]
            onset_indice = chunk_start + min_index + onset_index
            onset_indices.append(onset_indice)
        onset_indice_list.append(onset_indices)
        
    return onset_indice_list


def find_local_max(df, row_index, peak_index, left_interval, right_interval, window_size):
    if left_interval <= window_size*2:
        start_index = (peak_index - left_interval//2) + 1
    else:
        start_index = peak_index - window_size+1
    if right_interval <= window_size*2:
        end_index = (peak_index + right_interval//2)
    else:
        end_index =  peak_index + window_size
    peak_segment = df.iloc[row_index, start_index:end_index]
    local_max_index = peak_segment.idxmax()
    adjusted_index = df.columns.get_loc(local_max_index)
    
    return adjusted_index

def adjust_peaks(df, peaks_indices, window_size):
    x_adjusted = []
    for row_index, peaks in enumerate(peaks_indices):
        adjusted_peaks = []
        for k, peak_index in enumerate(peaks):
            if k == 0:
                left_interval = window_size+1
            else:
                left_interval = peaks[k] - peaks[k-1]
            if k == len(peaks)-1:
                right_interval = window_size+1
            else:
                right_interval = peaks[k+1] - peaks[k]
            adjusted_index = find_local_max(df, row_index, peak_index, left_interval, right_interval, window_size)
            adjusted_peaks.append(adjusted_index)
        x_adjusted.append(np.array(adjusted_peaks))

    return x_adjusted

def find_local_min(df, row_index, onset_index, left_interval, right_interval, window_size):
    if 0 <= left_interval <= window_size:
        start_index = onset_index - left_interval +1
    elif left_interval < 0: # consider if the adj_x of previous peak shifts to the right of onset_index
        start_index = onset_index - left_interval +1
        end_index = onset_index + right_interval
    else:
        start_index = onset_index - window_size
        
    if left_interval >= 0:
        if 0 < right_interval <= window_size:
            end_index = onset_index + right_interval
        elif right_interval == 0:
            end_index = onset_index
        elif right_interval < 0: # consider if the adj_x of peak shifts to the left of onset_index
            start_index = onset_index - left_interval +1
            end_index = onset_index + right_interval
        else:
            end_index = onset_index + window_size
    
    # peak_segment = df.iloc[row_index, start_index:end_index]
    # local_min_ind = peak_segment.idxmin()
    # adjusted_index = df.columns.get_loc(local_min_ind)
    
    peak_segment = df.iloc[row_index, start_index:end_index]
    min_value = peak_segment.min()
    threshold = min_value + 0.05
    below_threshold = peak_segment[peak_segment < threshold]
    last_below_threshold_index = below_threshold.index[-1]
    adjusted_index = df.columns.get_loc(last_below_threshold_index)
    
    return adjusted_index

def adjust_onset(df, adj_peaks_indices, onsets_indices, window_size):
    x_adjusted = []
    for row_index, (peaks, onsets) in enumerate(zip(adj_peaks_indices ,onsets_indices)):
        adjusted_onsets = []
        for k, (peak_index, onset_index) in enumerate(zip(peaks, onsets)):
            if k == 0:
                left_interval = onset_index
            else:
                left_interval = onset_index - peaks[k-1]
            right_interval = peak_index - onset_index
            
            adjusted_index = find_local_min(df, row_index, onset_index, left_interval, right_interval, window_size) # find local min adjusted index
            
            adjusted_onsets.append(adjusted_index)
        x_adjusted.append(np.array(adjusted_onsets))
        
    return x_adjusted

"""Event tail is defined in frames b/w the peak and next event onset. We set the first frame smaller than 37% event amplitude as tau; if tau is not found, the next event onst will be set as the current event tail"""
def find_tail (data, peaks_adj, onsets_adj, window_size, sm_level=3, upperlimit=30):
    tails = []
    for i, row in data.iterrows() if hasattr(data, 'iterrows') else enumerate(data):
        x = row.values if hasattr(row, 'values') else row
        row_tails = []
        for j, (peak, onset) in enumerate(zip(peaks_adj[i], onsets_adj[i])):
            if j == len(peaks_adj[i])-1:
                next_onset = len(x)-1
                next_onset = peak + upperlimit if (len(x)-1)-peak > upperlimit else next_onset
            else:
                next_onset = onsets_adj[i][j+1]
                next_onset = peak + upperlimit if onsets_adj[i][j+1]-peak > upperlimit else next_onset
            chunk_y = x[peak:next_onset+1]
            sm_chunk_y = gaussian_filter1d(chunk_y, sigma=sm_level)
            peak_amp = x[peak]
            decay = x[peak] - 0.63*peak_amp
            if np.min(sm_chunk_y) <= decay:
                decay_index = np.where(sm_chunk_y <= decay)[0][0]
            else:
                decay_index = len(sm_chunk_y)-1
            tail_indice_temp = peak + decay_index
            range_left = min(decay_index-1, window_size)
            range_right = min((next_onset - tail_indice_temp), window_size)
            chunk_adj = x[tail_indice_temp-range_left:tail_indice_temp+range_right+1]
            if np.min(chunk_adj) > decay:
                closest_indice = len(chunk_adj)-1
            else:
                closest_indice = np.argmin(chunk_adj)
            tail_indice = tail_indice_temp - range_left + closest_indice
            row_tails.append(tail_indice)
        tails.append(row_tails)
        
    return tails


#peak find with dF/F  
dff_distance_between_peaks = SR//5  
dff_prominence_threshold = 0.5 
dff_width_threshold = 3

dff_peaks_indices = find_peaks_in_array(sm_dff_df, noise_size_dff_df, distance=dff_distance_between_peaks, prominence=dff_prominence_threshold, width=dff_width_threshold)

dff_distance_between_peaks_e2 = SR//7.5
dff_prominence_threshold_e2 = 0.4
dff_width_threshold_e2 = 1

dff_peaks_indices_e2 = find_peaks_in_array(sm_dff_e2, noise_size_dff_e2, distance=dff_distance_between_peaks_e2, prominence=dff_prominence_threshold_e2, width=dff_width_threshold_e2)

x_adj_dff = adjust_peaks (dff_df, dff_peaks_indices, window_size=3)
x_adj_dff_e2 = adjust_peaks (dff_e2, dff_peaks_indices_e2, window_size=3)

# find onset in sm_trace
onset_indice_dff_sm = find_onset (sm_dff_df, dff_peaks_indices, upperlimit=15)
onset_indice_dff_sm_e2 = find_onset (sm_dff_e2, dff_peaks_indices_e2, upperlimit=15)

# adjust onset indice
onset_adj_dff = adjust_onset(dff_df, x_adj_dff, onset_indice_dff_sm, window_size = 5)
onset_adj_dff_e2 = adjust_onset(dff_e2, x_adj_dff_e2, onset_indice_dff_sm_e2, window_size = 5)


# calculate amplitude and filter out values are smaller than 0.5
def peak_size_calculations(df, peak_adj, onset_adj):
    amplitudes = []
    for row_idx in range(len(df)):
        row_amplitudes = []
        peaks = peak_adj[row_idx]
        onsets = onset_adj[row_idx]
        for peak_idx, (peak, onset) in enumerate(zip(peaks, onsets)):
            y_peak = df.iloc[row_idx, peak]
            amplitude = y_peak - df.iloc[row_idx, onset]
            row_amplitudes.append(amplitude)
        amplitudes.append(row_amplitudes)
    
    return amplitudes

peak_amplitudes_dff = peak_size_calculations(dff_df, dff_peaks_indices, onset_adj_dff)
peak_amplitudes_dff_e2 = peak_size_calculations(dff_e2, x_adj_dff_e2, onset_adj_dff_e2)

# filter out small peaks in dff
filtered_peaks = []
filtered_peaks_adj = []
filtered_onset = []
filtered_onset_adj = []

for list1, list2, list3, list4, list5 in zip(peak_amplitudes_dff, dff_peaks_indices, x_adj_dff, onset_indice_dff_sm, onset_adj_dff):
    filtered_pair = [(v1, v2, v3, v4, v5) for v1, v2, v3, v4, v5 in zip(list1, list2, list3, list4, list5) if v1 > 0.5]
    filtered_peaks.append([v2 for _, v2, _, _, _ in filtered_pair])
    filtered_peaks_adj.append([v3 for _, _, v3, _, _ in filtered_pair])
    filtered_onset.append([v4 for _, _, _, v4, _ in filtered_pair])
    filtered_onset_adj.append([v5 for _, _, _, _, v5 in filtered_pair])

dff_peaks_indices = filtered_peaks
x_adj_dff = filtered_peaks_adj
onset_indice_dff_sm = filtered_onset
onset_adj_dff = filtered_onset_adj

# filter out small peaks in dff_e2
filtered_peaks = []
filtered_peaks_adj = []
filtered_onset = []
filtered_onset_adj = []

for list1, list2, list3, list4, list5 in zip(peak_amplitudes_dff_e2, dff_peaks_indices_e2, x_adj_dff_e2, onset_indice_dff_sm_e2, onset_adj_dff_e2):
    filtered_pair = [(v1, v2, v3, v4, v5) for v1, v2, v3, v4, v5 in zip(list1, list2, list3, list4, list5) if v1 > 0.5]
    filtered_peaks.append([v2 for _, v2, _, _, _ in filtered_pair])
    filtered_peaks_adj.append([v3 for _, _, v3, _, _ in filtered_pair])
    filtered_onset.append([v4 for _, _, _, v4, _ in filtered_pair])
    filtered_onset_adj.append([v5 for _, _, _, _, v5 in filtered_pair])

dff_peaks_indices_e2 = filtered_peaks
x_adj_dff_e2 = filtered_peaks_adj
onset_indice_dff_sm_e2 = filtered_onset
onset_adj_dff_e2 = filtered_onset_adj


# find adjusted tail indice
tail_adj_dff = find_tail (dff_df, x_adj_dff, onset_adj_dff, window_size = 2, sm_level=4, upperlimit=30)
tail_adj_dff_e2 = find_tail (dff_e2, x_adj_dff_e2, onset_adj_dff_e2, window_size = 2, sm_level=4, upperlimit=30)

#%% peak_calculations
def peak_calculations(df, peak_adj, onset_adj, tail_adj):
    amplitudes = []
    y_peaks = []
    risetimes = []
    durations = []
    aucs = []
    
    for row_idx in range(len(df)):
        row_amplitudes = []
        row_y_peaks = []
        row_risetimes = []
        row_durations = []
        row_aucs = []
        
        peaks = peak_adj[row_idx]
        onsets = onset_adj[row_idx]
        tails = tail_adj[row_idx]
        
        for peak_idx, (peak, onset, tail) in enumerate(zip(peaks, onsets, tails)):
            y_peak = df.iloc[row_idx, peak]
            amplitude = y_peak - df.iloc[row_idx, onset]
            row_amplitudes.append(amplitude)
            row_y_peaks.append(y_peak)
            risetime = peak - onset
            row_risetimes.append(risetime)
            duration = tail - onset
            row_durations.append(duration)
            peak_values = df.iloc[row_idx, onset:tail+1]  
            auc = np.trapz(peak_values, dx=1)
            row_aucs.append(auc)
            
        amplitudes.append(row_amplitudes)
        y_peaks.append(row_y_peaks)
        risetimes.append(row_risetimes)
        durations.append(row_durations)
        aucs.append(row_aucs)
        
    return amplitudes, y_peaks, risetimes, durations, aucs

peak_amplitudes, y_peaks, peak_risetimes, peak_durations, peak_aucs = peak_calculations(zs_df, x_adj_dff, onset_adj_dff, tail_adj_dff)
peak_amplitudes_e2, y_peaks_e2, peak_risetimes_e2, peak_durations_e2, peak_aucs_e2  = peak_calculations(zs_e2, x_adj_dff_e2, onset_adj_dff_e2, tail_adj_dff_e2)

peak_amplitudes_dff, y_peaks_dff, peak_risetimes_dff, peak_durations_dff, peak_aucs_dff = peak_calculations(dff_df, x_adj_dff, onset_adj_dff, tail_adj_dff)
peak_amplitudes_dff_e2, y_peaks_dff_e2, peak_risetimes_dff_e2, peak_durations_dff_e2, peak_aucs_dff_e2 = peak_calculations(dff_e2, x_adj_dff_e2, onset_adj_dff_e2, tail_adj_dff_e2)


# find decay (37% of peak amplitude)
def find_decay (data, peaks_adj, tails_adj, peak_amps, window_size, sm_level=3):
    decays_indice = []
    taus = []
    for i, row in data.iterrows() if hasattr(data, 'iterrows') else enumerate(data):
        x = row.values if hasattr(row, 'values') else row
        row_decays_indice = []
        row_taus = []
        for j, (peak, tail, peak_amp) in enumerate(zip(peaks_adj[i], tails_adj[i], peak_amps[i])):
            chunk_y = x[peak:tail+1]
            sm_chunk_y = gaussian_filter1d(chunk_y, sigma=sm_level)
            decay = x[peak] - 0.63*peak_amp
            if np.min(sm_chunk_y) <= decay:
                decay_index = np.where(sm_chunk_y <= decay)[0][0]
                decay_indice_temp = peak + decay_index
                range_left = min(decay_index-1, window_size)
                range_right = min((tail - decay_indice_temp), window_size)
                chunk_adj = x[decay_indice_temp-range_left:decay_indice_temp+range_right+1]
                if np.min(chunk_adj) <= decay:
                    filtered_chunk = chunk_adj[chunk_adj <= decay]
                    closest_indice_filtered = np.abs(filtered_chunk-decay).argmin()
                    closest_value = filtered_chunk[closest_indice_filtered]
                    closest_indice = np.where(chunk_adj == closest_value)[0][0]
                    decay_indice = decay_indice_temp - range_left + closest_indice
                    decay_time = decay_indice - peak
                else:
                    decay_indice = np.nan
                    decay_time = np.nan
            else:
                decay_indice = np.nan
                decay_time = np.nan
            row_decays_indice.append(decay_indice)
            row_taus.append(decay_time)
        decays_indice.append(row_decays_indice)
        taus.append(row_taus)
        
    return decays_indice, taus


# find decay and tau
decay_adj_dff, peak_taus_dff = find_decay (dff_df, x_adj_dff, tail_adj_dff, peak_amplitudes_dff, window_size = 2, sm_level=4)
decay_adj_dff_e2, peak_taus_dff_e2 = find_decay (dff_e2, x_adj_dff_e2, tail_adj_dff_e2, peak_amplitudes_dff_e2, window_size = 2, sm_level=4)

#%% calculation of ROI parameters:
number_of_dffpeaks = [len(peaks) for peaks in dff_peaks_indices]
number_of_dffpeaks_e2 = [len(peaks) for peaks in dff_peaks_indices_e2]

dffpeak_Hz = [peaks / ((len(F_corrected_all[0]))/SR) for peaks in number_of_dffpeaks]
dffpeak_Hz_e2 = [peaks / ((len(F_corrected_all[0]))/SR) for peaks in number_of_dffpeaks_e2]

Fcorr = df_pyr.apply(lambda row: np.nanmean(row), axis=1).tolist() #this is F-Fneu
Fcorr_e2 = df_e2.apply(lambda row: np.nanmean(row), axis=1).tolist() #this is F-Fneu e2

Avg_dff = dff_df.apply(lambda row: np.nanmean(row), axis=1).tolist() #Avg df/f across entire trace
Avg_dff_e2 = dff_e2.apply(lambda row: np.nanmean(row), axis=1).tolist() #Avg df/f across entire trace e2

Avg_zs = zs_df.apply(lambda row: np.nanmean(row), axis=1).tolist() #Avg zs across entire trace
Avg_zs_e2 = zs_e2.apply(lambda row: np.nanmean(row), axis=1).tolist() #Avg zs across entire trace e2

if cellpose == 1:
    df_neuropil_pyr = pd.concat([pd.DataFrame(Fneu_crop[idx_pyr]),pd.DataFrame(Fneu_crop_cp[idx_pyr_cp])])
    Fneuropil_pyr = df_neuropil_pyr.apply(lambda row: np.nanmean(row), axis=1).tolist()
    
    df_neuropil_e2 = pd.concat([pd.DataFrame(Fneu_crop[idx_e2]),pd.DataFrame(Fneu_crop_cp[idx_e2_cp])])
    Fneuropil_e2 = df_neuropil_e2.apply(lambda row: np.nanmean(row), axis=1).tolist()
    
    df_F = pd.concat([pd.DataFrame(F_crop[idx_pyr]),pd.DataFrame(F_crop_cp[idx_pyr_cp])])
    Ftot = df_F.apply(lambda row: np.nanmean(row), axis=1).tolist()
    
    df_F_e2 = pd.concat([pd.DataFrame(F_crop[idx_e2]),pd.DataFrame(F_crop_cp[idx_e2_cp])])
    Ftot_e2 = df_F_e2.apply(lambda row: np.nanmean(row), axis=1).tolist()
    
else:
    df_neuropil_pyr = pd.DataFrame(Fneu_crop[idx_pyr])
    Fneuropil_pyr = df_neuropil_pyr.apply(lambda row: np.nanmean(row), axis=1).tolist()

    df_neuropil_e2 = pd.DataFrame(Fneu_crop[idx_e2])
    Fneuropil_e2 = df_neuropil_e2.apply(lambda row: np.nanmean(row), axis=1).tolist()

    df_F = pd.DataFrame(F_crop[idx_pyr])
    Ftot = df_F.apply(lambda row: np.nanmean(row), axis=1).tolist()

    df_F_e2 = pd.DataFrame(F_crop[idx_e2])
    Ftot_e2 = df_F_e2.apply(lambda row: np.nanmean(row), axis=1).tolist()

ROI_zpeak_amplitudes = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_amplitudes]
ROI_zpeak_yvalues = [np.nanmean(row) if len(row) > 0 else np.nan for row in y_peaks]
ROI_zpeak_aucs = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_aucs]

ROI_zpeak_amplitudes_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_amplitudes_e2]
ROI_zpeak_yvalues_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in y_peaks_e2]
ROI_zpeak_aucs_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_aucs_e2]



ROI_dffpeak_amplitudes = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_amplitudes_dff]
ROI_dffpeak_yvalues = [np.nanmean(row) if len(row) > 0 else np.nan for row in y_peaks_dff]
ROI_dffpeak_risetimes = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_risetimes_dff]
ROI_dffpeak_durations = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_durations_dff]
ROI_dffpeak_aucs = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_aucs_dff]
ROI_dffpeak_taus = [np.nanmean(row) if len(row) > 0 and not all(np.isnan(row)) else np.nan for row in peak_taus_dff]

ROI_dffpeak_amplitudes_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_amplitudes_dff_e2]
ROI_dffpeak_yvalues_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in y_peaks_dff_e2]
ROI_dffpeak_risetimes_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_risetimes_dff_e2]
ROI_dffpeak_durations_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_durations_dff_e2]
ROI_dffpeak_aucs_e2 = [np.nanmean(row) if len(row) > 0 else np.nan for row in peak_aucs_dff_e2]
ROI_dffpeak_taus_e2 = [np.nanmean(row) if len(row) > 0 and not all(np.isnan(row)) else np.nan for row in peak_taus_dff_e2]

#%% measure AUC of the entire trace
def AUC_trace(df):
    AUC = []
    for i in range(len(df)):
        a = df.iloc[i]
        a[a < 0] = 0
        AUC.append(np.trapz(a))
    return(AUC)

auc_dff_df = AUC_trace(sm_dff_df)
auc_dff_e2 = AUC_trace(sm_dff_e2)
auc_zs_df = AUC_trace(sm_z_scored_df)
auc_zs_e2 = AUC_trace(sm_z_scored_e2)

#%% get deconvolved spikes
def deconvolve_df(df, threshold=0.05):
    """
    Applies the Oasis AR1 FOOPSI deconvolution algorithm to each row of a DataFrame.
    Filters deconvolved events based on amplitude threshold. Additionally, calculates
    the number of deconvolved peaks and their average amplitude.

    Parameters:
        df (pd.DataFrame): DataFrame containing dF/F traces for deconvolution.
        threshold (float): Amplitude threshold for filtering deconvolved events.

    Returns:
        deconvolved_df: DataFrame with deconvolved signals where events below the threshold are set to zero.
        num_peaks_list: Number of deconvolved peaks per row above the threshold.
        avg_amplitude_list: Average amplitude of deconvolved peaks per row above the threshold.
    """
    deconvolved_signals = []
    freq_list = []
    avg_amplitude_list = []

    for index, row in df.iterrows():
        c, s, b, g, lam = deconvolve(row.values, penalty=1)  # Deconvolution with default parameters; adjust as needed
        
        # Remove events below the threshold
        s_filtered = np.where(s > threshold, s, 0)
        
        deconvolved_signals.append(s_filtered)

        # Count and average only the deconvolved events above the threshold
        peaks_above_threshold = s[s > threshold]
        freq_list.append(len(peaks_above_threshold)/df.shape[1])
        avg_amplitude = np.mean(peaks_above_threshold) if len(peaks_above_threshold) > 0 else 0
        avg_amplitude_list.append(float(avg_amplitude))

    # Convert the list of arrays back into a DataFrame for consistency with the input format
    deconvolved_df = pd.DataFrame(deconvolved_signals, index=df.index, columns=df.columns)

    return deconvolved_df, freq_list, avg_amplitude_list

deconvolved_df, decon_peakfreq, decon_peakamp = deconvolve_df(dff_df)

deconvolved_e2, decon_peakfreq_e2, decon_peakamp_e2 = deconvolve_df(dff_e2)


#%% final data results    

Pyr_df = meta_data_pyr.copy()
Pyr_df['LR'] = LR_labels_pyr_all
Pyr_df['zpeak Amp'] = ROI_zpeak_amplitudes
Pyr_df['zpeak Yvalue'] = ROI_zpeak_yvalues
Pyr_df['zpeak AUC'] = ROI_zpeak_aucs
Pyr_df['zs AUC'] = auc_zs_df

Pyr_df['dffpeak Hz'] = dffpeak_Hz
Pyr_df['dffpeak Amp'] = ROI_dffpeak_amplitudes
Pyr_df['dffpeak Yvalue'] = ROI_dffpeak_yvalues
Pyr_df['dffpeak Risetime'] = [x / SR for x in ROI_dffpeak_risetimes]
Pyr_df['dffpeak Duration'] = [x / SR for x in ROI_dffpeak_durations]
Pyr_df['dffpeak Tau'] = [x / SR for x in ROI_dffpeak_taus]
Pyr_df['dffpeak AUC'] = ROI_dffpeak_aucs
Pyr_df['dff AUC'] = auc_dff_df
Pyr_df['noise size'] = noise_size_dff_df

Pyr_df['F corrected'] = Fcorr
Pyr_df['F'] = Ftot
Pyr_df['Fneu'] = Fneuropil_pyr
Pyr_df['Avg dF/F'] = Avg_dff
Pyr_df['Avg zs'] = Avg_zs
Pyr_df['zs_baseline'] = z_baseline_df
Pyr_df['deconv Amp'] = decon_peakamp
Pyr_df['deconv Peaks'] = decon_peakfreq



E2_df = meta_data_e2.copy()
E2_df['LR'] = LR_labels_e2_all
E2_df['zpeak Amp'] = ROI_zpeak_amplitudes_e2
E2_df['zpeak Yvalue'] = ROI_zpeak_yvalues_e2
E2_df['zpeak AUC'] = ROI_zpeak_aucs_e2
E2_df['zs AUC'] = auc_zs_e2

E2_df['dffpeak Hz'] = dffpeak_Hz_e2
E2_df['dffpeak Amp'] = ROI_dffpeak_amplitudes_e2
E2_df['dffpeak Yvalue'] = ROI_dffpeak_yvalues_e2
E2_df['dffpeak Risetime'] = [x / SR for x in ROI_dffpeak_risetimes_e2]
E2_df['dffpeak Duration'] = [x / SR for x in ROI_dffpeak_durations_e2]
E2_df['dffpeak Tau'] = [x / SR for x in ROI_dffpeak_taus_e2]
E2_df['dffpeak AUC'] = ROI_dffpeak_aucs_e2
E2_df['dff AUC'] = auc_dff_e2
E2_df['noise size'] = noise_size_dff_e2

E2_df['F corrected'] = Fcorr_e2
E2_df['F'] = Ftot_e2
E2_df['Fneu'] = Fneuropil_e2
E2_df['Avg dF/F'] = Avg_dff_e2
E2_df['Avg zs'] = Avg_zs_e2
E2_df['zs_baseline'] = z_baseline_e2
E2_df['deconv Amp'] = decon_peakamp_e2
E2_df['deconv Peaks'] = decon_peakfreq_e2



def QC_ROI(df, criteria, ignore_nan=True):
    """
    Evaluate each row in the dataframe based on provided criteria and 
    append a 'QC' column to the dataframe to meta info indicating 'Pass' or 'Fail'.
    
    Parameters:
    - df: DataFrame to process.
    - criteria: Dictionary specifying the criteria for each column to be evaluated. 
                Format: {column_name: (condition, threshold)}
                Condition can be '>', '<', or '=', where '=' can handle lists for categorical inclusion.
    - ignore_nan: If True, NaN values are ignored (treated as satisfying the condition). If False, NaN leads to a 'Fail'.
    
    Returns:
    - DataFrame with an additional 'QC' column.
    """
    # Start with an array that marks all as Pass
    valid_mask = np.ones(len(df), dtype=bool)
    nan_mask = np.zeros(len(df), dtype=bool)
    
    for column, (condition, threshold) in criteria.items():
        series = df[column]
        if condition == '>':
            mask = series > threshold
        elif condition == '<':
            mask = series < threshold
        elif condition == '=':
            if isinstance(threshold, list):
                mask = series.isin(threshold)
            else:
                mask = series == threshold
        else:
            raise ValueError(f"Unsupported condition '{condition}'.")

        # Combine the current column's valid mask with the overall mask
        if ignore_nan:
            # Ignore NaN values by setting mask to True where series is NaN
            nan_mask = nan_mask | series.isna()
            mask = mask | series.isna() 
        valid_mask &= mask
    
    if ignore_nan:
        valid_mask = valid_mask & ~nan_mask

    # Set 'QC' column based on the valid_mask
    df['QC'] = np.where(valid_mask, 'Pass', 'Fail')
    
    return df

# Define criteria as dictionary where the key is the column name and the value is a tuple (condition, threshold)

criteria_pyr = {
    #'LR': ('=', ["Right","Left"]), # use 'LR':('=',"Left"), if you want just one side
    'zpeak Amp': ('>', 0),
    'zpeak Yvalue': ('>', 0),
    'zpeak AUC': ('>', 0),
    'zs AUC': ('>', 0),
    'dffpeak Hz': ('>', 0),
    'dffpeak Amp': ('>', 0),
    'dffpeak Yvalue': ('>', 0),
    'dffpeak Risetime': ('>', 0),
    'dffpeak Duration': ('>', 0),
    'dffpeak AUC': ('>', 0),
    'dff AUC': ('>', 0),
    'noise size': ('<', 1),
    'F corrected': ('>', 40),
    'F': ('>', 80),
    'Fneu': ('>', 0),
    'Avg dF/F': ('>', 0),
    'deconv Amp': ('>', 0),
    'deconv Peaks': ('>', 0),
}

criteria_e2 = {
    #'LR': ('=', ["Right","Left"]), # use 'LR':('=',"Left"), if you want just one side
    'zpeak Amp': ('>', 0),
    'zpeak Yvalue': ('>', 0),
    'zpeak AUC': ('>', 0),
    'zs AUC': ('>', 0),
    'dffpeak Hz': ('>', 0),
    'dffpeak Amp': ('>', 0),
    'dffpeak Yvalue': ('>', 0),
    'dffpeak Risetime': ('>', 0),
    'dffpeak Duration': ('>', 0),
    'dffpeak AUC': ('>', 0),
    'dff AUC': ('>', 0),
    'noise size': ('<', 1),
    'F corrected': ('>', 40),
    'F': ('>', 80),
    'Fneu': ('>', 0),
    'Avg dF/F': ('>', 0),
    'deconv Amp': ('>', 0),
    'deconv Peaks': ('>', 0),
}


Pyr_df = QC_ROI(Pyr_df, criteria_pyr, ignore_nan=True)
E2_df = QC_ROI(E2_df, criteria_e2, ignore_nan=True)

# Pyr_df['Flag']=""

print(f"Number of valid pyr: {(Pyr_df['QC'] == 'Pass').sum()}")
print(f"Number of valid e2 cells: {(E2_df['QC'] == 'Pass').sum()}")


#%% MultiFunctionviewer setup to plot pyramids
# plot z-scored trace and smoothed trace in upper window
data1 = viewer_dataset(zs_df.transpose(), mode='line', label='z df/f')
data3 = viewer_dataset(sm_z_scored_df.transpose(), mode='line', label='smoothed z_df/f')
data2 = viewer_dataset(mode='scatter', label='peaks', x=x_adj_dff, y=y_peaks)
datasets1 = [data1,data2,data3]

# plot dff trace and deconvolved data in lower window
data4 = viewer_dataset(dff_df.transpose(), mode='line', label='df/f')
data6 = viewer_dataset(deconvolved_df.transpose(), mode='line', label='deconvolved_df/f')
data5 = viewer_dataset(mode='scatter', label='peaks', x=x_adj_dff, y=y_peaks_dff)
datasets2 = [data4,data5,data6]

"""You can manually check all traces and mark the bad traces with flag"""
flagged_pyr = launch_multifunction_viewer(datasets1, datasets2, constants_df=Pyr_df)

if len(flagged_pyr) > 0: #adds a column to final df with Flag rows
    Pyr_df['Flag'] = ''
    Pyr_df.loc[flagged_pyr, 'Flag'] = 'Flag'
    print("Pyr Flagged indices:", flagged_pyr)
    
    Pyr_df.loc[Pyr_df['Flag'] == 'Flag', 'QC'] = 'Fail'

#%% MultiFunctionviewer setup to plot e2
# plot z-scored trace and smoothed trace in upper window
data1 = viewer_dataset(zs_e2.transpose(), mode='line', label='z df/f')
data3 = viewer_dataset(sm_z_scored_e2.transpose(), mode='line', label='filtered z_df/f')
data2 = viewer_dataset(mode='scatter', label='peaks', x=x_adj_dff_e2, y=y_peaks_e2)
datasets1 = [data1,data2,data3]

# plot dff trace and deconvolved data in lower window
data4 = viewer_dataset(dff_e2.transpose(), mode='line', label='df/f')
data6 = viewer_dataset(deconvolved_e2.transpose(), mode='line', label='deconvolved_df/f')
data5 = viewer_dataset(mode='scatter', label='peaks', x=x_adj_dff_e2, y=y_peaks_dff_e2)
datasets2 = [data4,data5,data6]

flagged_e2 = launch_multifunction_viewer(datasets1, datasets2, constants_df=E2_df)
if len(flagged_e2) > 0:
    E2_df['Flag'] = ''
    E2_df.loc[flagged_e2, 'Flag'] = 'Flag'
    print("E2 Flagged indices:", flagged_e2)
    
    E2_df.loc[E2_df['Flag'] == 'Flag', 'QC'] = 'Fail'

#%% exclude cell failing to QC

ind = Pyr_df[(Pyr_df['QC'] == 'Fail')].index
mask = np.ones(len(Pyr_df), dtype=bool)
mask[ind]=False

Pyr_df = Pyr_df[mask]
df_raw = df_raw[mask]
dff_df = dff_df[mask]
sm_dff_df = sm_dff_df[mask]
zs_df = zs_df[mask]
sm_z_scored_df = sm_z_scored_df[mask]
deconvolved_df = deconvolved_df[mask]

for idx in sorted(ind, reverse=True):
    del Pyr_LR_all[idx]
    del x_adj_dff[idx]
    del peak_amplitudes_dff[idx]
    del peak_amplitudes[idx]
    del onset_indice_dff_sm[idx]
    del onset_adj_dff[idx]
    del tail_adj_dff[idx]
    del z_baseline_df[idx]

if 'Flag' in Pyr_df.columns:
    Pyr_df = Pyr_df.drop('Flag', axis=1)

ind = E2_df[(E2_df['QC'] == 'Fail')].index
mask = np.ones(len(E2_df), dtype=bool)
mask[ind]=False

E2_df = E2_df[mask]
df_raw_e2 = df_raw_e2[mask]
dff_e2 = dff_e2[mask]
sm_dff_e2 = sm_dff_e2[mask]
zs_e2 = zs_e2[mask]
sm_z_scored_e2 = sm_z_scored_e2[mask]
deconvolved_e2 = deconvolved_e2[mask]

for idx in sorted(ind, reverse=True):
    del E2_LR_all[idx]
    del x_adj_dff_e2[idx]
    del peak_amplitudes_dff_e2[idx]
    del peak_amplitudes_e2[idx]
    del onset_indice_dff_sm_e2[idx]
    del onset_adj_dff_e2[idx]
    del tail_adj_dff_e2[idx]
    del z_baseline_e2[idx]

if 'Flag' in E2_df.columns:
    E2_df = E2_df.drop('Flag', axis=1)

#%%  calculate pearson correlation
"""df/f trace is smoothed first, then claculate the Pearson correlations b/w cell pairs. Distance b/w cell pair is measured for checking the relationship of distance"""
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from scipy.ndimage import uniform_filter1d

def filter_and_smooth(df, window_size=5):
    smoothed_df = df.apply(lambda x: uniform_filter1d(x, size=window_size), axis=1)
    return smoothed_df

def calculate_correlation_and_distances_1(df, coordinates):
    n = len(df)
    corr_matrix = np.zeros((n, n))
    distance_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):  # Avoid duplicate calculations
            corr, _ = pearsonr(df.iloc[i], df.iloc[j])
            dist = euclidean(coordinates[i], coordinates[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Symmetric matrix
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix
    return corr_matrix, distance_matrix

def calculate_correlation_and_distances_2(df, df2, coordinates, coordinates2):
    n1 = len(df)
    n2 = len(df2)
    corr_matrix = np.zeros((n1, n2))
    distance_matrix = np.zeros((n1, n2))
    for i in range(0, n1):
        for j in range(0, n2):
            corr, _ = pearsonr(df.iloc[i], df2.iloc[j])
            dist = euclidean(coordinates[i], coordinates2[j])
            corr_matrix[i, j] = corr
            distance_matrix[i, j] = dist
    return corr_matrix, distance_matrix

def correlation_and_distances(df, df2, coordinates, coordinates2, n1, n2):
    all_correlations = []
    all_distances = []
    if df.equals(df2):
        corr_matrix, dist_matrix = calculate_correlation_and_distances_1(df, coordinates)
        correlations = corr_matrix[np.triu_indices_from(corr_matrix, 1)]
        distances = dist_matrix[np.triu_indices_from(dist_matrix, 1)]
    else:
        corr_matrix, dist_matrix = calculate_correlation_and_distances_2(df, df2, coordinates, coordinates2)
        correlations = corr_matrix.flatten()
        distances = dist_matrix.flatten()
    all_correlations.extend(correlations)
    all_distances.extend(distances)

    return np.mean(all_correlations), all_correlations, all_distances

def analyze_correlations_with_bootstrap(sm_df1, sm_df2, deconvolved_df1, deconvolved_df2, df_LR1, df_LR2):
    coordinates1 = [xy for xy, _ in df_LR1]
    coordinates2 = [xy for xy, _ in df_LR2]
    results = {}
    for df_name, df1, df2 in [("DFF", sm_df1, sm_df2), ("Deconvolved", deconvolved_df1, deconvolved_df2)]:
        smoothed_df1 = filter_and_smooth(df1)
        smoothed_df2 = filter_and_smooth(df2)
        n1 = len(sm_df1)
        n2 = len(sm_df2)
        mean_corr, correlations, distances = correlation_and_distances(smoothed_df1, smoothed_df2,  coordinates1, coordinates2, n1, n2)
        results[f"{df_name}"] = (mean_corr, correlations, distances)
            
    return results

corr_results_Pyr = analyze_correlations_with_bootstrap(sm_dff_df, sm_dff_df, deconvolved_df, deconvolved_df, Pyr_LR_all, Pyr_LR_all)
corr_results_E2 = analyze_correlations_with_bootstrap(sm_dff_e2, sm_dff_e2, deconvolved_e2, deconvolved_e2, E2_LR_all, E2_LR_all)
corr_results_PE = analyze_correlations_with_bootstrap(sm_dff_df, sm_dff_e2, deconvolved_df, deconvolved_e2, Pyr_LR_all, E2_LR_all)

#%% calculate E2-Pyr network
def E2_network_changes(df, index_lists, value_lists, threshold, window):
    """
    Calculate the average changes in dataframe values around specified indices
    where corresponding values exceed a given threshold.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        index_lists (list of arrays): Each array contains indices around which the average change is calculated.
        value_lists (list of arrays): Each array contains values corresponding to the indices in `index_lists`.
        threshold (float): Value threshold above which indices are considered for change calculation.
        window (int): Number of points before and after the index to consider for averaging.
    
    Returns:
        pd.DataFrame: DataFrame where each column corresponds to the average changes calculated for each array in `index_lists`.
    """
    changes = []

    # Iterate over each pair of index list and corresponding value list
    for indices, values in zip(index_lists, value_lists):
        row_changes = []  # This will store the average change for each row computed from current indices array
        
        # Filter indices by value threshold
        filtered_indices = [idx for idx, val in zip(indices, values) if val > threshold]
        
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            individual_changes = []  # Store changes for current row for current filtered indices array
            
            # Calculate changes for each index in the filtered indices array for the current row
            for index in filtered_indices:
                if index < window or index + window >= len(row):
                    continue  # Skip indices that are too close to the start or end of the row to form a full window
                
                # Calculate pre-index and post-index averages
                pre_avg = np.mean(row[index-window:index])
                post_avg = np.mean(row[index+1:index+window+1])
                
                # Calculate change and store it
                change = (post_avg - pre_avg) / abs(pre_avg)
                individual_changes.append(change)
            
            # Store the average of all changes for the current row (if there are any valid changes)
            if individual_changes:
                row_changes.append(np.mean(individual_changes))
            else:
                row_changes.append(np.nan)  # Use NaN if no indices were valid
            
        # Store the list of row_changes as a column for the current indices array
        changes.append(row_changes)
    
    # Create a DataFrame from the changes
    changes_df = pd.DataFrame(changes).T  # Transpose to make each list of changes a column
    changes_df.columns = [f'Changes_{i+1}' for i in range(len(index_lists))]
    
    return changes_df

# measure E2 effect based on df/f trace
E2_pyr_network = E2_network_changes(zs_df, x_adj_dff_e2, peak_amplitudes_e2, 7.5, window=10) # 50% of peak > 7.5, and mean risetime of e2 peak is ~10

def euclidean_distance(xy1, xy2):
    # Ensure xy1 and xy2 are numpy arrays of the correct shape (n, 2)
    arr1 = np.atleast_2d(np.array(xy1))
    arr2 = np.atleast_2d(np.array(xy2))

    # Expand dims to broadcast and compute pairwise distances
    diffs = np.expand_dims(arr1, axis=1) - np.expand_dims(arr2, axis=0)
    dists = np.sqrt(np.sum(np.square(diffs), axis=2))

    # Return the mean of the pairwise distances
    return np.mean(dists)

def calculate_distances(E2_LR, Pyr_LR):
    """
    Calculate the Euclidean distances between coordinate sets in E2_LR and Pyr_LR.
    """
    distances = []

    # Iterate through each element in E2_LR
    for (xy_e2, _) in E2_LR:
        current_distances = []

        # Compute distance from current E2 element to all Pyr elements
        for (xy_pyr, _) in Pyr_LR:
            dist = euclidean_distance(xy_e2, xy_pyr)
            current_distances.append(dist)

        distances.append(current_distances)

    # Create a DataFrame from the list of distances
    distance_df = pd.DataFrame(distances).T
    distance_df.columns = [f'E2_{i+1}' for i in range(len(E2_LR))]

    return distance_df

E2_pyr_distance = calculate_distances(E2_LR_all, Pyr_LR_all)

E2_pyr_df = pd.DataFrame()
for i in range(E2_pyr_network.shape[1]):
    df_temp = pd.DataFrame()
    df_temp["cell"]= [i]*E2_pyr_network.shape[0]
    df_temp["change"]= E2_pyr_network.iloc[:,i]
    df_temp["distance"]= E2_pyr_distance.iloc[:,i]
    
    E2_pyr_df = pd.concat([E2_pyr_df,df_temp], axis=0)

#%% add coordinate of cell into sum
coor_x_df=[]
coor_y_df=[]
for i in range(len(Pyr_LR_all)):
    coor_x_df.append(Pyr_LR_all[i][0][0])
    coor_y_df.append(Pyr_LR_all[i][0][1])
Pyr_df['coor_x'] = coor_x_df
Pyr_df['coor_y'] = coor_y_df

coor_x_e2=[]
coor_y_e2=[]
for i in range(len(E2_LR_all)):
    coor_x_e2.append(E2_LR_all[i][0][0])
    coor_y_e2.append(E2_LR_all[i][0][1])
E2_df['coor_x'] = coor_x_e2
E2_df['coor_y'] = coor_y_e2

#%% convert corr_result into df
df_corr_Pyr = pd.DataFrame()
df_corr_E2 = pd.DataFrame()
df_corr_PE = pd.DataFrame()

for i in ("DFF","Deconvolved"):
    df_corr_Pyr[f"{i}_corr"] = corr_results_Pyr[i][1]
    df_corr_Pyr[f"{i}_dist"] = corr_results_Pyr[i][2]

for i in ("DFF","Deconvolved"):
    df_corr_E2[f"{i}_corr"] = corr_results_E2[i][1]
    df_corr_E2[f"{i}_dist"] = corr_results_E2[i][2]

for i in ("DFF","Deconvolved"):
    df_corr_PE[f"{i}_corr"] = corr_results_PE[i][1]
    df_corr_PE[f"{i}_dist"] = corr_results_PE[i][2]
    
#%% convert list of array into df
x_adj_list = [x_adj_dff, x_adj_dff_e2]
onset_adj_list = [onset_adj_dff, onset_adj_dff_e2]
tail_adj_list = [tail_adj_dff, tail_adj_dff_e2]

for index, i in enumerate(x_adj_list):
    max_length = max(len(arr) for arr in i)
    data_temp = [pd.Series(arr).reindex(range(max_length)).values for arr in i]
    x_adj_list[index] = pd.DataFrame(data_temp)

for index, i in enumerate(onset_adj_list):
    max_length = max(len(arr) for arr in i)
    data_temp = [pd.Series(arr).reindex(range(max_length)).values for arr in i]
    onset_adj_list[index] = pd.DataFrame(data_temp)

for index, i in enumerate(tail_adj_list):
    max_length = max(len(arr) for arr in i)
    data_temp = [pd.Series(arr).reindex(range(max_length)).values for arr in i]
    tail_adj_list[index] = pd.DataFrame(data_temp)

#%% write measurement into HDF5 file
"""Need to define file name and output path first"""
out_path1 = r"C:\Users\xxx\date of rec\sum" # set your output directory for summary data
out_path2 = r"C:\Users\xxx\date of rec\raw" # set your output directory for raw data

file_name = f'rec_{rec}.h5'
file_path1 = f'{out_path1}/{file_name}'
file_path2 = f'{out_path2}/{file_name}'

with pd.HDFStore(file_path1, mode='w') as store:
    store.put('sum_pyr', Pyr_df)
    store.put('sum_e2', E2_df)
    store.put('corr_pyr', df_corr_Pyr)
    store.put('corr_e2', df_corr_E2)
    store.put('corr_pe', df_corr_PE)
    store.put('E2_pyr_network', E2_pyr_df)

with pd.HDFStore(file_path2, mode='w') as store:
    store.put('Fcor_pyr', df_raw)
    store.put('Fcor_e2', df_raw_e2)
    store.put('dff_pyr', dff_df)
    store.put('dff_e2', dff_e2)
    store.put('sm_dff_pyr', sm_dff_df)
    store.put('sm_dff_e2', sm_dff_e2)
    store.put('zs_pyr', zs_df)
    store.put('zs_e2', zs_e2)
    store.put('sm_zs_pyr', sm_z_scored_df)
    store.put('sm_zs_e2', sm_z_scored_e2)
    store.put('dff_peak_x_pyr', x_adj_list[0]) 
    store.put('dff_peak_x_e2', x_adj_list[1])
    store.put('dff_peak_onset_pyr', onset_adj_list[0]) 
    store.put('dff_peak_onset_e2', onset_adj_list[1])
    store.put('dff_peak_tail_pyr', tail_adj_list[0]) 
    store.put('dff_peak_tail_e2', tail_adj_list[1])

print('!!! Analysis complete! Data is saved !!!')
