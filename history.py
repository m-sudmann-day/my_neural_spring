import os
import sys
import shutil
import csv
from time import gmtime, strftime
from glob import glob
from pathlib import Path
from misc import *

class history():

    original_stdout = None
    original_stderr = None
    relative_history_root = "../history"

    def __init__(self, save_stdout=True, save_stderr=True):

        self.timestr = strftime("%y%m%d_%H%M%S", gmtime())
        
        if not os.path.exists(self.relative_history_root):
            os.mkdir(self.relative_history_root)

        self.output_path = os.path.join (self.relative_history_root, self.timestr)
        os.mkdir(self.output_path)
        
        if save_stdout:
            self.original_stdout = sys.stdout
            out_file = self.get_absolute_path("stdout.txt")
            sys.stdout = stream_tee(sys.stdout, open(out_file, "w+"))
        
        if save_stderr:
            self.original_stderr = sys.stderr
            err_file = self.get_absolute_path("stderr.txt")
            sys.stderr = stream_tee(sys.stderr, open(err_file, "w+"))
        
        self.copy_files_to_new_folder("./*py", "scripts")
        #self.copy_files_to_new_folder("../images/*", "images")

    def __enter__(self):
                
        return self

    def __exit__(self, type, value, traceback):
        
        self.output_path = None
        
        if self.original_stdout != None:
            sys.stdout = self.original_stdout
            self.original_stdout = None
            
        if self.original_stderr != None:
            sys.stderr = self.original_stderr
            self.original_stderr = None

    def copy_files_to_new_folder(self, source_path_relative_to_cwd, relative_dest_folder):
        
        dest_folder = self.create_folder(relative_dest_folder)
        
        for filename in glob(source_path_relative_to_cwd, recursive=False):
            shutil.copy(filename, os.path.join(dest_folder, filename))
        
    def create_folder(self, relative_path):
        
        folder = self.get_absolute_path(relative_path)
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder
        
    def get_absolute_path(self, relative_path):
        
        path = os.path.join(self.output_path, relative_path)
        return path
        
    def write_stub(self, filename):
        
        filename = self.get_absolute_path(filename)
        Path(filename).touch()
    
    def write_text_file(self, relative_path, notes):
        
        path = self.get_absolute_path(relative_path)
        with open(path, 'w+') as f:
            f.write(notes)
            
    def write_dictionary_to_csv(self, relative_path, dict):

        # Treat the dictionary as 2-dimensional array with each
        # value in the dictionary containing a column.  Assumes each
        # value/column contains the same number of items/rows.
        
        path = self.get_absolute_path(relative_path)
        keys = list(dict.keys())
        first_col = dict[keys[0]]                
        
        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(keys)
            for row in range(len(first_col)):
                writer.writerow(dict[key][row] for key in keys)
