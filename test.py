import os
import glob
import time
list_of_files =glob.glob('./data/video/*')
oldest_vid =max(list_of_files, key=os.path.getmtime)
print(oldest_vid)
#os.remove(oldest_vid)
