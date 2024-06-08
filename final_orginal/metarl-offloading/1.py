import glob
import os
import re

# Get list of all .ckpt files
checkpoint_files = glob.glob("meta_model_inner_step1/meta_model_*.ckpt")

# Function to extract the numeric part from the filename
def extract_number(filename):
    # Extract the number from the filename
    number = re.search(r'\d+', os.path.basename(filename))
    return int(number.group()) if number else 0

# Sort the files based on the numeric part in their filenames
checkpoint_files.sort(key=extract_number)

# Now checkpoint_files is sorted by the numeric part in their filenames
print(checkpoint_files)
