import os
import sys
from urllib.request import urlretrieve
from tqdm import tqdm
from PIL import Image

# Change current working path
CURRENT_PATH="./datasets/fivek_dl" # Path of this file
os.chdir(CURRENT_PATH) # Change current path

# A list of image names
img_lst = []
# Read picture name list
with open('filesAdobe.txt', 'r') as f:
    for line in f.readlines():
        img_lst.append(line.rstrip("\n")) # Remove newlines

with open('filesAdobeMIT.txt', 'r') as f:
    for line in f.readlines():
        img_lst.append(line.rstrip("\n")) # Remove newlines

# Callback function of urlretrieve function, showing download progress
def cbk(a,b,c):
    '''Callback 
         @a: The number of downloaded data packages
         @b: The size of the data block
         @c: The size of the remote file
    '''
    per=100.0*a*b/c
    if per>100:
        per=100
    # Update progress in the terminal
    sys.stdout.write("progress: %.2f%%   \r" % (per))
    sys.stdout.flush()

# Download pictures according to the url of the file
not_dl = []
for i, fname in enumerate(img_lst):
    if os.path.exists(fname + '.tif'):
        print(f'{fname}.tif already exists')
        continue
    
    URL='https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/'+fname+'.tif' # Download the image adjusted by C (the other four types of images can be downloaded as needed)
    print('Downloading ' + fname + ':')
    try:
        urlretrieve(URL, fname + '.tif', cbk) # Store the acquired pictures in a local address
    except Exception as e:
        print(f'{fname}: {e} occured.')
        not_dl.append(fname)
        continue

print('Convert tif to jpg')
for fname in tqdm(img_lst):
    fname = fname + '.jpg'
    if os.path.exists(fname): continue
    with Image.open(i) as img:
        img = img.convert('RGB')
        img.save(fname, 'JPEG', quality=100)

print()
for nd in not_dl:
    print(nd)