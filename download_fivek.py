import os
import sys
from urllib.request import urlretrieve
from tqdm import tqdm
from PIL import Image
import rawpy
import imageio

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

img_lst.sort()
# img_lst1 = img_lst[:len(img_lst)//2]
# img_lst2 = img_lst[len(img_lst)//2:]
# img_lst = img_lst2

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
    if os.path.exists(fname + '.dng'):
        print(f'{fname}.dng already exists')
        continue
    URL='https://data.csail.mit.edu/graphics/fivek/img/dng/'+fname+'.dng' # Download the image adjusted by C (the other four types of images can be downloaded as needed)
    print('Downloading ' + fname + ':')
    try:
        urlretrieve(URL, fname+'.dng', cbk) # Store the acquired pictures in a local address
    except Exception as e:
        print(f'{fname}: {e} occured.')
        not_dl.append(fname)
        continue

print()
for nd in not_dl:
    print(nd+'.dng')

print('Convert dng to jpg')
out_dir = 'Jpeg_input_original'
os.makedirs(out_dir, exist_ok=True)

for fname in tqdm(img_lst):
    out_name = os.path.join(out_dir, fname+'.jpg')
    if os.path.exists(out_name): continue
    try:
        with rawpy.imread(fname+'.dng') as raw:
            # こうすれば元画像と同じような画像になる
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            imageio.imsave(out_name, rgb)
    except Exception as e:
        print(f'{e} ouuerd in processing {fname}.dng')
        continue

print()
for nd in not_dl:
    print(nd+'.dng')