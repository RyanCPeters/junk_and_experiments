# Note that by convention, comments using the # will typically precede the line(s) of code they are
# describing. Whereas the """multi-line comment syntax""" typically follows the line(s) of code
# that they are describing.
import cv2
"""
The cv2 library will be used to express our arrays of fractal data as actual graphical images.

    cv2.namedWindow(window_name:str, cv2.WINDOW_<type enum>)
        creates a named window, initially without any output image data, that we can later use
        as a point for outputting a sequence of images.
    cv2.windowResize(window_name:str, width:int, height:int)
    cv2.imshow(window_name:str,img_array:np.ndarray)
    
"""

import numpy as np
"""
The numpy library provides highly optimized c code for manipulating the arrays of fractal data we
will be computing. The numpy library interacts very well with the cv2 library and will serve as the
underlying data matrices that the cv2 library will show in the output windows.

    np.zeros(size_tuple:tuple, data_type:np.dtype)
        IMPORTANT: numpy arrays specify the dimensions of an array as height-width,
                   whereas cv2 specifies array dimensions in width-height.

Recall that all axis of our numpy arrays use zero-based indexing. ie., a 10 element array would have
the following set of indices {0,1,2,3,4,5,6,7,8,9}.

Also note that numpy arrays allow us to perform `vectorized` operations. That is to say, we can
perform optimized modifications to an entire row, column, or even region of an array with a single
command.
    E.G.:
        # np.ones((4,4,3),dtype=np.uint8) is doing a couple of things at once here.
        # First it creates a new array with 4 rows, 4 columns, and 3 "channels".
        # Second it initializes each element in the array to be 8-bit (aka, byte) sized integers
        # with each having an initial value of 1.
        # NOTE: 8-bit integers can be used to express 2 possible value ranges.
        #       * If we use them to as signed bytes, then they can express values in the range
        #         of -128 to 127, inclusive.
        #       * if we use them as unsigned bytes, then they can express values in the range
        #         of 0 to 255, inclusive.
        example_array = np.ones((4,4,3),dtype=np.uint8)
        example_array -= 1 # this decrements the value of all elements in example_array.
        example_array[1,:,:] += 1 # this increments all elements in the second row of the array.
        example_array[:,1,:] += 2 # this increments all elements in the second column of the array.
        print(example_array)
            # note that example_array is a 3-dimensional array, where the third dimension is
            # 3 elements deep for all (row,col) locations.
            # In the context of image data, this is how we manage our rgb color channels.
            # printed representation of `example_array`
                      # row , column
            [[[0 0 0] # 0   , 0
              [2 2 2] # 0   , 1
              [0 0 0] # 0   , 2
              [0 0 0]]# 0   , 3
             [[1 1 1] # 1   , 0
              [3 3 3] # 1   , 1
              [1 1 1] # 1   , 2
              [1 1 1]]# 1   , 3
             [[0 0 0] # 2   , 0
              [2 2 2] # 2   , 1
              [0 0 0] # 2   , 2
              [0 0 0]]# 2   , 3
             [[0 0 0] # 3   , 0
              [2 2 2] # 3   , 1
              [0 0 0] # 3   , 2
              [0 0 0]]]#3   , 3

"""
import concurrent.futures as cf
"""
This library is how we are managing the process parallelism in the program. It provides a super easy
api interface for managing multi-processing and multi-threading.

We will use the process/thread pool classes. A nice aspect of these classes is that they implement
the python 3 context idiom, which handles resource allocation and release for us. All we have to do
is use the `with cf.<pool class>() as name:`
"""
import os
"""
The os library is a built-in set of functions that streamline interactions with the operating system.
In this program we will call the os.cpu_count() function to get the number of available cpu's in the
machine executing this code.
"""
import numba
"""
numba is a third party library that lets us optimize our computations by performing
just in time (jit) compilation of annotated sections of our code.
"""
import PySimpleGUI as sg
"""
PySimpleGUI is a third party library provides easy to use api interfaces for producing simple gui
elements.

Specifically, we are using it to produce the "Stop Animation" button window.
"""
from PIL import Image, ImageTk
"""
PIL is a third-party image manipulation library that most GUI libraries are configured to work with.
We need it here to configure our image data so that the PySimpleGUI library can easily interface
our image data with their own gui tool-chain on the back-end.
"""
import io
"""
The io library is a built-in library, like the os library above, and we'll be using it to convert
our slightly abstract image data arrays into literal byte data when we are first booting up our gui
elements.

Specifically, the PySimpleGUI is using TKinter to handle the process of integrating image data in
the gui, and TKinter requires that we configure the first image passed into the tool-chain as a
byte data, rather than the more abstract numpy arrays we are working with in our computations.
"""

# cX, cY = -0.7, 0.27015
cX, cY = -0.7, 0.265

# cX, cY = -1., 0.
# cX, cY = .6, 0.55

maxIter = 255
scale = 500 # this is where you change the resolution of the output
w, h = scale,scale# int(scale*(1080/1920))
proc_num = max(os.cpu_count(), 1)
region_size = h//proc_num
@numba.njit()
def degree_4(zx:float,zy:float):
    i = 0
    while zx+zy<2 and i<maxIter:
        zxzx = zx*zx*zx*zx
        zyzy = zy*zy*zy*zy
        zy, zx = zyzy+cY, zxzx+cX
        i += 1
    return i

def alt_degree_2(zx:float,zy:float):
    i = 0
    while zx+zy<2 and i<maxIter:
        zxzx = zx*zx
        zyzy = zy*zy
        zy, zx = zyzy+cY, zxzx+cX
        i += 1
    return i

@numba.njit()
def degree_2(zx:float,zy:float):
    i = maxIter
    while zx+zy<2 and i>1:
        zxzx = zx*zx
        zyzy = zy*zy
        zy, zx = 2.0*zx*zy+cY, zxzx-zyzy+cX
        i -= 1
    return i

def single_row(part_id:int,zoom:float,moveX:float,moveY:float):
    img = np.zeros((region_size, w, 3), dtype=np.uint8)
    y_denom = (0.35*zoom*h)
    x_denom = (0.35*zoom*w)
    part_h = (part_id*region_size)
    x_offset = -w/2
    y_offset = part_h-h/2
    
    for x in range(w):
        _zx = (x+x_offset)/x_denom+moveX
        for y in range(region_size):
            zx = _zx
            zy = (y+y_offset)/y_denom+moveY
            img[y, x] += degree_2(zx,zy)
    return part_id, cv2.applyColorMap(img, cv2.COLORMAP_JET)

def main():
    # moveX, moveY = 0.01119713333, 0.0094754444444445
    moveX, moveY = 0.0143425224525, -0.05074481534
    zoom = 1.
    base = np.zeros((h, w, 3), dtype=np.uint8)
    with cf.ProcessPoolExecutor(proc_num) as ppe:
        # this initial setup is to establish the base
        ftrs = [ppe.submit(single_row,part,zoom,moveX,moveY) for part in range(proc_num)]
        for ftr in cf.as_completed(ftrs):
            part_id, img = ftr.result()
            base[part_id*region_size:(part_id+1)*region_size] += img
        # to display the created fractal
        imgs = [base]
        cv2.namedWindow("bitmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("bitmap", 700, int(700*(1080/1920)))
        cv2.imshow("bitmap", imgs[-1])
        cv2.waitKey(0)
        cv2.imwrite("./fractal_base.png",imgs[-1])
        for _ in range(200):
            zoom *= 1.3
            imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
            ftrs = [ppe.submit(single_row, part,zoom,moveX,moveY)
                    for part in range(proc_num)]
            for ftr in cf.as_completed(ftrs):
                part_id, img = ftr.result()
                imgs[-1][part_id*region_size:(part_id+1)*region_size] += img
            cv2.imshow("bitmap",imgs[-1])
            cv2.waitKey(1)
        win = sg.Window(title="Animation exit control",size=(len("Animation exit control")*5,len("Animation exit control")))
        win.AddRow(
            sg.Button(
                    'Stop Animation',
                    button_color=None,
                    focus=True,
                    bind_return_key=True,
                    auto_size_button=True,
                    key='stop'))
        idx = 0
        button,values = win.Read(timeout=0)
        while button != 'stop' and button is not None:
            cv2.imshow("bitmap",imgs[idx])
            cv2.waitKey(20)
            idx = (idx+1)%len(imgs)
            button,values = win.Read(timeout=0)
    cv2.destroyAllWindows()
    
def gui():
    # Get the folder containin:g the images from the user
    folder = sg.popup_get_folder('Image folder to open', default_path='')
    if not folder:
        sg.popup_cancel('Cancelling')
        raise SystemExit()
    
    # PIL supported image types
    img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
    
    # get list of files in folder
    flist0 = os.listdir(folder)
    
    # create sub list of image files (no sub folders, no wrong file types)
    fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
    
    num_files = len(fnames)  # number of iamges found
    if num_files==0:
        sg.popup('No files in folder')
        raise SystemExit()
    
    del flist0  # no longer needed
    
    
    # ------------------------------------------------------------------------------
    # use PIL to read data of one image
    # ------------------------------------------------------------------------------
    
    def get_img_data(f, maxsize=(700, int(700*(1080/1920))), first=False):
        """Generate image data using PIL
        """
        img = Image.open(f)
        img.thumbnail(maxsize)
        if first:  # tkinter is inactive the first time
            with io.BytesIO() as bio:
                img.save(bio, format="PNG")
                return bio.getvalue()
        return ImageTk.PhotoImage(img)
    
    
    # ------------------------------------------------------------------------------
    
    # make these 2 elements outside the layout as we want to "update" them later
    # initialize to the first file in the list
    filename = os.path.join(folder, fnames[0])  # name of first file in list
    image_elem = sg.Image(data=get_img_data(filename, first=True))
    filename_display_elem = sg.Text(filename, size=(80, 3))
    file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))
    
    # define layout, show and read the form
    col = [[filename_display_elem],
           [image_elem]]
    
    col_files = [[sg.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox')],
                 [sg.Button('Next', size=(8, 2)), sg.Button('Prev', size=(8, 2)),
                  file_num_display_elem]]
    
    layout = [[sg.Column(col_files), sg.Column(col)]]
    
    window = sg.Window('Image Browser', layout, return_keyboard_events=True,
                       location=(0, 0), use_default_focus=False)
    # loop reading the user input and displaying image, filename
    i = 0
    while True:
        # read the form
        event, values = window.read()
        print(event, values)
        # perform button and keyboard operations
        if event is None:
            break
        elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
            i += 1
            if i>=num_files:
                i -= num_files
            filename = os.path.join(folder, fnames[i])
        elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
            i -= 1
            if i<0:
                i = num_files+i
            filename = os.path.join(folder, fnames[i])
        elif event=='listbox':  # something from the listbox
            f = values["listbox"][0]  # selected filename
            filename = os.path.join(folder, f)  # read this file
            i = fnames.index(f)  # update running index
        else:
            filename = os.path.join(folder, fnames[i])
        
        # update window with new image
        image_elem.update(data=get_img_data(filename, first=True))
        # update window with filename
        filename_display_elem.update(filename)
        # update page display
        file_num_display_elem.update('File {} of {}'.format(i+1, num_files))
    
    window.close()
    
if __name__ == '__main__':
    main()
