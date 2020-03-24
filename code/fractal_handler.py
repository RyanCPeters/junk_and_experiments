import cv2
import numpy as np
import concurrent.futures as cf
import os
# import numba
import PySimpleGUI as sg
# import pickle
# import bz2

cX, cY = -0.7, 0.27015  # 0.27015
moveX, moveY = 0.0, 0.0
maxIter = 255
scale = 500
w, h = scale,scale# int(scale*(1080/1920))
proc_num = max(os.cpu_count(), 1)
region_size = h//proc_num


def single_row(part_id:int,zoom:float):
    img = np.zeros((region_size, w, 3), dtype=np.uint8)
    y_denom = (0.5*zoom*h)+moveY
    x_denom = (0.5*zoom*w)+moveX
    x_offset = -w/2
    y_offset = -h/2
    part_h = (part_id*region_size)
    for x in range(w):
        _zx = 1.5*(x+x_offset)/x_denom
        for y in range(region_size):
            zx = _zx
            zy = 1.0*(y+part_h+y_offset)/y_denom
            i = 0
            zxzx = zx*zx
            zyzy = zy*zy
            while zxzx+zyzy<4 and i<maxIter:
                zy, zx = 2.0*zx*zy+cY, zxzx-zyzy+cX
                zxzx = zx*zx
                zyzy = zy*zy
                i += 1
            img[y, x] += i
    return part_id,img

def main():
    zoom = 1.
    base = np.zeros((h, w, 3), dtype=np.uint8)
    imgs = []
    with cf.ProcessPoolExecutor(proc_num) as ppe:
        # this initial setup is to establish the base
        ftrs = [ppe.submit(single_row,part,zoom)
                   for part in range(proc_num)]
        for ftr in cf.as_completed(ftrs):
            part_id, img = ftr.result()
            base[part_id*region_size:(part_id+1)*region_size] += img
        # to display the created fractal
        bitmap = cv2.applyColorMap(base, cv2.COLORMAP_JET)
        imgs.append(bitmap.copy())
        cv2.namedWindow("base", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("base", 700, int(700*(1080/1920)))
        cv2.imshow("base", bitmap)
        cv2.waitKey(0)
        cv2.namedWindow("bitmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("bitmap", 700, int(700*(1080/1920)))
        cv2.waitKey(1)
        loops = 0
        while loops<100:
            loops += 1
            zoom += 1.
            # moveX += 1.
            bitmap = np.zeros((h, w, 3), dtype=np.uint8)
            ftrs = [ppe.submit(single_row, part,zoom) for part in range(proc_num)]
            for ftr in cf.as_completed(ftrs):
                part_id, img = ftr.result()
                bitmap[part_id*region_size:(part_id+1)*region_size] += img
            imgs.append(cv2.applyColorMap(bitmap.copy(), cv2.COLORMAP_JET))
            cv2.imshow("bitmap",imgs[-1])
            cv2.waitKey(1)
        win = sg.Window(title="Animation exit control")
        win.AddRow(
            sg.Button(
                    'Stop Animation',
                    button_color=None,
                    focus=True,
                    bind_return_key=True,
                    pad=((20, 5), 3),
                    size=(5, 1),
                    key='stop'))
        idx = 0
        button,values = win.Read(timeout=0)
        while button != 'stop' and button != None:
            cv2.imshow("bitmap",imgs[idx])
            cv2.waitKey(1)
            idx = (idx+1)%len(imgs)
            button,values = win.Read(timeout=0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()
