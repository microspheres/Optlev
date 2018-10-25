import time
import PIL.ImageGrab as ig

outdir = r"C:\data\20180129\bead3_um_POL_NS_SiO2_10um\screenshot"


for i in range(1000,1000000):
    im = ig.grab()
    im.save(outdir + "\pump_down_%d.png"%i)
    time.sleep(600)
