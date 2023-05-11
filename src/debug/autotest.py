from __future__ import absolute_import
from data.REDS_loader import getDataLoader  
from model.generator import Generator
import torch
import cv2
import albumentations as A

OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"

def autotest(conf):
    tot = 0
    passed = 0

    try:
        data_loader = getDataLoader(conf)
        sample = next(iter(data_loader))

        # "x" is images/referencePath  x  element in the list of neighbors x batch element
        if conf['DEFAULT'].getboolean("debugging"):
            cv2.imshow("BLUR_"+sample["referencePath"][0], sample["x"][len(sample["x"])//2][0].numpy())
            cv2.waitKey()
            cv2.imshow("SHARP_"+sample["referencePath"][0], sample["y"][0].numpy())
            cv2.waitKey()

        print("[TEST] Dataset loading... "+OK)
        passed+=1
    except Exception as e: 
        print("[TEST] Dataset loading... "+NO)
        print(e)

    tot+=1

    try:
        s=torch.stack(sample["x"],dim=0)  #need normalization because of the too large size->program crashes
        s=s.to(torch.float32)
        s=s.permute(1,0,4,2,3)
        g = Generator(conf)
        #g = Generator(num_frames=num_frames,num_extr_blocks=1,num_ch_in=3,num_features=16)
        y=g(s)
        if conf['DEFAULT'].getboolean("debugging"):
            print(f"generator output shape = {y.shape}")
        print("[TEST] Generator flow... "+OK)
        passed+=1
    except Exception as e:
        print("[TEST] Generator flow... "+NO)
        print(e)

    tot+=1 

    print(f"[TEST] {passed}/{tot} tests passed")
