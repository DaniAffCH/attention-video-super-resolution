from __future__ import absolute_import
from data.REDS_loader import REDS_loader  
from model import generator
import torch
import cv2

OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"

def autotest(conf):
    tot = 0
    passed = 0

    try:
        rl = REDS_loader(conf)

        data_loader = torch.utils.data.DataLoader(
            rl,
            batch_size=conf['DEFAULT'].getint("batch_size")
        )

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
        print(s.shape)
        g= generator.Generator(4,1,3,8)
        y=g(s)
        print(y.shape)
        print("[TEST] Generator flow... "+OK)
        passed+=1
    except Exception as e:
        print("[TEST] Generator flow... "+NO)
        print(e)

    tot+=1 

    print(f"[TEST] {passed}/{tot} tests passed")
