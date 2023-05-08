from __future__ import absolute_import
from data.sharp_loader import SharpLoader  
from model import generator
import torch
import cv2

OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"

def autotest(conf):
    tot = 0
    passed = 0

    try:
        sl = SharpLoader(conf)

        data_loader = torch.utils.data.DataLoader(
            sl,
            batch_size=8
        )

        sample = next(iter(data_loader))

        # sample is images/referencePath  x  element in the list of neighbors x batch element
        if conf['DEFAULT'].getboolean("debugging"):
            cv2.imshow(sample["referencePath"][0], sample["images"][len(sample["images"])//2][0].numpy())
            cv2.waitKey()

        print("[TEST] Dataset loading... "+OK)
        passed+=1
    except Exception as e: 
        print("[TEST] Dataset loading... "+NO)
        print(e)

    tot+=1

    try:
        s=torch.stack(sample["images"],dim=0)
        s=s.to(torch.float32)
        s=s.permute(1,0,4,2,3)
        print(s.shape)
        g= generator.Generator(8,1,3,64)
        y=g(s)
        print(y.shape)
        print("[TEST] Generator flow... "+OK)
        passed+=1
    except Exception as e:
        print("[TEST] Generator flow... "+NO)
        print(e)

    tot+=1 

    print(f"[TEST] {passed}/{tot} tests passed")
