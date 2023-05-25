from __future__ import absolute_import
from data.REDS_loader import getDataLoader  
from model.generator import Generator
import torch
import cv2
from train.earlyStop import EarlyStopping
from train.loss import Loss
from train.train_one import trainOne


OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"

def autotest(conf):
    tot = 0
    passed = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    data_loader = None
    try:
        data_loader = getDataLoader(conf, "train")
        sample = next(iter(data_loader))

        # "x" is images/referencePath  x  element in the list of neighbors x batch element
        if conf['DEFAULT'].getboolean("debugging"):
            cv2.imshow("BLUR_"+sample["referencePath"][0], sample["x"][len(sample["x"])//2][0].numpy())
            cv2.waitKey()
            cv2.imshow("SHARP_"+sample["referencePath"][0], sample["y"][0].numpy())
            cv2.waitKey()
            cv2.destroyAllWindows()

        print("[TEST] Dataset loading... "+OK)
        passed+=1
    except Exception as e: 
        print("[TEST] Dataset loading... "+NO)
        print(e)

    tot+=1
    g = None
    try:
        s=torch.stack(sample["x"],dim=0)  #need normalization because of the too large size->program crashes
        s=s.permute(1,0,4,2,3).to(device)
        
        g = Generator(conf).to(device)
        y=g(s)
        y=y.cpu()
        #print(y)
        cv2.imshow("gen-test",y[0].permute(1,2,0).detach().numpy())
        cv2.waitKey()
        if conf['DEFAULT'].getboolean("debugging"):
            print(f"generator output shape = {y.shape}")
        print("[TEST] Generator flow... "+OK)
        passed+=1
        del y
    except Exception as e:
        print("[TEST] Generator flow... "+NO)
        print(e)

    tot+=1 

    try:
        optimizerGen = torch.optim.Adam(g.parameters(),lr=conf["TRAINING"].getfloat("generator_learning_rate"))
        lossfac = Loss()
        trainOne(g, data_loader, optimizerGen, device, lossfac, conf, True)

        print("[TEST] Training step... "+OK)
        passed+=1
    except Exception as e: 
        print("[TEST] Training step... "+NO)
        print(e)

    tot+=1



    print(f"[TEST] {passed}/{tot} tests passed")
