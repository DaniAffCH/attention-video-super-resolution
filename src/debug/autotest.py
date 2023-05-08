from __future__ import absolute_import
from data.sharp_loader import SharpLoader  


OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"

def autotest(conf):
    tot = 0
    passed = 0

    try:
        sl = SharpLoader(conf)
        assert(all(elem is not None for elem in sl[0]))
        print("[TEST] Dataset loading... "+OK)
        passed+=1
    except Exception as e: 
        print("[TEST] Dataset loading... "+NO)
        print(e)
    tot+=1 

    print(f"[TEST] {passed}/{tot} tests passed")
