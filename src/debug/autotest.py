from __future__ import absolute_import


OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"

def autotest():
    tot = 0
    passed = 0



    print(f"[TEST] {passed}/{tot} tests passed")
