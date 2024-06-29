# Testing the compliance with FMPy API: https://github.com/CATIA-Systems/FMPy

from dummy_FMUEnv import get_dummy_FMUEnv
import os
import numpy as np

# create FMUEnv object from manually validated dummy environment
fmugym_test = get_dummy_FMUEnv()


def test_get_fmu_output():
    outputs = fmugym_test._get_fmu_output()
    
    for out in outputs.values(): 
        assert isinstance(out, np.ndarray)

def test_close():
    dir_exists_before = os.path.isdir(fmugym_test.unzipdir)
    assert dir_exists_before == True

    fmugym_test.close()
    dir_exists_after = os.path.isdir(fmugym_test.unzipdir)
    assert dir_exists_after == False