import pdb

def test_function(x):
    pdb.set_trace()  # 设置断点
    return x + 1

print(test_function(10))
