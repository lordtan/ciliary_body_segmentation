# 分割参考

https://github.com/qubvel/segmentation_models.pytorch

## 安装依赖包

```shell
# pip安装requirement.txt
pip install -r requirements.txt

# 使用镜像安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## labelme源码修改

需要对labelme源码进行修改，不然无法读出正确的参数
```python

# 修改文件 如下 Lib\site-packages\imgviz\label.py 第43行以下

cmap = np.stack((r, g, b), axis=1).astype(np.uint8)  

''' 添加如下代码，相当于是把这个数组重新赋值了 '''
cmap[0, :] = [0,0,0]  # 黑色
cmap[1, :] = [255,0,0]  # 红色
cmap[2, :] = [0,255,0]  # 绿色
cmap[3, :] = [0,0,255]  # 蓝色
```