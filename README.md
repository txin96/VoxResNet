# VoxResNet
This is an unofficial implementation of [VoxResNet: Deep Voxelwise Residual Networks for Volumetric Brain Segmentation](https://arxiv.org/pdf/1608.05895.pdf).

# Requirements
- python 3.6
- TensorFlow
- TensorLayer
- NiBabel
- Numpy
- SciPy

# Data Setup
1. Download ADNI dataset from http://adni.loni.usc.edu. (or other dataset you want)
2. Modify file path and run preprocess.py.

# Training
1. Follow steps in the Data Setup
2. Run train.py

# Testing
1. Follow steps in the Data Setup
2. Run segment.py, and the outputs are saved to save_path
3. Run dice.py, dice is defined as:
<a href="https://www.codecogs.com/eqnedit.php?latex=Dice(A,&space;B)&space;=&space;\frac{2|A&space;\cap&space;B|}{|A|&plus;|B|}" align = center target="_blank"><div align=center><img src="https://latex.codecogs.com/gif.latex?Dice(A,&space;B)&space;=&space;\frac{2|A&space;\cap&space;B|}{|A|&plus;|B|}" title="Dice(A, B) = \frac{2|A \cap B|}{|A|+|B|}" /></div></a>
