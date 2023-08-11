# EPIT

This is the official implementation of "*Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution*", ICCV 2023. 

[[arXiv](https://arxiv.org/abs/2302.08058)] [[project](https://zhengyuliang24.github.io/EPIT/)]
<br>


## Training & Evaluation
* Download the EPFL, HCInew, HCIold, INRIA and STFgantry datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder **`./datasets/`**.
* Run **`Generate_Data_for_SSR_Training.py`** to generate training data, and begin to train the EPIT (on 5x5 by default) for 2x/4x SR:
```
  $ python train.py --scale_factor $2/4$
```
* Run **`Generate_Data_for_SSR_Test.py`** to generate evaluation data, and you can quick run **`test.py`** to perform network inference by using our released models.  
```
  python test.py
```
<br>


## Quantitative Results
### <img src="https://raw.github.com/ZhengyuLiang24/EPIT/main/figs/QuantitativeResults.png" width="1000">
<br>


## Visual Comparison
### <img src="https://raw.github.com/ZhengyuLiang24/EPIT/main/figs/VisualComparison.png" width="1000">
<br>


## Citiation
If you find this work helpful, please consider citing:
```
@InProceedings{Liang2023EPIT,
    author    = {Liang, Zhengyu and Wang, Yingqian and Wang, Longguang and Yang, Jungang and Zhou, Shilin and Guo, Yulan},
    title     = {Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution},
    booktitle = {ICCV}, 
    year      = {2023},   
}
```
<br>


## Related Projects
* [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)
* [NTIRE 2023 LFSR Challenge](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023)
<br>


## Contact
Welcome to raise issues or email to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn) for any questions regarding our EPIT.


