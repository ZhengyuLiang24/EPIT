# EPIT

## Official implementation of "*Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution*", 2023. 
[[arXiv]()]

### <img src="https://raw.github.com/ZhengyuLiang24/EPIT/main/figs/AttributionMaps.png" width="500">
<br>


## Highlights
1. We address the importance of exploiting non-local spatial-angular correlation in LF image SR, and propose a simple yet effective method to handle this problem.
2. We develop a Transformer-based network to learn the non-local spatial-angular correlation from horizontal and vertical EPIs, and validate the effectiveness of our method through extensive experiments and visualizations. 
3. Compared to existing state-of-the-art LF image SR methods, our method achieves superior performance on public LF datasets, and is much more robust to disparity variations. 
<br><br>


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


## Perspective Comparison
### <img src="https://raw.github.com/ZhengyuLiang24/EPIT/main/figs/PerspectiveComparison.png" width="1000">
<br>



## Effectiveness for Angular SR
### <img src="https://raw.github.com/ZhengyuLiang24/EPIT/main/figs/AngularSR.png" width="1000">
<br>



## Citiation
If you find this work helpful, please consider citing:
```
@Article{EPIT,
    author    = {Liang, Zhengyu and Wang, Yingqian and Wang, Longguang and Yang, Jungang and Zhou Shilin and Guo, Yulan},
    title     = {Learning Non-Local Spatial-Angular Correlation for Light Field Image Super-Resolution},
    journal   = {arXiv preprint arXiv:}, 
    year      = {2023},   
}
```
<br>


## Related Projects
* [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)
* [NTIRE 2023 LFSR Challenge](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023)
<br>


## Contact
Welcome to raise issues or email to [zyliang@nudt.edu.cn](zyliang@nudt.edu.cn) for any question regarding our EPIT.


