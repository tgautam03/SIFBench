# SIFBench
SIFBench, a benchmark database developed to support the advancement and validation of machine learning methods for stress intensity factor (SIF) prediction. The database contains approximately 5 million unique configurations of cracks, geometries, and loading conditions, organized into 37 distinct scenarios. These scenarios are divided into single-crack and twin-crack cases, comprising roughly 1.6 million and 3.3 million geometries, respectively.


The single-crack category includes semi-elliptical surface cracks in a finite plate subjected to tensile loading, as well as various quarter-elliptical and through-thickness corner cracks located at straight and countersunk bolt holes. Most configurations are analyzed under three independent loading conditions: ***tension***, ***bending***, and ***bearing***. The twin-crack category consists of paired quarter-elliptical and through-thickness corner cracks, also positioned at straight and countersunk bolt holes and evaluated under the same three loading conditions, each applied separately.

## Directory Details
1. **files:** Data, models, predictions files. All these can be accessed from https://huggingface.co/datasets/rvn03/SIFBench/tree/main
2. **nb_analysis:** Jupyter notebooks showing the detailed dataset analysis.
3. **src:** Source files for training FNN and FNO. 

## Results
### Mean Normalized L2 Error


### Mean Normalized Absolute Error

## Reproducing Results

### Single Crack
0. **Surface Crack in a Plate:** 
    - ***Training:*** `python 00_SURFACE_CRACK.py Train`
    - ***Testing:*** `python 00_SURFACE_CRACK.py Test`
1. **Quarter-Elliptical Corner Crack at the straight hole in a Plate:** 
    - ***Training:*** `python 01_CORNER_CRACK_BH_QE.py Train`
    - ***Testing:*** `python 01_CORNER_CRACK_BH_QE.py Test`
2. **Through-Thickness Corner Crack at the straight hole in a Plate:** 
    - ***Training:*** `python 02_CORNER_CRACK_BH_Th.py Train`
    - ***Testing:*** `python 02_CORNER_CRACK_BH_Th.py Test`
3. **Quarter-Elliptical Corner Crack at the countersunk hole (with $b/t=0.75$) in a Plate:** 
    - ***Training:*** `python 03_CORNER_CRACK_CS1_QE.py Train`
    - ***Testing:*** `python 03_CORNER_CRACK_CS1_QE.py Test`
4. **Quarter-Elliptical Corner Crack at the countersunk hole (with $b/t=0.5$) in a Plate:** 
    - ***Training:*** `python 04_CORNER_CRACK_CS2_QE.py Train`
    - ***Testing:*** `python 04_CORNER_CRACK_CS2_QE.py Test`
5. **Through-Thickness Corner Crack at the countersunk hole (with $b/t=0.5$) in a Plate:** 
    - ***Training:*** `python 05_CORNER_CRACK_CS2_Th.py Train`
    - ***Testing:*** `python 05_CORNER_CRACK_CS2_Th.py Test`
6. **Quarter-Elliptical Corner Crack at the countersunk hole (with $b/t=0.25$) in a Plate:** 
    - ***Training:*** `python 06_CORNER_CRACK_CS3_QE.py Train`
    - ***Testing:*** `python 06_CORNER_CRACK_CS3_QE.py Test`
7. **Quarter-Elliptical Corner Crack at the countersunk hole (with $b/t=0.05$) in a Plate:** 
    - ***Training:*** `python 07_CORNER_CRACK_CS4_QE.py Train`
    - ***Testing:*** `python 07_CORNER_CRACK_CS4_QE.py Test`
8. **Through-Thickness Corner Crack at the countersunk hole (with $b/t=0.05$) in a Plate:** 
    - ***Training:*** `python 08_CORNER_CRACK_CS4_Th.py Train`
    - ***Testing:*** `python 08_CORNER_CRACK_CS4_Th.py Test`

### Twin Crack
9. **Twin Quarter-Elliptical Corner Cracks at the straight hole in a Plate:** 
    - ***Training:*** `python 09_TWIN_CORNER_CRACK_BH_QE.py Train`
    - ***Testing:*** `python 09_TWIN_CORNER_CRACK_BH_QE.py Test`
10. **Twin Through-Thickness Corner Crack at the straight hole in a Plate:** 
    - ***Training:*** `python 10_TWIN_CORNER_CRACK_BH_Th.py Train`
    - ***Testing:*** `python 10_TWIN_CORNER_CRACK_BH_Th.py Test`
11. **Twin Quarter-Elliptical Corner Crack at the countersunk hole (with $b/t=0.5$) in a Plate:** 
    - ***Training:*** `python 11_TWIN_CORNER_CRACK_CS2_QE.py Train`
    - ***Testing:*** `python 11_TWIN_CORNER_CRACK_CS2_QE.py Test`
12. **Twin Through-Thickness Corner Crack at the countersunk hole (with $b/t=0.5$) in a Plate:** 
    - ***Training:*** `python 12_TWIN_CORNER_CRACK_CS2_Th.py Train`
    - ***Testing:*** `python 12_TWIN_CORNER_CRACK_CS2_Th.py Test`