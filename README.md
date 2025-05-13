# SIFBench
SIFBench, a benchmark database developed to support the advancement and validation of machine learning methods for stress intensity factor (SIF) prediction. The database contains approximately 5 million unique configurations of cracks, geometries, and loading conditions, organized into 37 distinct scenarios. These scenarios are divided into single-crack and twin-crack cases, comprising roughly 1.6 million and 3.3 million geometries, respectively.


The single-crack category includes semi-elliptical surface cracks in a finite plate subjected to tensile loading, as well as various quarter-elliptical and through-thickness corner cracks located at straight and countersunk bolt holes. Most configurations are analyzed under three independent loading conditions: ***tension***, ***bending***, and ***bearing***. The twin-crack category consists of paired quarter-elliptical and through-thickness corner cracks, also positioned at straight and countersunk bolt holes and evaluated under the same three loading conditions, each applied separately.

## Directory Details
1. **files:** Data, models, predictions files. Data can be accessed from https://huggingface.co/datasets/tgautam03/SIFBench/tree/main
2. **nb_analysis:** Jupyter notebooks showing the detailed dataset analysis.
3. **src:** Source files for training FNN and FNO. 

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

## Single Crack Results
### Mean Normalized L2 Error
| Scenario                               | Loading | RFR    | SVR    | FNN    | FNO    |
|----------------------------------------|---------|--------|--------|--------|--------|
| QE SC (Finite Plate)                   | Tension | 0.0355 | 0.0156 | 0.0076 | 0.0451 |
| QE CC (Straight Hole)                  | Tension | 0.0210 | 0.0847 | 0.0145 | 0.0680 |
| QE CC (Straight Hole)                  | Bending | 0.0529 | 0.2465 | 0.0616 | 0.1275 |
| QE CC (Straight Hole)                  | Bearing | 0.0364 | 0.2478 | 0.0362 | 0.2049 |
| TT CC (Straight Hole)                  | Tension | 0.0294 | 0.1601 | 0.0184 | 0.1550 |
| TT CC (Straight Hole)                  | Bending | 0.0268 | 0.9059 | 0.0719 | 0.1751 |
| TT CC (Straight Hole)                  | Bearing | 0.0674 | 0.5235 | 0.0889 | 0.2706 |
| QE CC (Countersunk Hole; $b/t=0.75$)   | Tension | 0.0195 | 0.1448 | 0.0138 | 0.2509 |
| QE CC (Countersunk Hole; $b/t=0.75$)   | Bending | 0.0219 | 0.1250 | 0.0118 | 0.2037 |
| QE CC (Countersunk Hole; $b/t=0.75$)   | Bearing | 0.0301 | 0.1592 | 0.0168 | 0.2229 |
| QE CC (Countersunk Hole; $b/t=0.5$)    | Tension | 0.0231 | 0.1798 | 0.0232 | 0.1795 |
| QE CC (Countersunk Hole; $b/t=0.5$)    | Bending | 0.0256 | 0.2187 | 0.0146 | 0.2111 |
| QE CC (Countersunk Hole; $b/t=0.5$)    | Bearing | 0.1771 | 0.5372 | 0.3090 | 0.4605 |
| TT CC (Countersunk Hole; $b/t=0.5$)    | Tension | 0.1671 | 0.1634 | 0.1757 | 0.1174 |
| TT CC (Countersunk Hole; $b/t=0.5$)    | Bending | 0.2261 | 0.4616 | 0.3021 | 0.2079 |
| TT CC (Countersunk Hole; $b/t=0.5$)    | Bearing | 0.1757 | 0.1758 | 0.1738 | 0.1175 |
| QE CC (Countersunk Hole; $b/t=0.25$)   | Tension | 0.0198 | 0.1595 | 0.0144 | 0.2377 |
| QE CC (Countersunk Hole; $b/t=0.25$)   | Bending | 0.0240 | 0.1301 | 0.0139 | 0.2138 |
| QE CC (Countersunk Hole; $b/t=0.25$)   | Bearing | 0.0305 | 0.1681 | 0.0173 | 0.2572 |
| QE CC (Countersunk Hole; $b/t=0.05$)   | Tension | 0.0179 | 0.1590 | 0.0142 | 0.2271 |
| QE CC (Countersunk Hole; $b/t=0.05$)   | Bending | 0.0189 | 0.1366 | 0.0111 | 0.2219 |
| QE CC (Countersunk Hole; $b/t=0.05$)   | Bearing | 0.0297 | 0.1498 | 0.0138 | 0.2226 |
| TT CC (Countersunk Hole; $b/t=0.05$)   | Tension | 0.0284 | 0.1229 | 0.0373 | 0.0888 |
| TT CC (Countersunk Hole; $b/t=0.05$)   | Bending | 0.0286 | 0.1351 | 0.0416 | 0.1073 |
| TT CC (Countersunk Hole; $b/t=0.05$)   | Bearing | 0.0352 | 0.1373 | 0.0377 | 0.0910 |

### Mean Normalized Absolute Error
| Scenario                               | Loading | RFR      | SVR      | FNN      | FNO      |
|----------------------------------------|---------|----------|----------|----------|----------|
| QE SC (Finite Plate)                   | Tension | 0.0341   | 0.0138   | 0.0063   | 0.0429   |
| QE CC (Straight Hole)                  | Tension | 0.0183   | 0.0783   | 0.0126   | 0.0654   |
| QE CC (Straight Hole)                  | Bending | 0.2494   | 1.7469   | 0.1756   | 0.5131   |
| QE CC (Straight Hole)                  | Bearing | 0.0578   | 0.8123   | 0.0718   | 0.7145   |
| TT CC (Straight Hole)                  | Tension | 0.0277   | 0.1458   | 0.0155   | 0.1437   |
| TT CC (Straight Hole)                  | Bending | 0.0420   | 2.0930   | 0.3294   | 0.7150   |
| TT CC (Straight Hole)                  | Bearing | 0.0610   | 0.5495   | 0.0800   | 0.2668   |
| QE CC (Countersunk Hole; $b/t=0.75$)   | Tension | 0.0168   | 0.1415   | 0.0124   | 0.2454   |
| QE CC (Countersunk Hole; $b/t=0.75$)   | Bending | 0.0192   | 0.1159   | 0.0103   | 0.2020   |
| QE CC (Countersunk Hole; $b/t=0.75$)   | Bearing | 0.0263   | 0.1694   | 0.0155   | 0.2127   |
| QE CC (Countersunk Hole; $b/t=0.5$)    | Tension | 0.0299   | 0.2242   | 0.0233   | 0.1907   |
| QE CC (Countersunk Hole; $b/t=0.5$)    | Bending | 0.0271   | 0.2888   | 0.0189   | 0.2522   |
| QE CC (Countersunk Hole; $b/t=0.5$)    | Bearing | 0.6051   | 1.4120   | 0.8022   | 0.7981   |
| TT CC (Countersunk Hole; $b/t=0.5$)    | Tension | 25.8989  | 214.8582 | 63.1247  | 102.6938 |
| TT CC (Countersunk Hole; $b/t=0.5$)    | Bending | 11.7549  | 284.9972 | 107.7207 | 118.3158 |
| TT CC (Countersunk Hole; $b/t=0.5$)    | Bearing | 13.3025  | 204.2497 | 44.5282  | 103.0246 |
| QE CC (Countersunk Hole; $b/t=0.25$)   | Tension | 0.0212   | 0.1895   | 0.0152   | 0.2408   |
| QE CC (Countersunk Hole; $b/t=0.25$)   | Bending | 0.0211   | 0.1208   | 0.0122   | 0.2123   |
| QE CC (Countersunk Hole; $b/t=0.25$)   | Bearing | 0.0334   | 0.2833   | 0.0189   | 0.2609   |
| QE CC (Countersunk Hole; $b/t=0.05$)   | Tension | 0.0155   | 0.1499   | 0.0123   | 0.2220   |
| QE CC (Countersunk Hole; $b/t=0.05$)   | Bending | 0.0167   | 0.1261   | 0.0098   | 0.2199   |
| QE CC (Countersunk Hole; $b/t=0.05$)   | Bearing | 0.0263   | 0.1464   | 0.0123   | 0.2165   |
| TT CC (Countersunk Hole; $b/t=0.05$)   | Tension | 0.0269   | 0.1220   | 0.0374   | 0.0888   |
| TT CC (Countersunk Hole; $b/t=0.05$)   | Bending | 0.0270   | 0.2550   | 0.0386   | 0.2342   |
| TT CC (Countersunk Hole; $b/t=0.05$)   | Bearing | 0.0366   | 0.1517   | 0.0431   | 0.1464   |

## Twin Cracks Results
### Mean Normalized L2 Error
| Scenario            | Loading | RFR C1 | RFR C2 | SVR C1 | SVR C2 | FNN C1 | FNN C2 | FNO C1 | FNO C2 |
|---------------------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
| QE CC (Straight)    | Tension | 0.020  | 0.020  | 0.314  | 0.320  | 0.036  | 0.031  | 0.261  | 0.264  |
| QE CC (Straight)    | Bending | 0.026  | 0.027  | 0.468  | 0.474  | 0.039  | 0.035  | 0.338  | 0.353  |
| QE CC (Straight)    | Bearing | 0.021  | 0.021  | 0.559  | 0.575  | 0.028  | 0.040  | 0.365  | 0.359  |
| TT CC (Straight)    | Tension | 0.034  | 0.033  | 0.236  | 0.230  | 0.038  | 0.045  | 0.208  | 0.183  |
| TT CC (Straight)    | Bending | 0.032  | 0.031  | 0.915  | 0.930  | 0.203  | 0.076  | 0.216  | 0.221  |
| TT CC (Straight)    | Bearing | 0.039  | 0.043  | 0.591  | 0.589  | 0.038  | 0.041  | 0.326  | 0.332  |
| QE CC (Countersunk) | Tension | 0.013  | 0.013  | 0.080  | 0.079  | 0.014  | 0.014  | 0.060  | 0.056  |
| QE CC (Countersunk) | Bending | 0.013  | 0.013  | 0.110  | 0.112  | 0.012  | 0.012  | 0.097  | 0.100  |
| QE CC (Countersunk) | Bearing | 0.015  | 0.015  | 0.104  | 0.101  | 0.013  | 0.016  | 0.056  | 0.057  |
| TT CC (Countersunk) | Tension | 0.028  | 0.020  | 0.095  | 0.077  | 0.077  | 0.084  | 0.108  | 0.088  |
| TT CC (Countersunk) | Bending | 0.043  | 0.029  | 0.159  | 0.171  | 0.035  | 0.033  | 0.138  | 0.139  |
| TT CC (Countersunk) | Bearing | 0.033  | 0.026  | 0.092  | 0.078  | 0.074  | 0.303  | 0.076  | 0.081  |

### Mean Normalized Absolute Error
| Scenario            | Loading | RFR C1 | RFR C2 | SVR C1 | SVR C2 | FNN C1 | FNN C2 | FNO C1 | FNO C2 |
|---------------------|---------|--------|--------|--------|--------|--------|--------|--------|--------|
| QE CC (Straight)    | Tension | 0.017  | 0.017  | 0.322  | 0.331  | 0.032  | 0.028  | 0.269  | 0.274  |
| QE CC (Straight)    | Bending | 0.073  | 0.134  | 2.263  | 3.181  | 0.117  | 0.149  | 1.890  | 1.966  |
| QE CC (Straight)    | Bearing | 0.017  | 0.018  | 0.932  | 0.984  | 0.029  | 0.040  | 0.449  | 0.460  |
| TT CC (Straight)    | Tension | 0.032  | 0.031  | 0.223  | 0.217  | 0.033  | 0.039  | 0.196  | 0.173  |
| TT CC (Straight)    | Bending | 0.054  | 0.044  | 2.337  | 2.179  | 0.637  | 0.264  | 1.118  | 0.841  |
| TT CC (Straight)    | Bearing | 0.035  | 0.039  | 0.641  | 0.630  | 0.032  | 0.037  | 0.326  | 0.334  |
| QE CC (Countersunk) | Tension | 0.010  | 0.011  | 0.073  | 0.071  | 0.012  | 0.012  | 0.059  | 0.055  |
| QE CC (Countersunk) | Bending | 0.013  | 0.015  | 0.170  | 0.190  | 0.015  | 0.015  | 0.297  | 0.407  |
| QE CC (Countersunk) | Bearing | 0.012  | 0.012  | 0.107  | 0.102  | 0.013  | 0.015  | 0.060  | 0.058  |
| TT CC (Countersunk) | Tension | 0.025  | 0.017  | 0.085  | 0.070  | 0.069  | 0.076  | 0.100  | 0.082  |
| TT CC (Countersunk) | Bending | 0.169  | 0.149  | 0.889  | 1.187  | 0.128  | 0.145  | 0.898  | 1.141  |
| TT CC (Countersunk) | Bearing | 0.029  | 0.022  | 0.083  | 0.074  | 0.071  | 0.339  | 0.068  | 0.084  |
