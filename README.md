# face_recognition

The present project represents a comparative analysis of facial projection techniques (PCA, ICA, and LDA) designed to contrast the extent to which the use of fairness-aware training affects the performance of these models. The techniques are tested on two 450-image subsets of the FERET image dataset, one which aims to preserve the ethnical makeup of the population in the United States (70% Caucasian, 20% African, 10% South-East Asian), while the other displays a uniform distribution of ethnical background (33% Caucasian, 33% African, 33% South-East Asian). Both datasets contain two images per class (person), aiming to simulate law enformcenet applications, where the number of available images per invidual is expected to be low.

The training methodology is based on the one used in [1]. Training images are first preprocessed by mean substraction and standardization. PCA is then performed, resulting in a 180-dimensional subspace (40% of 450) which preserves 99.66% of the information in population-influenced data, and 99.68% in fairness-aware data respectively. These projections were then used as input data for ICA and LDA. The resulting spaces were used to project previously unseen images, and to test the accuracy of the projection by comparing them to other projected facial images of the same individual.

For testing purposes the standard FERET methodogoly was adopted. The FERET test dataset contains 5 subsets: a gallery set of 1,196 images and four probe sets (fb, fc, dup1, and dup2) of different sizes that present the subjects in different postures or at a later time. The methodology involved training a kNN model on the gallery images, and finding the nearest neighbour of each projected image in the probe set. Cumulating the success rates gives the performance of a projection technique on that test set. Three different distance measures were used for comparison: L1, L2, and cosine angle. The results have been compared between the population-influenced and fairness-aware datasets, and can be found in the table below:

<p align="center">
  <table>
    <tr>
      <td></td>
      <td></td>
      <td colspan="3" align="center">Population-influenced dataset</center></td>
      <td colspan="3" align="center"><center>Fairness-aware dataset</center></td>
      <td align="center">Average difference</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td align="center">L1</td>
      <td align="center">L2</td>
      <td align="center">COS</td>
      <td align="center">L1</td>
      <td align="center">L2</td>
      <td align="center">COS</td>
      <td></td>
    </tr>
    <tr>
      <td>FB probe set</td>
      <td align="center">PCA</td>
      <td align="center">73.31%</td>
      <td align="center">73.81%</td>
      <td align="center"><b>76.23%</b></td>
      <td align="center">72.22%</td>
      <td align="center">73.14%</td>
      <td align="center"><b>75.48%</b></td>
      <td align="center">-0.83%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA1</td>
      <td align="center">62.01%</td>
      <td align="center">63.26%</td>
      <td align="center"><b>77.07%</b></td>
      <td align="center">60.59%</td>
      <td align="center">61.34%</td>
      <td align="center"><b>74.39%</b></td>
      <td align="center">-2%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA2</td>
      <td align="center">60%</td>
      <td align="center">62.68%</td>
      <td align="center"><b>76.82%<b/></td>
      <td align="center">58.24%</td>
      <td align="center">60.42%</td>
      <td align="center"><b>74.14%</b></td>
      <td align="center">-2.23%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">LDA</td>
      <td align="center"><b>76.23%</b></td>
      <td align="center">73.81%</td>
      <td align="center"><b>76.23%</b></td>
      <td align="center"><b>78.74%</b></td>
      <td align="center">73.14%</td>
      <td align="center">75.56%</td>
      <td align="center"><b>0.39%</b></td>
    </tr>
    <tr>
      <td>FC probe set</td>
      <td align="center">PCA</td>
      <td align="center"><b>70.62%</b></td>
      <td align="center">55.15%</td>
      <td align="center">50.52%</td>
      <td align="center"><b>65.46%</b></td>
      <td align="center">53.09%</td>
      <td align="center">47.94%</td>
      <td align="center">-3.27%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA1</td>
      <td align="center">65.98%</td>
      <td align="center">67.01%</td>
      <td align="center"><b>72.16%</b></td>
      <td align="center">62.37%</td>
      <td align="center">63.4%</td>
      <td align="center"><b>72.68%</b></td>
      <td align="center">-2.23%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA2</td>
      <td align="center">61.86%</td>
      <td align="center">65.98%</td>
      <td align="center"><b>72.16%</b></td>
      <td align="center">61.86%</td>
      <td align="center">63.4%</td>
      <td align="center"><b>72.16%</b></td>
      <td align="center">-0.86%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">LDA</td>
      <td align="center"><b>64.95%</b></td>
      <td align="center">55.15%</td>
      <td align="center">50.52%</td>
      <td align="center"><b>57.73%</b></td>
      <td align="center">53.09%</td>
      <td align="center">47.94%</td>
      <td align="center">-3.95%</td>
    </tr>
    <tr>
      <td>DUP1 probe set</td>
      <td align="center">PCA</td>
      <td align="center"><b>31.44%</b></td>
      <td align="center">29.5%</td>
      <td align="center">32.69%</td>
      <td align="center"><b>32.41%</b></td>
      <td align="center">28.81%</td>
      <td align="center">30.75%</td>
      <td align="center">-0.55%</td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA1</td>
      <td align="center">25.07%</td>
      <td align="center">27.56%</td>
      <td align="center"><b>40.72%</b></td>
      <td align="center">29.22%</td>
      <td align="center">30.06%</td>
      <td align="center"><b>42.24%</b></td>
      <td align="center"><b>2.72%</b></td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA2</td>
      <td align="center">21.05%</td>
      <td align="center">27.15%</td>
      <td align="center"><b>40.72%</b></td>
      <td align="center">27.84%</td>
      <td align="center">29.92%</td>
      <td align="center"><b>42.38%</b></td>
      <td align="center"><b>3.74%</b></td>
    </tr>
    <tr>
      <td></td>
      <td align="center">LDA</td>
      <td align="center"><b>35.18%</b></td>
      <td align="center">29.5%</td>
      <td align="center">32.69%</td>
      <td align="center"><b>32.41%</b></td>
      <td align="center">28.81%</td>
      <td align="center">30.75%</td>
      <td align="center">-1.8%</td>
    </tr>
    <tr>
      <td>DUP2 probe set</td>
      <td align="center">PCA</td>
      <td align="center">19.66%</td>
      <td align="center">18.8%</td>
      <td align="center"><b>23.5%</b></td>
      <td align="center"><b>23.93%</b></td>
      <td align="center">19.23%</td>
      <td align="center">23.08%</td>
      <td align="center"><b>1.42%</b></td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA1</td>
      <td align="center">14.1%</td>
      <td align="center">15.81%</td>
      <td align="center"><b>37.18%</b></td>
      <td align="center">19.23%</td>
      <td align="center">18.8%</td>
      <td align="center"><b>39.32%</b></td>
      <td align="center"><b>3.42%</b></td>
    </tr>
    <tr>
      <td></td>
      <td align="center">ICA2</td>
      <td align="center">12.82%</td>
      <td align="center">14.96%</td>
      <td align="center"><b>36.75%</b></td>
      <td align="center">18.8%</td>
      <td align="center">18.38%</td>
      <td align="center"><b>38.89%</b></td>
      <td align="center"><b>3.84%</b></td>
    </tr>
    <tr>
      <td></td>
      <td align="center">LDA</td>
      <td align="center"><b>25.21%</b></td>
      <td align="center">18.8%</td>
      <td align="center">23.5%</td>
      <td align="center"><b>24.36%</b></td>
      <td align="center">19.23%</td>
      <td align="center">23.08%</td>
      <td align="center">-0.28%</td>
    </tr>
  </table>
</p>



[1] Delac, Kresimir & Grgic, Mislav & Grgic, Sonja. (2005). Independent comparative study of PCA, ICA, and LDA on the FERET data set. International Journal of Imaging Systems and Technology. 15. 252 - 260. 10.1002/ima.20059. 