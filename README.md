# face_recognition

The present project represents a comparative analysis of image projection techniques (PCA, ICA, and LDA) designed to contrast the extent to which the use of fairness-aware datasets affects the performance of these models. The techniques are tested on two subsets of the FERET image dataset, one which aims to preserve the ethnical makeup of the population in the United States (70% white, 20% black, 10% asian), while the other displays a uniform distribution of ethnical background (33% white, 33% black, 33% asian). 

The training methodology is based on the one used in Delac et al. (2006). Training images are first preprocessed by mean substraction and standardization. PCA is then performed, resulting in a 180-dimensional subspace (40% of 450) which preserves 99.66% of the information in population-influenced data, and 99.68% in fairness-aware data respectively. These projections were then used as input data for ICA and LDA.

For testing purposes the standard FERET methodogoly was adopted. The FERET test dataset contains 5 subsets: a gallery set of 1,196 images and four probe sets (fb, fc, dup1, and dup2) of different sizes that . The methodology involved training a kNN model on the gallery images, and finding the nearest neighbour of each projected image in the probe set. Cumulating the success rates gives the performance of a projection technique on that test set. Three different distance measures were used for comparison: L1, L2, and cosine angle. The results have been compared to Delac et al. (2006), ..., and can be found in the table below:

<table>
  <tr>
    <td></td>
    <td colspan="3">Population-influenced dataset</td>
    <td colspan="3">Fairness-aware dataset</td>
  </tr>
  <tr>
    <td></td>
    <td>L1</td>
    <td>L2</td>
    <td>COS</td>
    <td>L1</td>
    <td>L2</td>
    <td>COS</td>
  </tr>
</table>

Delac, Kresimir & Grgic, Mislav & Grgic, Sonja. (2005). Independent comparative study of PCA, ICA, and LDA on the FERET data set. International Journal of Imaging Systems and Technology. 15. 252 - 260. 10.1002/ima.20059. 