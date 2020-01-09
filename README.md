# face_recognition

The present project represents a comparative analysis of image projection techniques (PCA, ICA, and LDA) designed to contrast the extent to which the use of fairness-aware datasets affects the performance of these models. The techniques are tested on two subsets of the FERET image dataset, one which aims to preserve the ethnical makeup of the population in the United States (70% white, 20% black, 10% asian), while the other displays a uniform distribution of ethnical background (33% white, 33% black, 33% asian). 

The training methodology is based on the one used in Delac et al. (2006). Training images are first preprocessed by mean substraction and standardization. PCA is then performed, resulting in a 180-dimensional subspace (40% of 450) which preserves 99.66% of the information in population-influenced data, and 99.68% in fairness-aware data respectively. These projections were then used as input data for ICA and LDA.

For testing purposes the standard FERET methodogoly was adopted. The FERET test dataset contains 5 subsets: a gallery set of 1,196 images and four probe sets (fb, fc, dup1, and dup2) of different sizes that . The methodology involved training a kNN model on the gallery images, and finding the nearest neighbour of each projected image in the probe set. Cumulating the success rates gives the performance of a projection technique on that test set. Three different distance measures were used for comparison: L1, L2, and cosine angle. The results have been compared to Delac et al. (2006), ..., and can be found in the table below:

<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan="3">Population-influenced dataset</td>
    <td colspan="3">Fairness-aware dataset</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>L1</td>
    <td>L2</td>
    <td>COS</td>
    <td>L1</td>
    <td>L2</td>
    <td>COS</td>
  </tr>
  <tr>
    <td>FB probe set</td>
    <td>PCA</td>
    <td>73.31%</td>
    <td>73.81%</td>
    <td>76.23%</td>
    <td>72.22%</td>
    <td>73.14%</td>
    <td>75.48%</td>
  </tr>
  <tr>
    <td></td>
    <td>ICA1</td>
    <td>62.01%</td>
    <td>63.26%</td>
    <td>77.07%</td>
    <td>60.59%</td>
    <td>61.34%</td>
    <td>74.39%</td>
  </tr>
  <tr>
    <td></td>
    <td>ICA2</td>
    <td>60%</td>
    <td>62.68%</td>
    <td>76.82%</td>
    <td>58.24%</td>
    <td>60.42%</td>
    <td>74.14%</td>
  </tr>
  <tr>
    <td></td>
    <td>LDA</td>
    <td>76.23%</td>
    <td>73.81%</td>
    <td>76.23%</td>
    <td>78.74%</td>
    <td>73.14%</td>
    <td>75.56%</td>
  </tr>
  <tr>
    <td>FC probe set</td>
    <td>PCA</td>
    <td>70.62%</td>
    <td>55.15%</td>
    <td>50.52%</td>
    <td>65.46%</td>
    <td>53.09%</td>
    <td>47.94%</td>
  </tr>
  <tr>
    <td></td>
    <td>ICA1</td>
    <td>65.98%</td>
    <td>67.01%</td>
    <td>72.16%</td>
    <td>62.37%</td>
    <td>63.4%</td>
    <td>72.68%</td>
  </tr>
  <tr>
    <td></td>
    <td>ICA2</td>
    <td>61.86%</td>
    <td>65.98%</td>
    <td>72.16%</td>
    <td>61.86%</td>
    <td>63.4%</td>
    <td>72.16%</td>
  </tr>
  <tr>
    <td></td>
    <td>LDA</td>
    <td>64.95%</td>
    <td>55.15%</td>
    <td>50.52%</td>
    <td>57.73%</td>
    <td>53.09%</td>
    <td>47.94%</td>
  </tr>
</table>

Delac, Kresimir & Grgic, Mislav & Grgic, Sonja. (2005). Independent comparative study of PCA, ICA, and LDA on the FERET data set. International Journal of Imaging Systems and Technology. 15. 252 - 260. 10.1002/ima.20059. 