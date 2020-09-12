# Aravind Preliminary Task

## Task Description

  Glaucoma is a disease in the eye where elevated pressure causes eye damage. Due to the pressure, the optic nerve does not function properly and can lead to permanent loss of vision. However, the disease can be treated if detected at its early stages since the disease progresses very slowly. Current diagnostic tests are not the most efficient. The most common method used is tonography, a diagnostic test which records the eye pressure over a 4-minute period. 

  New research has shown that a diagnosis can be made by just using the fundus image using deep learning algorithms. Good results have been produced, however, not much research has been done which includes multiple sources of data. Limiting the dataset to one source can introduce bias and limits the number of images used for training. My goal was to create a model that uses data from several resources and create a full testing pipeline to perform a diagnosis. 

## Methods

  To start the project off, we were given 650 images from the ORIGA database. However, the number of images in this single dataset is too few to create a model without overfitting. To add more images, I introduced two new datasets with fundus images. This made a total of three datasets which also reduced the chance of overfitting and prevented any bias from a single source. 
  
  **Sources Used**
  * ORIGA (168 Glaucoma, 482 Normal)
  * [ACRIMA](https://figshare.com/s/c2d31f850af14c5b5232) (396 Glaucoma, 309 Normal)
  * [G1020](http://www.dfki.de/SDS-Info/G1020/) (296 Glaucoma, 724 Normal)
  
  Using the additional databases, increased the number of images from 650 to 2,375 with the Glaucoma Images making up approximately 36.2% of all data. This slight imbalance was addressed during the training and I include the AUC score as an evaluation metric. Before I could train the classification model, I performed an image segmentation task. 
  
## Image Segmentation

The fundus images from the 3 datasets come in two forms. Some include the full fundus image, while others include solely the optic disc. According to the paper below, glaucoma affects mainly the optic disc and its nearby surroundings. In fact, they proved that it was more effective to use just the optic disc than to use the full fundus image. This gives the Convolutional Neural Network model one specific area to look at.

  >Orlando JI, Prokofyeva E, del Fresno M, Blaschko MB. Convolutional neural network transfer for automated glaucoma identification. In: SPIE proceedings. 2017, p. 10160–10. https://doi.org/10.1117/12.2255740
  
  The optic disc carries a lot of importance when diagnosing glaucoma since it gives information on the cup to disc ratio. 

  >Diaz-Pinto, A., Morales, S., Naranjo, V. et al. CNNs for automatic glaucoma assessment using fundus images: an extensive validation. BioMed Eng OnLine 18, 29 (2019). https://doi.org/10.1186/s12938-019-0649-y
  
