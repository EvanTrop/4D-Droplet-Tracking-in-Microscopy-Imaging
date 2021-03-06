# 4D Droplet Tracking in Microscopy Imaging

## Contents
Cycle_GAN_Denoise.ipynb - contains code for training and testing a Cycle GAN model <br>
CycleGan_Modifications - contains modified versions of the original source code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix w/ modifications in train.py , image_folder.py , cycle_gan_model, and base_options.py + train_options.py <br>
Image_Synthesis_Conversion.ipynb - conntains code for generating synthetic data as well as running the connected component algorithm on images

Cancer cells can avoid senescence or apoptosis by maintaining telomere length above the critical threshold by alternative lengthening of telomeres (ALT) or by telomerase. Professor Zhang’s lab @ CMU has speculated that the ALT mechanism results from liquid - liquid phase separated droplets that form at the telomeres. In order to test their new hypothesis that ALT induces rearrangement of chromatin which is related to cancer, it is important for them to be able to quantify telomere clustering from cell microscopy imaging using their chemically-induced dimerization tool. <br>

The questions that the lab would like to answer given the above motivation are how many droplets are there within a cell, are droplets colocalized, what size are the droplets, and how does the volume of droplets change during coalescence. Bioimage analysis techniques are well suited to answer these questions. <br>

I worked alongside two teammembers to devise and implement a method for precise counting of droplets in images provided by Professor Zhang’s lab. I specifically worked on the process of denoising images for further downstream input into a counting algorithm. I used a deep learning method called Cycle Generative Adversarial Network (GAN) to map images from a noisy input domain to a noise free output domain. Cycle GAN is a powerful tool as it does not require a paired image dataset like previous methods and allows us to train Cycle GAN with a synthetic dataset containging the desired features.

## Methods
### Preprocessing 
Raw images from the training set were first processed by projecting the z - axis onto the x - y plane using the max value across the z - axis for a specific x,y coordinate and was done for each channel separately. Because values contained in the image correspond to intensity of the fluorophore, values were converted to a 0 - 255 scale. 

### Cycle GAN
Cycle GAN is a deep learning method based on the use of two pairs of discriminator(D) and generator(G) networks and is used for image translation between two domains. A single GAN works by training a generator to take in random noise and create an output that the discriminator will label as coming from the training set. The discriminator is trained to distinguish between samples that have come from the original dataset against ones that have come from the generator. The generator and discriminator are jointly trained and the result is a generator network that recovers the training data distribution. The networks used in the original paper outlining GAN’s were multilayer perceptrons with the generator outputting vectors in the training data space and the discriminator outputting a scalar value corresponding to the probability of the input coming from the data.

In Cycle GAN the generator network I used was a nine block residual network and the discriminator a 70 x 70 PatchGAN. A generator, discriminator pair are created for image domain A and B separately. The goal is to learn a mapping G<sub>A</sub> : A → B and G<sub>B</sub> : B → A so that at inference time given an input image in domain A, we can generate an image that is within the distribution of images from domain B, and vice versa from B to A. Additionally, to constrain the distributions of the generators, Cycle GAN implements a cycle consistency which can be represented as follows G<sub>B</sub>(G<sub>A</sub>(X)) ≈ X. The interpretation of this is that given an input image in domain A, the mapping to domain B and again back to domain A should result in an image that is very similar to the original input.
 
### Implementation of the simulated data 	
The original motivation for using Cycle GAN was to improve the results from a connected component algorithm which counted the number of droplets in an image. We hoped that Cycle GAN could denoise/enhance the input images and bring the output of the connected component algorithm closer to the manually labeled droplet counts in the test dataset.

In order to train Cycle GAN to map noisy images to clean images I needed to generate a dataset of images that resembled all characteristics of real images minus the noise. To accomplish this I first sample the number of droplets from a uniform distribution and select the radius for each droplet from a normal distribution based on statistics provided by Professor Zhang’s lab.

The next step involved masking pixels corresponding to a nucleus so that each droplet could be placed within the nucleus as the telomeres are located on chromosomes. To do this I first represented the nucleus as an ellipse but to further take into account asymmetry in real cell nuclei shape I created a second method. This second method randomly selects bounding box axes from a uniform distribution, the center location of the bounding box within a padded version of the original blank image, and the rotation of the bounding box. With the bounding box located within the blank image I randomly select four points within the bounding box and connect the points using bezier curves. This method provides a more realistic shape of the nucleus than an ellipse as it maintains a smooth curve but also allows for randomness in the nucleus geometry. With pixels corresponding to the nucleus masked, each droplet is placed inside the nucleus with care taken to prevent collisions between droplets. As the droplets take liquid form and are assumed to be spherical, I use OpenCV’s circle function to draw droplets in the image. OpenCV draws an approximate circle with ragged edges so a gaussian blur is used to smooth the edges. This overall process can be thought of as some form of domain randomization.


## Results

Qualitative result showing the output of passing raw images from two different time points through the trained generator networks. Left is the raw image, right is the image after being passed through the trained network. <br>
![alt text](https://github.com/EvanTrop/4D-Droplet-Tracking-in-Microscopy-Imaging/blob/main/Raw_Denoise_0.png)
![alt text](https://github.com/EvanTrop/4D-Droplet-Tracking-in-Microscopy-Imaging/blob/main/Raw_Denoise_1.png)

Number of droplets vs time for the test data set. Denoised corresponds to raw test images who were run through the trained model and used as input into the connected component algorithm. Calculated corresponds to raw test images who were processed with traditional denoising and enhancing methods. Finally, actual corresponds to manual annotations for number of droplets counted by a lab member.

The graph below is the result from standard Cycle GAN training with 100 training images from GFP channel/100 training images from mCherry channel belonging to image domain A, 200 synthetic images belonging to image domain B, batch size 4, learning rate .001, and image crop size of 256 x 256. <br>
![alt text](https://github.com/EvanTrop/4D-Droplet-Tracking-in-Microscopy-Imaging/blob/main/Standard_Training.png)

The graph below is the result of carrying out supervised pretraining before the standard Cycle GAN training. A dataset of 20 paired images was created and supverised training on the generator networks was carried out for 10 epochs before standard Cycle GAN training.<br>
![alt text](https://github.com/EvanTrop/4D-Droplet-Tracking-in-Microscopy-Imaging/blob/main/Supervised_Pretraining.png)

## Discussion
It appears that both the standard and pretraining Cycle GAN models capture the trend of the number of droplets that is traced by the manual annotations. Both training scenarios show that the learned model is more sensitive and captures a larger number of droplets than the traditional methods. Because the manual annotations are approximatitons and not the true ground truth its difficult to say whether false positives are captured. However, since photobleaching occurs in this imaging process, the ability of the annotator to pick up on droplets could be reduced as time progresses. 

Additionally we can see that difference in droplet counts for the pretrained model vs the manual annotations increases over time whereas the difference for the standard trained model vs the mannual annotations appears to remain more equal over time.
