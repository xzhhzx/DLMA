# Paper review: Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary



## Problem Statement and Motivation

Deep learning is widely used in various applications of image segmentation. However, for medical images, there exists ambiguous structure boundaries, which is a predominant problem compared with other image types and cannot be solved directly with conventional CNN architectures such as FCN and U-Net (as shown in Fig 1). To tackle with this problem, interactive segmentation algorithms have been proposed, which interactively receive manual inputs (e.g. key points/bounding boxes)  from user during inference time. Although the assist from human improves the performance, it also requires expertise knowledge during test time and is not fully automated.

<img src="figures\image-20201117082030502.png" alt="image-20201117082030502" style="zoom:50%;" />

Fig 1. The ambiguous structure boundary problem in medical image segmentation domain. Traditional methods such as U-Net fails in this case. 



To fully automate the process of segmenting medical image with ambiguous boundaries, this paper proposed a framework for 2D image segmentation. It integrates a novel network structure called Boundary Preserving Block (BPB) into conventional segmentation networks, which enhances the boundary information. Moreover, the framework takes advantage of the key points on the boundary and forces the network to learn additional boundary information by applying another network called Shape Boundary-aware Evaluator (SBE). We will see how they work in details in the following section. 



## Methodology

The entire framework (shown in Fig 2.) consists of three parts: Boundary Key Point Selection Module, Boundary Preserving Block (BPB) and Shape Boundary-aware Evaluator (SBE). Since they are not highly coupled, I would like to first introduce each separately and then bring them together when they are trained as a whole, in order to avoid information overloading right at the beginning.

<img src="figures\image-20201111135338324.png" alt="image-20201111135338324" style="zoom:80%;" />

Fig 2. Overview of the proposed novel framework. It consists of three parts: Boundary Key-point Selection Module (the gray box on the bottom left corner), the Boundary Preserving Block (integrated inside the auto-encoder network) and the Shape Boundary-aware Evaluator (purple box on the bottom right corner). In this blog, the upper part of the figure (auto-encoder + BPB) would be referred as segmentation network, and the SBE would be referred as SBE network.



###  Boundary Key Point Selection Module

This module is a simple algorithm that aims at finding key points on the boundary of ground truth segmentation map. We would like to prepare those key points for two purposes:

* use as ground truth when training the BPB
* use as input for SBE 

First, the boundary of the target object is obtained from ground-truth segmentation map by using Canny edge detector. Then, N points are selected on the boundary and connected to form a polygon. Repeat for T times and select out the key point sets with the highest IOU. The pseudo code is shown in Fig 3.  

<img src="figures\alg.png" alt="alg" style="zoom:50%;" />

Fig 3. Proposed boundary key point selection algorithm. From the experimental setups of the paper, T=40000 and N=6.

<img src="figures\image-20201117084336526.png" alt="image-20201117084336526" style="zoom:50%;" />

Fig 4. A visual example of boundary key point selection algorithm. The best key points set (as shown with the blue box) is selected based on the Dice score between ground truth map and the constructed polygon.



After the best key points set have been selected, there is another step for processing the key points into the final ground truth key point probability map. A small circle (disk) can be drawn around each key point which turns a single point (pixel) into a small region. As mentioned in the paper, the purpose of this step is to allow the tolerance of the key points position in the training phase [1]. Therefore, even if the prediction of key points does not exactly overlap with the ground truth key points (but very close), it is still considered as a good prediction. 

Again, keep in mind that this boundary key point map $M_{GT}$ would be used for both BPB and SBE, which will be introduced in the following two sections respectively.



###  Boundary Preserving Block (BPB)

The Boundary Preserving Block (BPB) is a novel architectural unit in CNN that **enhances the boundary information of input**, shown in Fig 5. It receives a feature map as input and outputs a feature map of the same size. Therefore this unit can actually be embedded into any network. Similar ideas of devising an architectural unit that can be easily embedded to perform a particular function are not new. A relatively famous one is called ["Squeeze-and-Excitation" Block (SE-Block)](https://arxiv.org/abs/1709.01507), which enhances important channel-wise features. 

<img src="figures\image-20201112020243919.png" alt="image-20201112020243919" style="zoom:67%;" />

Fig 5. Detailed structure of Boundary Preserving Block (BPB). 



Regarding details of BPB, the most important component is what they called boundary point map generator (shown as a yellow box in Fig 5) for producing a key point prediction map $\hat{M^i}$. It takes use of [dilation convolution](https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25) with various dilation rates. Different dilation rates focus on different size of receptive fields, in other words, small dilation rate is in charge of extracting near-by features and big dilation rate is for long-distance features. This design is similar to the idea of [GoogleNet](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43022.pdf). 

Here comes the final and also interesting part: calculating the loss of key point map. Compared with SE-Block, the proposed BPB not only embeds the architectural structure, but also embeds the loss. This loss (referred as $L^i_{Map}$) is calculated as the cross-entropy loss between key point prediction map $\hat{M^i}$ and ground-truth boundary key point map $M^i_{GT}$ for the i-th layer of the segmentation network. Here another question arises: how can we obtain $M^i_{GT}$? Remember that we have $M_{GT}$ from previous section, so we can obtain $M^i_{GT}$ by down-sampling $M_{GT}$, and in turn can be used to calculate the loss.



###  Shape Boundary-aware Evaluator (SBE)

Contrary to previous segmentation network, the Shape Boundary-aware Evaluator (SBE) is for classification. Or more specifically, from the perspective of GAN, it is a discriminator. Let's first have a look at its input and structure, then back to discuss how it can help to preserve boundary information as a discriminator. 

As shown in Fig 6, the input of SBE is a concatenation of segmentation map (predicted or ground-truth) $S$ and boundary key point map $M_{GT}$. The network is a simple conventional CNN that outputs an evaluation score.



<img src="figures\image-20201112032626334.png" alt="image-20201112032626334" style="zoom:67%;" />

Fig 6. Details of Shape Boundary-aware Evaluator (SBE) network. 



Now the question is, how can this SBE be trained and how can it contribute to boundary preservation? Well, the SBE and the segmentation network together are trained in an adversarial way similar to training a GAN. The SBE can be regarded as the discriminator which tells whether the predicted segmentation map $\hat{S}_{Pred}$ is consistent with the boundary key point map $M_{GT}$ or not [1]. On the hand, the segmentation network can be regarded as the (generalized) generator. During training, the segmentation network tries its best to produce a prediction map $\hat{S}_{Pred}$ that coincides well with the ground-truth key point map $M_{GT}$, so that it can foul the discriminator SBE. Therefore, this adversarial training forces the segmentation network to learn about the key point information and in turn preserves boundary. Also notice that, the SBE network can be discarded after training since it has completed its mission of supervision. 


### Loss definition and training

Now it's the time to bring all of these together. The training of the entire framework involves the adversarial training of segmentation network and SBE network. First let's look at their loss respectively. 

The **segmentation network** is trained with a sum of three different losses as shown in (1):

1. Conventional segmentation CE loss ($\hat{S}_{Pred}$ and $S_{GT}$)
2. Boundary-aware loss (can be regarded as generator loss, measures the similarity between $\hat{S}_{Pred}$ and $M_{GT}$, aims at minimizing the difference between prediction map and GT key point map), D(.) is the the SBE discriminator function
3. Key point map loss (measures the similarity between $\hat{M^i}$ and $M^i_{GT}$, aims at minimizing the difference between the predicted and ground-truth point maps)

<img src="figures\image-20201112040808001.png" alt="image-20201112040808001" style="zoom:50%;" />

On the other hand, the **SBE** network is trained with a single discriminator loss as shown in (5):

<img src="figures\image-20201112041344265.png" alt="image-20201112041344265" style="zoom:50%;" />

In the original paper, researchers trained the segmentation network 8 times and the SBE network 3 times in an adversarial way on each iteration [1].



## Results and Conclusions

The authors used two datasets for experiments: the PH2+ISBI 2016 Skin Lesion Challenge dataset and the Transvaginal Ultrasound (TVUS) dataset. From my perspective, their experiments can be mainly divided into three parts:

* Quantitative and qualitative evaluation between baseline and baseline+BPB+SBE 
* Ablation study to verify the contribution and generalization of each component of the framework
* Other experiments for tuning hyperparameters

First, they compared their method quantitatively with previous SOTA baseline methods such as U-Net and FCN. Results are shown below on the two datasets, which indicates that the novel framework outperform all previous methods. For further statistical significance proof, they also conducted paired [t-test](https://en.wikipedia.org/wiki/Student's_t-test) between baseline methods and their novel framework. For qualitative evaluation (shown in Fig 7), the authors compared the segmentation results with U-Net and proved the the effectiveness of their work.

<img src="figures\image-20201117074100700.png" alt="image-20201117074100700" style="zoom:67%;" />

<img src="figures\image-20201117074851580.png" alt="image-20201117074851580" style="zoom:67%;" />

Fig 7. Qualitative evaluation with visualization of segmentation map and key point map on two datasets. From the two examples one can see that the novel framework overcomes the erosion and dilation problems on the target region boundary comparing with U-Net. Also the key point map indicates that the network is learning to accurately detect key points which further facilitates segmentation.



Secondly, ablation study was also introduced to show that the BPB and SBE components both contribute to the performance improvement, regardless of the underlying network architecture, as shown in Fig 8.

<img src="figures\image-20201117075726683.png" alt="image-20201117075726683" style="zoom:67%;" />

Fig 8. Ablation results on two datasets. Each component of the framework was added incrementally to the baseline which also incrementally improves the performance.

Finally, other experiments such as the effect of different number and positions of BPBs and the effect of number of key point are conducted. In short, they found that more BPBs brings more performance gain and should be placed widely across the network. As for the number of key point, n=6 is the best on the TVUS dataset, but might vary for different tasks. Details can be found in the original paper if interested.



## Strengths, Weaknesses

**Strengths**

* The framework is fully automated without any interaction from user. 
* Taking use of the boundary key points is one of the most important idea of the whole framework, which is novel and also intuitive.
* It cleverly uses a GAN structure to perform training in an adversarial way.
* The experiments are very comprehensive which prove and explain the performance gain.



**Weaknesses**: 

* The ablation study would be more complete if it also includes Baseline+SBE in order to compare which part of the framework (i.e. BPB or SBE) contributes more to the performance gain.



## Suggestions for improvement / future work

There might be a better way to selection boundary key points instead of perform random attempts, for example:

* transferring to an image processing problem (the key points are usually on the sharp edges of the boundary)
* transferring to an optimization problem