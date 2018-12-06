---
layout: post
title: Proposal
---

## Training data generation for Deep Learning-based Robotic Vision Systems

The goal of this thesis is to design and evaluate a pipeline for automatically creating
realistic training data from simulations for real-world object classification and
detection algorithms.
Visually localising and classifying objects are core tasks for autonomous and
manipulating robots. They represent the basis for downstream robot capabilities,
such as grasping, manipulation or navigation. In recent years, Convolutional Neural
Networks [1] have become the method of choice as they have been beating
competing approaches by a high margin.
However, this class of algorithms is particularly data hungry requiring thousands to
millions of training data images. To make things worse, generating training data in
robotics is expensive requiring manual labelling and the usage of the robot’s
actuators. A solution to this dilemma is using simulation environments. However,
models trained on synthetic images have turned out be incapable of generalising
their knowledge to the real-world environment.
In recent years, image generating approaches, such as Generative Adversarial
Networks[2](GANs) have been increasingly used to augment synthetically created
data closing the realism gap. With these approaches, a high amount of automatically
labelled photorealistic images can be created to train models. Exemplary
applications are bee marker detection [3], hand pose estimation and eye tracking
[4], face verification [5], robot grasping [6] and navigation [7].
This thesis strives to evaluate the prospects and limits of current image generation
approaches for creating synthetic indoor object classification and detection training
data. The image generation process will be structured into three parts: background
environment image generation, foreground object image generation and final image
refinement. To create indoor scenes, the Gibson Environment [7] can be employed.

Objects of interest are provided by the ShapeNet dataset [8], which contains a
variety of household objects as 3D models. To enhance photorealism, the image
containing background and object foreground has to be post-processed using
approaches, such as Domain Adversarial Training [9] or Goggles [7]. Finally,
experiments with real-world objects shall investigate the effectiveness of
additionally created data on classification and localisation tasks.

## References
[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep
convolutional neural networks." Advances in neural information processing systems. 2012.
[2] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information
processing systems. 2014.
[3] Sixt, Leon, Benjamin Wild, and Tim Landgraf. "Rendergan: Generating realistic labeled data."
Frontiers in Robotics and AI5 (2018): 66.
[4] Shrivastava, Ashish, et al. "Learning from Simulated and Unsupervised Images through
Adversarial Training." CVPR. Vol. 2. No. 4. 2017.
[5] Huang, Rui, et al. "Beyond face rotation: Global and local perception gan for photorealistic and
identity preserving frontal view synthesis." arXiv preprint arXiv:1704.04086 (2017).
[6] Bousmalis, Konstantinos, et al. "Using simulation and domain adaptation to improve efficiency
of deep robotic grasping." arXiv preprint arXiv:1709.07857 (2017).
[7] Xia, Fei, et al. "Gibson Env: Real-World Perception for Embodied Agents." Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition. 2018.
[8] Chang, Angel X., et al. "Shapenet: An information-rich 3d model repository." arXiv preprint
arXiv:1512.03012 (2015).
[9] Wang, Mei, and Weihong Deng. "Deep Visual Domain Adaptation: A Survey."
Neurocomputing (2018).

## Supervisor:

Dr.-Ing. The Duy Nguyen
nguyen@gestalt-robotics.com
+49 30 616 515 60
www.gestalt-robotics.com