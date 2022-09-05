# CycleGAN implementation (apples to oranges) 
09/05/2022

While working on unpaired style transfer: \
Implementation of the [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593?amp=1) paper 
by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros with the help of educational resources.

The model translates apple pictures to the corresponding orange pictures. \
Made with Python and Keras.

### Results on test data: 

![617746-a1b3ac4ecbb13127](https://user-images.githubusercontent.com/37712544/188406519-4174004f-2fe6-42c8-93cc-56824d4641c3.png)
<img width="749" alt="Screen Shot 2022-09-05 at 10 48 31 AM" src="https://user-images.githubusercontent.com/37712544/188408557-833855e2-ef72-47fc-bb4e-bd7e32ea3fd5.png">

The model uses:
- a cycle consistency, identity and reconstruction loss
- Instance normalization instead of batch normalization
- a PatchGAN discriminator, which only penalizes structure at the scale of image patches rather than the whole image.
- a U-Net generator (encoder-decoder architecture with skip connections)
