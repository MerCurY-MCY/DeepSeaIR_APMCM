# Underwater Image Enhancement_APMCM 2024_A


本项目来自APMCM2024年A题。旨在利用轻量级卷积神经网络（Light-weight CNN）与多场景生成对抗网络（Multi-Scenario GAN）技术，对深海图像进行高效的复原与增强。


我们首先根据退化类型对海床图像进行分类。针对每种不同的Jerlov水类型，应用相应的随机退化方法生成多样化的退化图像。随后，采用轻量级CNN进行特征提取，并将其与多场景GAN框架相结合，以执行针对每种退化场景的定向恢复和增强。


我们的方法不仅提高了恢复效率，还确保了恢复图像细节和质量的高保真度。

##
在竞赛结束后，以学习为目的，我们进行了更多网络架构尝试。在生成器方面手动实现了U-Net架构生成器和ResNet架构生成器，并替换进行了比较。

## References
1. Li, C., Anwar, S., & Porikli, F. (2020). Underwater scene prior inspired deep underwater
image and video enhancement. Pattern Recognition, 98, 107038. https://doi.org/10.1016/j.patcog.2019.107038
2. Uplavikar, P., Wu, Z., & Wang, Z. (2019). All-in-one underwater image enhancement using
domain-adversarial learning. Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition Workshops (CVPRW), 512–519.https://doi.org/10.1109/CVPRW.2019.00076
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale
image recognition. arXiv. https://doi.org/10.48550/arXiv.1409.1556
4. Park, J., Han, D. K., & Ko, H. (2019). Adaptive weighted multi-discriminator CycleGAN for
underwater image enhancement. Journal of Marine Science and Engineering, 7(7), 200.
https://doi.org/10.3390/jmse7070200
5. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with
conditional adversarial networks. Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 1125–1134. https://doi.org/10.1109/CVPR.2017.632
6. Wang, Y., Zhang, J., Cao, Y., & Wang, Z. (2017, September). A deep CNN method for
underwater image enhancement. In 2017 IEEE international conference on image processing
(ICIP) (pp. 1382-1386). IEEE.
Team apmcm24209075 page 26 of 39
7. Anwar, S., Li, C., & Porikli, F. (2019). Underwater image enhancement and restoration
techniques: A comprehensive review. IEEE Access, 7, 84003–84021. https://doi.org/10.1109/ACCESS.2019.292422
