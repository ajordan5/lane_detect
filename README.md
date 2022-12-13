# Semantic Segmentation Lane Detect

Lane detection based on [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

![image](https://user-images.githubusercontent.com/70724204/162772743-8063bce0-3c9e-461f-b754-218fd464b604.png)

To train the model I created lane masks on the [TuSimple dataset](https://paperswithcode.com/dataset/tusimple) using OpenCV. During training, the model output was compared with the prepared lane mask. Only two outputs are accepted (1 - Lane Pixel, 0 - Non-lane Pixel). The trained model classifies each pixel in view as either lane or background.

The fully trained model was validated in real-time with a dash-cam on my personal vehicle. **You can see a video of it [here](https://www.youtube.com/watch?v=zTQr3NC0Ax0)**

See Lane_Detection_on_TuSimple.ipynb notebook for a walkthrough of my method.
