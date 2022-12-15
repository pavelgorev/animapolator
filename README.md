# Animapolator

Animapolator is a system of neural networks to generate 2D animations using interpolation. **This is a work in progress.**

Currently, it can generate two simple animations by a simple text description. This is trained only on two text descriptions with a dataset generated with Blender. 

The system consists of two GANs. The first GAN takes a character image as its input and generates the first and the last images of the animation. The second GAN generates a frame between two given frames, so it performs their interpolation.

The network may be trained for only two possible text descriptions.

The network works with 32x32 images.

# Performance

The results below are provided from the validation stage. The training and validation examples are generated in Blender and they differ by the colours and the shapes of their body parts. The blender file is provided in the repository.

This is how the trained network works for images interpolation.
The first image is the first frame of the animation. The second image - expected interpolation. The third one - last frame. The last one - actual interpolation
![download (2)](https://user-images.githubusercontent.com/38492449/207952055-e24466d7-2f3f-4afc-b643-7646d9d201b0.png)

It can be seen that the colours on the interpolated images are not precise. Probably, that's because I haven't trained it enough.

Below is the example for the edges generator. The first image is a reference image of a character. The 2nd and the 3rd images - expected first and last frames correspondingly. 4rd and 5th images - actually generated first and last frames correspondingly.

Character drinking
![image](https://user-images.githubusercontent.com/38492449/207952883-2adc6690-d17d-46af-a4cc-ef465241c0f6.png)

Character standing up
![image](https://user-images.githubusercontent.com/38492449/207953235-db383314-559b-44af-89e2-1b57dd502d2b.png)

# Further work

* Increasing diversity of graphics
* Increasing the number of possible text descriptions (the target is to reach the possibility to generate images by a description which has not been seen by the network during training)

# System description

Please refer the Wiki for details.

# Licence

Provided by MIT License.

Copyright (c) 2022 Pavel Gorev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
