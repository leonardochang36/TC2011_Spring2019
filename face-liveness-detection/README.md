# Face Liveness Detection

The purpose of this project is to prove the concept of using face light reflections to authenticate a real, live person against pictures of that person. The project is based on the paper *Face Flashing: a Secure Liveness Detection Protocol based on Light Reflections*, which can be found [here](https://arxiv.org/pdf/1801.01949.pdf).


# Description

The software relies on the OpenCV library and it has 2 main components.

The first component is face detection. The software uses the Haar Cascade Frontal Face Alt 2 classifier to detect faces. Haar Cascade is a machine learning based approach where a cascade function is trained from a lot of positive and negative images and then it is used to detect objects in other images, in this case, faces.

The second component is the Eigen Faces Recognizer. A facial image is a point from a high-dimensional image space and a lower-dimensional representation is found, where classification becomes easy. The lower-dimensional subspace is found with Principal Component Analysis, which identifies the axes with maximum variance.

Using these 2 main components, the software can detect faces from the computer's webcam for training the Eigen Faces model and for authenticating against the model. For training the model, the software captures 100 pictures of a real person and 100 pictures of a picture/image of a person.

Once it's trained, the software can use the authenticate feature, which will use the trained model to compare 20 pictures taken in that moment against the model to identify if the person in front of the computer is real or not (a picture of a person).

Since Eigen Faces is sensitive to light changes, different light patterns can be displayed on the screen to make different light reflections in the face of the person to be authenticated. Pictures of a picture will not have this reflections since it's a flat surface, unlike a real, 3D face that can generate shadows and it can distribute light differently.

If only one person is used to train the model, there is a high chance it will only identify that one person as real or fake, since it needs more data to accurately know if random people are real or not.

# Author
Alejandro Herce Bernal
A01021150
