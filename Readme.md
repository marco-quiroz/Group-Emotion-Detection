# Group Emotion Detection Based on Social Robot Perception

The objective of this project is to detect group emotions in social environments. For this, emotions are estimated individually, by frame and by scenes. The emotion of a scene represents the emotion of a group of people. With the area of the faces a scene is determined, the faces also serve to determine the emotions. The neural network used in this work was VGGFace, this model was trained to recognize the six basic emotions (happy, sad, fear, angry, disgust, surprise), but the system as a final result determines the emotion of a scene as positive, neutral and negative.

<div align="center">
  <img src="figures/scenes_method.png" alt="Group Emotions Detection" width="800"/>
</div>

The final result is shown in the following [Video](https://youtu.be/sabojnDs630). The simulation of the social environment is done on Ubuntu 18.04. This repository includes the implementation of the proposed method in a Google Colab environment, therefore only one video is analyzed to detect group emotions.
