# Group Emotion Detection Based on Social Robot Perception

The objective of this project is to detect group emotions in social environments. For this, emotions are estimated individually, by frame and by scenes. The emotion of a scene represents the emotion of a group of people. With the area of the faces a scene is determined, the faces also serve to determine the emotions. The neural network used in this work was VGGFace, this model was trained to recognize the six basic emotions (happy, sad, fear, angry, disgust, surprise), but the system as a final result determines the emotion of a scene as positive, neutral and negative.

<div align="center">
  <img src="figures/scenes_method.png" alt="Group Emotions Detection" width="800"/>
</div>

The final result is shown in the following [Video](https://youtu.be/sabojnDs630). The simulation of the social environment is done on Ubuntu 18.04. This repository includes the implementation of the proposed method in a Google Colab environment, therefore only one video is analyzed to detect group emotions.


## Data Used

<img src="figures/Emotions.png" alt="data example" width="800"/>
<!-- ![alt text](https://github.com/juan1t0/multimodalDLforER/blob/master/figures/data_example.png) -->

The used data are shared [here](https://drive.google.com/file/d/1JAGejLFaymrIsq44icV42IdaAdydSdk9/view?usp=sharing), this zip contains all the data for each modality in numpy array format.

This dataset is acquire form the original [EMOTIC dataset](http://sunai.uoc.edu/emotic/download.html).

Moreover, the number of annotated emotions in EMOTIC (26) were reduced by grouping, following the taxonomy of Mr. Plutchik, into eight groups.



The weighted random sampler from pytorch was used in training time trying to solve the unbalancing of the EMOTIC dataset.



### Citation
If you use our code or models in your research, please cite with:
```
@article{quiroz2022group,
  title={Group emotion detection based on social robot perception},
  author={Quiroz, Marco and Pati{\~n}o, Raquel and Diaz-Amado, Jos{\'e} and Cardinale, Yudith},
  journal={Sensors},
  volume={22},
  number={10},
  pages={3749},
  year={2022},
  publisher={MDPI}
}
```

#### Acknowledgments
This research was supported by the FONDO NACIONAL DEDESARROLLO CIENTÍFICO, TECNOLÓGICO Y DE INNOVACIÓN TECNOLÓGICA - FONDECYT as executing entity of CONCYTEC under grant agreement no.01-2019-FONDECYT-BM-INC.INV in the project RUTAS: Robots for Urban Tourism,Autonomous and Semantic web based.
