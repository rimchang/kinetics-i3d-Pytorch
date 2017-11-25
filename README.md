# I3D models trained on Kinetics Pytorch

this repo implements the network of I3D with Pytorch, pre-trained model weights are converted from tensorflow. 

### Sample code

you can convert tensorflow model to pytorch

```
# ./convert

./convert.sh

```

you can evaluate sample 

```
./multi-evaluate.py

```


There is a slight difference from the original model. you can compare original model output with pytorch model output in out directory

### Original Model (imagenet_joint.txt)

```
Norm of logits: 138.468658

Top classes and probabilities
1.0 41.8137 playing cricket
1.49716e-09 21.494 hurling (sport)
3.84312e-10 20.1341 catching or throwing baseball
1.54923e-10 19.2256 catching or throwing softball
1.13602e-10 18.9154 hitting baseball
8.80112e-11 18.6601 playing tennis
2.44157e-11 17.3779 playing kickball
1.15319e-11 16.6278 playing squash or racquetball
6.13194e-12 15.9962 shooting goal (soccer)
4.39177e-12 15.6624 hammer throw
2.21341e-12 14.9772 golf putting
1.63072e-12 14.6717 throwing discus
1.54564e-12 14.6181 javelin throw
7.66915e-13 13.9173 pumping fist
5.19298e-13 13.5274 shot put
4.26817e-13 13.3313 celebrating
2.72057e-13 12.8809 applauding
1.8357e-13 12.4875 throwing ball
1.61348e-13 12.3585 dodgeball
1.13884e-13 12.0101 tap dancing
```

### Pytorch Converted Model (imagenet_joint.txt)
```
Norm of logits: 140.496307

Top classes and probabilities
1.0 42.8406 playing cricket
2.52364e-08 25.3456 hurling (sport)
1.62931e-08 24.9081 catching or throwing baseball
3.08371e-09 23.2434 catching or throwing softball
1.28434e-09 22.3676 hitting baseball
8.13334e-11 19.6081 playing tennis
3.45132e-11 18.7509 playing kickball
3.37848e-11 18.7296 playing squash or racquetball
3.83346e-12 16.5533 shooting goal (soccer)
2.92696e-12 16.2835 hammer throw
1.03301e-12 15.242 pumping fist
4.33064e-13 14.3727 applauding
2.47362e-13 13.8127 tap dancing
2.40989e-13 13.7866 throwing ball
2.15394e-13 13.6743 throwing discus
1.96578e-13 13.5829 celebrating
1.89317e-13 13.5452 playing badminton
1.63867e-13 13.4009 headbutting
1.61229e-13 13.3846 dodgeball
1.00397e-13 12.9109 golf putting

```


Reference:

[kinetics-i3d](https://github.com/deepmind/kinetics-i3d)  
[tensorflow-model-zoo.torch](https://github.com/Cadene/tensorflow-model-zoo.torch)
