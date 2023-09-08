"""

FASHION RECOMMEDNATION SYSTEM

Feature for e-websites(specially fashion related)
User can upload desired image related to fashion and system can recommmend similar products(initialy 5 peoducts)
Its basically reverse image search - recommending images similar to uploaded image
Used by- Google, Pintrest, Amazon
- Powerful technique of grouping visually similar products

PRE-REQUISITES
- Deep learning
- CNN
- Transfer learning - using pre-trained models
    Here we will use RESNET(it is trained on imageNET dataset)
- Max pooling layer

FLOW
1. Import Model - importing resnet
2. Extract features - using resnet (44k,2048 2D array consisting of features)
                                   1 [1,2,3,4,5.....................2048]
                                   2 [1,2,3,4,5.....................2048]
                                   .
                                   .
                                2048 [1,2,3,4,5.....................2048]
3. Export features
4. Generate recommendations


Implementation specifications:
Python 3.7 (better tensorflow support)

Used:
python 3.7.5
tensorflow 2.3.1
protobuf 3.19.0-3.20.x


Model summary:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
resnet50 (Functional)        (None, 7, 7, 2048)        23587712
_________________________________________________________________
global_max_pooling2d (Global (None, 2048)              0
=================================================================
Total params: 23,587,712
Trainable params: 0
Non-trainable params: 23,587,712
_________________________________________________________________
None




STEPS
1 Upload file
2 Load file -> extract features
3 Generate recommendations
4 Print recommendations


IMPROVEMENTS SUGGESTED AND SHOULD BE IMPLEMENTED
1. Optimization
    44K ---> 10M then site might become slow
    can use library annoy which is also used by spotify
2. Deployment challenges
    storing 44k images that are abt 10gb
    accessing the same while displaying

"""