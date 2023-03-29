# Larisch2023_EventBasedSNN

Python code for the Larisch, Berger, Hamker (2023) publication.


# Authors

- René Larisch (rene.larisch@informatik.tu-chemnitz.de)
- Lucien Berger (lucien.berger@informatik.tu-chemnitz.de)

## Dependencies

- Python v3.9
- numpy v1.22.3
- matplotlib v3.5.1
- ANNarchy v4.7.1.4 (https://annarchy.github.io/)
- Tonic v1.1.2 (https://tonic.readthedocs.io)
- tqdm v4.62.2

## Usage

### Train

The **Train** directory contains the python script to learn the V1 layer 4 models on the event-based MNIST (N-MNIST) dataset.
The dataset is provided by the Tonic package.

Start the learning with:

```
python ann_simple.py
```

or with four cores:

```
python ann_simple.py -j4
```

Please be aware, that depending on your hardware, training the full $400.000$ samples can take several days.


### Evaluate

The **Evaluation** directory contains five subdirectories, one for each of the five tested corruptions.
Please perform the following steps to evaluate the finally trained network on one of the corruptions:

1. Copy the weights matrices from the *output* directory, created by the *ann_simple.py*, to the *network_data* directory.
   Please remove the number from the file name. The names of the files should be:
    - InhibW.txt: feedforward weights from the LGN/input population to the inhibitory population
    - INLat.txt: lateral inhibitory weights between all inhibitory neurons
    - INtoV1.txt: inhibitory feedback weights from the inhibitory population to the excitatory simple cell population
    - V1toIN.txt: feedforward weights from the simple cell population to the inhibitory population
    - V1weight.txt: feedforward weights from the LGN/input population to the excititatory simple cell population

2. To present the complete N-MNIST training set and the test set with the corresponding corruption use (* depends on the corruption):

```
python nmnist_run_*.py
```

The corruption is presented on different levels to the network, including a corruption strength of zero (no corruption). 
An *output_svm* directory is created, where the activity on the training set is saved, and in corresponding subdirectories the activities on the test set with a certain level of corruption.

3. To evaluate the classification robustness with respect to a certain corruption use start the *svm.py* script.
It first fits a support-vector machine with the network activities on the N-MNIST training set and then predicts the classes based on the network activities, observed on the test set with the corruption.
For each level of corruption strength, it creates one *.txt file, containing the accuracy score.
Additionally, in each subdirectory in the *output_svm* directory, the image of a confusion matrix is created.

## Folder Structure

- Train
    - ann_simple.py: contains the network defintion and training loop
    - Make_lilarray.py: script to process the event list from Tonic to create a list of lists for ANNarchy
    - plots.py: some additional plots to monitor the network activity
    - rezeptiv.py: creates receptive fields of the final trained network
       
- Evaluation
    - directories for the five different corruption which are tested
    - each directory contains the network definition (net.py), a script to run the support vector machine for classification (svm.py), and a script to present the N-MNIST dataset with the corresponding corruption
    - svm_L_Transform_DropArea: Present the normal N-MNIST dataset and the drop area corruption for different area sizes (nmnist_run_droparea.py)
    - svm_L_Transform_DropEvent: Present the normal N-MNIST dataset and the drop event corruption for levels of drop rates for single events (nmnist_run_drop_event.py)
    - svm_L_Transform_UniformNoise_custom: Present the normal N-MNIST dataset and the uniform noise corruption for different number of additional events (nmnist_run_UniformNoise.py)
    - svm_L_Transform_SpatialJitter: Present the normal N-MNIST dataset and the spatial jitter corruption whith different levels of shifts along the space dimensions (nmnist_run_Jitter.py)
    - svm_L_Transform_TimeJitter: Present the normal N-MNIST dataset and the spatial jitter corruption whith different levels of shifts along the time dimension (nmnist_run_Jitter.py)


