## Meta-Demodulator

This repository contains code for "[Learning to Demodulate from Few Pilots via Offline and Online Meta-Learning](https://arxiv.org/abs/1908.09049)" - 
Sangwoo Park, Hyeryung Jang, Osvaldo Simeone, and Joonhyuk Kang (revised fully and to appear in IEEE Transactions on Signal Processing, arXiv post updated on 4 Dec. 2020)

### Updated (4 Dec. 2020) 

The code has been updated in order to capture more interesting aspects of meta-learning usages for demodulation (e.g., I/Q imbalance, comparison with conventional communication scheme: MMSE channel estimator + Maximum Likelihood demodulator). Essential part of meta-learning schemes has not been changed. Current version contains full code which may contain deprecated modules for final experiments (i.e., figures for the paper). Most essential experimental set-up has been organized in the paper in detail but if any ambiguity raises, please feel free to contact the authors (e.g., sangwoop@kaist.ac.kr).

### Dependencies

This program is written in python 3.7 and uses PyTorch 1.1.0 and scipy.
Tensorboard for pytorch is used for visualization (e.g., https://pytorch.org/docs/stable/tensorboard.html).
- pip install tb-nightly, pip install future, and pip install scipy might be useful.

### Usage for offline scenario

- Train model:
    
    To train the demodulator with default settings, execute
    ```
    python main_offline.py
    ```
    For the default settings and other argument options, see top of `main_offline.py`
    
- Test model:
    
    To test the demodulator with default settings, execute
    ```
    python main_offline.py --path_for_meta_trained_net </path/to/saved-meta-model> --path_for_bm2_net </path/to/saved-joint-trained-model> 
    ```
    
-  In 'run' folder, basic examples (including experiments for the figures in the paper) can be found.
    
   In 'saved_nets' folder, trained models with most preferable settings in the paper (network for Fig. 6, 7, 9) can be found.
    
   In 'generated_data' folder, meta-training and meta-test set with 100 meta-devices for the realistic scenario can be found. 


### Usage for online scenario

- Run model:
    
    To train and test the demodulator in an online manner with default settings, execute
    ```
    python main_online.py
    ```
    For the default settings and other argument options, see top of `main_online.py`
    
    In 'run' folder, basic examples (including experiments for the figures in the paper) can be found.
    
