# Deep Distillation Recursive Network For Remote Sensing Video Imagery Super-Resolution

Code havs been available.
 

We are glad to hear if you have any suggestions, questions about implementation or sequences for testing.

## Citation

#If DDRN is useful for your research, please consider citing:
Jiang, K.; Wang, Z.; Yi, P.; Jiang, J.; Xiao, J.; Yao, Y. Deep Distillation Recursive Network for Remote Sensing Imagery Super-Resolution. Remote Sens. 2018, 10, 1700. 

## Contact

Please send email to kuijiang@whu.edu.cn


#Testing
Run sampling_scale.py to generat the testing samples.
You should copy the "def forward() " part in the training code (e.g., DDRN15_X4.py) and place them in the same position in the Testing code (e.g., Test_DDRN.py).
Then run Test_DDRN.py, you will obtain the SR results in the folder same as the testing samples.

#Training
Run sampling_scale.py to generat the training and evaluating samples.
Then run filelist.py to read these datasets and generate a train.txt and val.txt. Put them in the folder "\data" .
Run DDRN15_X4.py.