deep_net_03_vx.py : Trying to figure out the right batch_size, num_workers etc. These are the first attempts of training. \
Speed_test.txt are the performance results of those short trainings. 

trainer_v1.py : First script to do train+test \
trainer_v2.py : v1 was getting killed as the net is larger now so I load data tto gpu as I train not at the beginning \
trainer_v3.py : Use more than one GPUs since we run out of GPU Ram.
