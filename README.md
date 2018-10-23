# Local cuda-9.0 install (no sudo version)
##### wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
##### chmod +x cuda_9.0.176_384.81_linux-run
##### mkdir cuda-9.0
##### mv cuda_9.0.176_384.81_linux-run cuda-9.0
##### cd cuda-9.0
##### ./cuda_9.0.176_384.81_linux-run -override
##### export LC_ALL="en_US.UTF-8" export LANGUAGE="en_US.UTF-8"
##### [accept] EULA , [no] driver installation , [/home/nlplab/ting/cuda-9.0] cuda Location [no] symbol link [no] cudasamples
##### tar cudnn-9.0-linux-x64-v7.tgz (download from https://developer.nvidia.com/cudnn)
##### cp cuda/include/cudnn.h /home/nlplab/ting/cuda-9.0/include
##### cp cuda/lib64/libcudnn* /home/nlplab/ting/cuda-9.0/lib64
##### chmod a+r /home/nlplab/ting/cuda-9.0/include/cudnn.h /home/nlplab/ting/cuda-9.0/lib64/libcudnn*
##### vim ~/.zshrc
##### export PATH=export PATH=$HOME/cuda-9.0/bin:$PATH
##### export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda-9.0/lib64/
##### source ~/.zshrc
##### nvcc --version
##### you can look [Cuda compilation tools, release 9.0, V9.0.176]
##### nvidia-smi
##### you can look gpu

#  pip install tensorflow-gpu==1.8.0 (you must python<=3.7.0)

##### Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
##### Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
##### Runs the op.
print(sess.run(c))

