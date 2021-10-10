## WGAN implementation

I took a lot of "config/utlities" code from different repos as I was also trying to make a generic intro point method.
I ended up making [entry.py](/entry.py) which acts as the entry point for the GAN

The WGAN code that I based it on, uses keras and extends it.
My current problem is that I cannot provide to it my own custom dataset even though I tried to do so in the `entry.py` but it seems the images are not valid to the model, even after resizing. I believe they might need to be grayscale but it has been many hours on this.

- I renamed the files to be numeral only
- What next?


```
PS C:\Users\chris\Documents\Repos\cGAN\wgan> python .\entry.py
hello world
2021-10-10 22:49:54.726046: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-10-10 22:49:55.289699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3967 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5
Model: "discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 16, 16, 64)        4864
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 16, 16, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 128)         204928
_________________________________________________________________
batch_normalization (BatchNo (None, 8, 8, 128)         512
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 256)         819456
_________________________________________________________________
batch_normalization_1 (Batch (None, 4, 4, 256)         1024
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 4, 4, 256)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 2, 2, 512)         3277312
_________________________________________________________________
batch_normalization_2 (Batch (None, 2, 2, 512)         2048
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 2, 2, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 1)                 2049
=================================================================
Total params: 4,312,193
Trainable params: 4,310,401
Non-trainable params: 1,792
_________________________________________________________________
Model: "generator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         [(None, 128)]             0
_________________________________________________________________
dense_1 (Dense)              (None, 8192)              1056768
_________________________________________________________________
batch_normalization_3 (Batch (None, 8192)              32768
_________________________________________________________________
re_lu (ReLU)                 (None, 8192)              0
_________________________________________________________________
reshape (Reshape)            (None, 4, 4, 512)         0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 8, 8, 512)         6554112
_________________________________________________________________
batch_normalization_4 (Batch (None, 8, 8, 512)         2048
_________________________________________________________________
re_lu_1 (ReLU)               (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 16, 16, 256)       3277056
_________________________________________________________________
batch_normalization_5 (Batch (None, 16, 16, 256)       1024
_________________________________________________________________
re_lu_2 (ReLU)               (None, 16, 16, 256)       0
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 32, 32, 128)       819328
_________________________________________________________________
batch_normalization_6 (Batch (None, 32, 32, 128)       512
_________________________________________________________________
re_lu_3 (ReLU)               (None, 32, 32, 128)       0
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 32, 32, 3)         9603
_________________________________________________________________
layer (Layer)                (None, 32, 32, 3)         0
=================================================================
Total params: 11,753,219
Trainable params: 11,735,043
Non-trainable params: 18,176
_________________________________________________________________
datasets/frida_kahlo/
[*] Epoch 0 / 50:   0%|                                                                                                                                     | 0/24 [00:00<?, ?it/s]2021-10-10 22:49:57.052341: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
[*] Epoch 0 / 50:   0%|                                                                                                                                     | 0/24 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "C:\Users\chris\Documents\Repos\cGAN\wgan\entry.py", line 67, in <module>
    model.train(dataset)
  File "C:\Users\chris\Documents\Repos\cGAN\wgan\model.py", line 144, in train
    for n_iter, batch in enumerate(loader):
  File "C:\Users\chris\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\tqdm\std.py", line 1180, in __iter__
    for obj in iterable:
  File "C:\Users\chris\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\tensorflow\python\data\ops\iterator_ops.py", line 761, in __next__
    return self._next_internal()
  File "C:\Users\chris\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\tensorflow\python\data\ops\iterator_ops.py", line 744, in _next_internal
    ret = gen_dataset_ops.iterator_get_next(
  File "C:\Users\chris\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\tensorflow\python\ops\gen_dataset_ops.py", line 2727, in iterator_get_next
    _ops.raise_from_not_ok_status(e, name)
  File "C:\Users\chris\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\tensorflow\python\framework\ops.py", line 6941, in raise_from_not_ok_status
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: StringToNumberOp could not correctly convert string: o
         [[{{node StringToNumber}}]] [Op:IteratorGetNext]
```