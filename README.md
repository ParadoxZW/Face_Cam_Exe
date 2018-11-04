# Face_Cam_Exe

## About

This is a repo for build an execution file of a face recognition system implemented by python/Mxnet. The models are from [insightface](https://github.com/deepinsight/insightface) and [mxnet-mtcnn](https://github.com/pangyupo/mxnet_mtcnn_face_detection).

## Requirement

- system<br>
  win10
- python environment<br>
  You need python3, opencv-python and mxnet to execute the python scripts if you want to do some test.
- requirement for build exe<br>
  *pyinstaller* and *pycryptodemo*(you can pip to install them, but I recommend you downloading source file and use setup.py to install pyinstaller)

## Usage

I already put my build config in `camexe.spec`, just run:

``` bash
pyinstaller camexe.spec
```

then you can build an exe app. If you want to build one can run on `OS X` or `Linux` you need generate `camexe.spec` by your own and config it when you receive errors during building.<br>
If you have problems about models or about how to generating `base.txt` and `base.npy`, you can push issues to discuss.

## License

MIT LICENSE