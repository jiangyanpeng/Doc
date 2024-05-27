
import onnxruntime as ort
import torch
import onnxruntime as rt
import torch.onnx
import cv2
import numpy as np
import onnx
import argparse


def test_onnx(model_path, input_data, transpose=False):
    # 消除onnxruntime中的警告
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = rt.InferenceSession(model_path, so)

    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    print("----------------------------")
    # 打印输入节点的名字，以及输入节点的shape
    for i in range(len(sess.get_inputs())):
        print(sess.get_inputs()[i].name, sess.get_inputs()[i].shape)

    print("----------------------------")
    # 打印输出节点的名字，以及输出节点的shape
    for i in range(len(sess.get_outputs())):
        print(sess.get_outputs()[i].name, sess.get_outputs()[i].shape)
    print("----------------------------")

    for data in input_data:
        if transpose:
            data = data.transpose(0, 3, 1, 2)
        x = []
        x.append(data)

        pred_onx = sess.run([out_name], {input_name: x})
        print(pred_onx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx runtime model")
    parser.add_argument(
        "-m", "--model", help="onnx model path", type=str, required=True
    )
    parser.add_argument(
        "-i", "--input", help="onnx model input", type=str, required=False
    )
    parser.add_argument(
        "-g", "--image", help="onnx model image", type=str, required=False
    )
    parser.add_argument(
        "--save-input", help="save model input data",
        type=bool, default=False, required=False
    )
    parser.add_argument(
        "--save-output", help="save model output data",
        type=bool, default=False, required=False
    )
    args = parser.parse_args()
    # print(args)
    if args.input is not None:
        data = np.fromfile(args.input, dtype=np.float32)
        data = data.reshape(1, 3, 128, 128)
        test_onnx(args.model, data)
    else:
        print("----------------")
