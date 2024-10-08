import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_face_recognition/model/user.dart';
import 'package:flutter_face_recognition/screens/face%20recognition/image_converter.dart';
import 'package:flutter_face_recognition/utils/local_db.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imglib;

class MLService {
  late Interpreter interpreter;
  List? predictedArray;

  Future<User?> predict(
      CameraImage cameraImage, Face face, bool loginUser, String name) async {
    List input = _preProcess(cameraImage, face);

    input = input.reshape([1, 112, 112, 3]);

    List output = List.generate(1, (index) => List.filled(128, 0));

    await initializeInterpreter();

    interpreter.run(input, output);
    output = output.reshape([128]);

    predictedArray = List.from(output);

    if (!loginUser) {
      LocalDB.setUserDetails(User(name: name, array: predictedArray!));
      return null;
    } else {
      User? user = LocalDB.getUser();
      List userArray = user.array!;
      int minDist = 999;
      double threshold = 1.5;
      var dist = euclideanDistance(predictedArray!, userArray);
      if (dist <= threshold && dist < minDist) {
        return user;
      } else {
        return null;
      }
    }
  }

  euclideanDistance(List l1, List l2) {
    double sum = 0;
    for (int i = 0; i < l1.length; i++) {
      sum += pow((l1[i] - l2[i]), 2);
    }

    return pow(sum, 0.5);
  }

  Future<void> initializeInterpreter() async {
    Delegate? delegate;
    try {
      if (Platform.isAndroid) {
        delegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
          ),
        );
      } else if (Platform.isIOS) {
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
            allowPrecisionLoss: true,
          ),
        );
      }

      var interpreterOptions = InterpreterOptions();

      // Try adding the GPU delegate, but catch any issues related to it
      if (delegate != null) {
        interpreterOptions.addDelegate(delegate);
      }

      interpreter = await Interpreter.fromAsset(
        'assets/facenet_model.tflite',
        options: interpreterOptions,
      );
      debugPrint('Interpreter initialized successfully.');
    } catch (e) {
      debugPrint(
          'Failed to load interpreter with delegate. Trying without delegate.');
      debugPrint(e.toString());
      // Try initializing without the GPU delegate if the first attempt failed
      try {
        interpreter =
            await Interpreter.fromAsset('assets/facenet_model.tflite');
        debugPrint('Interpreter initialized successfully without delegate.');
      } catch (e) {
        debugPrint('Failed to initialize interpreter even without delegate.');
        debugPrint(e.toString());
        rethrow; // Rethrow the exception or handle it as necessary
      }
    }
  }

  List _preProcess(CameraImage image, Face faceDetected) {
    imglib.Image croppedImage = _cropFace(image, faceDetected);
    imglib.Image img = imglib.copyResizeCropSquare(croppedImage, size: 112);

    Float32List imageAsList = _imageToByteListFloat32(img);
    return imageAsList;
  }

  imglib.Image _cropFace(CameraImage image, Face faceDetected) {
    imglib.Image convertedImage = _convertCameraImage(image);
    double x = faceDetected.boundingBox.left - 10.0;
    double y = faceDetected.boundingBox.top - 10.0;
    double w = faceDetected.boundingBox.width + 10.0;
    double h = faceDetected.boundingBox.height + 10.0;
    return imglib.copyCrop(convertedImage,
        x: x.round(), y: y.round(), width: w.round(), height: h.round());
  }

  imglib.Image _convertCameraImage(CameraImage image) {
    var img = convertToImage(image);

    var img1 = imglib.copyRotate(img!, angle: -90);
    return img1;
  }

  Float32List _imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        imglib.Pixel pixel = image.getPixel(i, j);

        // var pixel = image.getPixel(j, i);

        // Extract RGBA components from pixel value
        var r = pixel.r; // Red component
        var g = pixel.g; // Green component
        var b = pixel.b; // Blue component

        // int r = (pixel >> 24) & 0xFF;
        // int g = (pixel >> 16) & 0xFF;
        // int b = (pixel >> 8) & 0xFF;

        // Normalize and store pixel values
        buffer[pixelIndex++] = (r - 128) / 128.0;
        buffer[pixelIndex++] = (g - 128) / 128.0;
        buffer[pixelIndex++] = (b - 128) / 128.0;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }
}
