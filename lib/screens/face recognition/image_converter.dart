import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as imglib;
import 'package:camera/camera.dart';

imglib.Image? convertToImage(CameraImage image) {
  try {
    if (image.format.group == ImageFormatGroup.yuv420) {
      print("This is the format yuv420");
      return _convertYUV420(image);
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      print("This is the format bgra8888");
      return _convertBGRA8888(image);
    }
    throw Exception('Image format not supported');
  } catch (e) {
    debugPrint("ERROR:" + e.toString());
  }
  return null;
}

imglib.Image _convertBGRA8888(CameraImage image) {
  // Convert the Uint8List to ByteBuffer
  ByteBuffer byteBuffer = image.planes[0].bytes.buffer;

  return imglib.Image.fromBytes(
    width: image.width,
    height: image.height,
    bytes: byteBuffer,
    format: imglib.Format.uint8, // Assuming the image data is in uint8 format
    numChannels: 4, // BGRA has 4 channels
    order: imglib.ChannelOrder.bgra, // Ensure channel order matches BGRA
  );
}

imglib.Image _convertYUV420(CameraImage image) {
  int width = image.width;
  int height = image.height;

  var img = imglib.Image(
      height: height,
      width: width); // Create an image with the specified width and height.
  final int uvyButtonStride = image.planes[1].bytesPerRow;
  final int? uvPixelStride = image.planes[1].bytesPerPixel;

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      final int uvIndex = uvPixelStride! * (x ~/ 2) +
          uvyButtonStride * (y ~/ 2); // Calculate UV index.
      final int index = y * width + x;
      final yp = image.planes[0].bytes[index]; // Y plane value.
      final up = image.planes[1].bytes[uvIndex]; // U plane value.
      final vp = image.planes[2].bytes[uvIndex]; // V plane value.

      // Convert YUV to RGB.
      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
          .round()
          .clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

      // Set the pixel color.
      img.setPixelRgb(x, y, r, g, b);
    }
  }
  return img;
}
