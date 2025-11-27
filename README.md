Create a conda environment.

Activate the conda environment.

Install required packages in the conda environment.

Check the available cameras.

Test the detection model.

To test the detection model using your computer camera, run:

yolo predict model=./runs/detect/vegetable_train/exp_final/weights/best.pt source= show=True

write the camera number you are using after source.

Running depth perception.

To run the depth perception of the RealSense camera, run:

python ./scripts/depth_display.py
