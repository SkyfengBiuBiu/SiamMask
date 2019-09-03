## SiamMask

This is implemented based on https://github.com/foolwood/SiamMask

Following the instruction of “SiamMask”, we could build the working environment. Besides those, some codes are added or modified for more functions.

Implementation details
I.	Because the demo codes require the continuous frame of the same video. The first step is to convert the video to individual frames. Since this network is trained at 55 frames per second. In this way, the specified video shall be transformed at 55 fps.

II.	This tracking method is limited to single-object tracking. In practice, we need the multi-object tracking more frequently. Therefore, the codes:” demo.py”, “test.py” (in “tools” folder) and “custom.py” (in “experiments/siammask_sharp” folder) are required to be modified to accept multiple objects. In my project, I added "new-demo.py","test_.py"and "custom_.py" so that I don't need to change the original codes.

III.	To classify and plot the target objects automatically, we import the “PyTorch-SSD -to-Object-Detection” model in the second section. To save the time for execution time, we restore the bounding boxes and labels for the target objects. And initialize the objects in frames every 30 frames. During this initialization, the IOU between the tracking bounding boxes and the bounding boxes from SSD models would be calculated. The tracking boxes would be removed, and the boxes detected by SSD model would be appended when their IOU are below the threshold. Specially, “target” list which stores the information of tracking objects would be cleared and reinitialize with the boxes detected by SSD model every 60 frames.

