# Person Re ID - Final Project Max

The purpose of this project was to investigate and implement the person re id algorithm. Luckily, there were many papers and videos to make some research, but it wasn't until I had to implement when things got complicated. But first I needed to land some ideas of what person reid meant to me at the beginning of this course, so I asked my teacher to solve some doubts. At the end of this course, I was able to implement the algorithm with the help of this repo: https://github.com/KaiyangZhou/deep-person-reid, but of course I had to make some tweaks here and there, learn what the original author did in the code, and try to add some features to the end result.

The requirements for this project were:
    - Have multiple cameras recording the same place in different perspectives. So I asked my friends for help, and recorded a common area in my university. Videos are available in the folder videos/, the third video had to be deleted due to Github rule of not uploading larger files than 50 MB.
    - Create a clean image dataset of the recorded videos in which each image was a frame of the video with an identified person.
    - Input the dataset into the algorithm and repo of the author in the original repo.
    - Train the model, and the test it with some data query.

In order for this project to work, you will need to access the file main-image.py and have my image dataset in the folder reid-data/new_dataset, but due to Github rule of not uploading larger commits than 100MB, it had to be deleted. But I still have it locally.

But in the main-image.py you could train the model to re identify a person in a multiple camera system.

You input the image query in reid-data/new_dataset/query, although it could be multiple images to search in the videos.

Once the model has been trained, the model would be test and it would output the images found in the dataset similar to the person in the image query.

Finally, the project would show the input videos with the re idenfied person that was query in the dataset.
