# Fake ID Card Detection System.

## This is the project for the Itelligent System subject.

The project consisted in the creation of a system that will detect if an ID card was fake or not. To do that, I used the Open CV 3.3.1 version, specially The akaze tutorial code named planar_tracking.cpp located in **samples -> cpp -> tutorial_code -> features2D -> AKAZE_tracking**.

To complete the project the basics were:

- Use a generic id image. We don't want to check if the information in the card is real or not. We want to match if the id card matches with the generic image.
- For this example I used a Mexican INE card, which has an hologram as part of its security system. We want to use this feature integrated in the card to see if the ID card is a paper copy.
- The last point, shows that for this project we only want to know if the ID is a paper copy, not a really good fake ID that looks almost the same as the real.
- We also want to give some rendom orders to the user to move the card in certain dirction and with the angle transformation prove if the user is giving a real id or if he is using a pre recorded video.

Unfortunately I wasn't able to finish all the steps mentioned before. I could make the program find the matches between the images using the planar_tracking.cpp. I modied the original program making it to use a preloaded image, not an image captured in the moment. Also, I deleted all th orbs parts. I could not make the correct angle transformation to find if the rotation of the id card in real time orresponded to a real time action and not a video. An finally I could not modify it to find if the light reflected from the hologram corresponded to a real ID card and not a paper copy.

## Why I used Akaze?

The main reason to use the Akaze algorithm is that it was the algorithm that gave the most matches. I tried with flann, surf and orb, but all of them gave me few matches. That is why I decided to use Akze.