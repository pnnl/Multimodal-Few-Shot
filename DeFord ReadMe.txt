I wasn't sure how to edit the readme file so and I don't know anything about markdown so this is my temporary solution.

Repo Info and Data Location
This repo is a mess but there is at least some logic to it. 
You will probably have to update some of the file paths before you run anything because I've tried to remove 
unecessary things. Everything is contained in the Notebooks folder. I've deleted everything that isn't being used.
Also, you can find the .emd data I was using in my teams channel under files in a folder named "raw_data". "few-shot" 
contains modules and python code. "Testing" contains processed data and saved information. "Testing/multimodal1" is
the only folder I added to. The rest is just a copy from pyChip.

Capstone Code
Basically everything the capstone group used has been tossed. The only thing I'm still using is a couple of lines 
of code from the "TEM data" notebook to save the spectra and the images from the data. I highly recommend that is 
updated to read it in as pytorch tensor data so that the conversions don't have to happen in the code for the
classifier as it currently does. Honestly, it would probably be better to set up a spectra data processing notebook
or something to make it easier to change and remove it from inside the classifier where it is now
The rest of their code doesn't really work and can be found on the main branch if you need it.

Changes to pyChip
I built this out of the pyChip software. The only changes from the original software at the time I was working on it
are found in the "Basic_UI.ipynb" file and the modules in "few-shot"; "pychip_classifier.py" and "protonet.py". If
you are comparing with the pyChip code it's based on I have marked every line of code I've added or edited with a
comment "# Added". Most of them have a small snippet after to explain what they do.

How to Use it
Using it is just as simple as pyChip. Open "Basic_UI", update file paths and start running through the cells. The
only difference is that you can look at the spectra for different chips in the cell after that. Just put in the name
if the chip you want to look at and the spectra will be plotted. Currently it's being softmaxed and cut up so feel
free to remove those so you can see the raw data again.

The Way it Works
The changes I've made are pretty simple, basically the file that "TEM data" outputs gets read into the classifier at
the same time as the image data does. Then when the image is chipped, all of the spectral data for the same area gets
summed together to make a single spectrum for that chip. Then when the support sets are made the spectra for all the 
chips for each class get summed together for later (and currently softmaxed, that should probably be removed) producing
a single spectra for each class. 
With all that prep work done, the support set, the class spectra, the chip spectra, and each batch of the image chips
get passed into the protonet. The protonet then converts the support images into tensors and averages them before 
concatenating the class spectra to produce the fused support set. It then does the same with every chip in the batch 
and computes the distances and classifies the chips.

What Needs Work
Currently, it doesn't do a good job of classifying with multimodal data. I'd like to try classifying with spectral 
data alone (no image data) but I didn't have time. I think the spectral data should be processed in advance and then 
saved and passed in ready to use. As far as making it work goes, sometimes it worked and sometimes it didn't but I 
didn't have enough time at the end to be very systematic about it. I think the key is in the way the spectral data 
is processed. I think important insights into what is needed will be found by seeing how the spectral data performs
by itself.