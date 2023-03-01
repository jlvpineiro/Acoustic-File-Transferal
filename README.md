# Acoustic-File-Transferal
Python script that implements a MOSQUITO covert channel between two air-gapped computers using audio jack re-tasking.

This script implements a MOSQUITO covert channel between two air-gapped computers using audio jack re-tasking. One computer converts a file to an audio signal and sends out the audio using its speaker. Then the other computer uses its microphone to receive the audio signal and convert it back to the original file. The computers will have to use the same conversion scheme for going between file and audio signal so that when one receives it, the signal can be converted back to the original file.

The background for this project comes from the work of Mordechai Guri, Yosef Solewicz, Andrey Daidakulov, and Yuval Elovici in their publication “MOSQUITO: Covert Ultrasonic Transmissions between Two Air-Gapped Computers using Speaker-to-Speaker Communication,” in which they demonstrated that this method of communication was possible and described the success they achieved.
