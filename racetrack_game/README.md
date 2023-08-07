Here, I implemented the Racetrack game, see https://en.wikipedia.org/wiki/Racetrack_(game). 
It was made for the _Základy umělé inteligence_ (BI-ZUM) subject on _CVUT_ _FIT_, which I did through the _prg.ai/minor_ programme. 
Game can be started by running **racetrack_main.py**. 
At the beginning, the player appears in a menu, where he can write the name of his or a pre-prepared track (using BACKSPACE, he can delete characters), ENTER starts the race. 
Loading the track (finding the AI route) for the first time takes approximately 15 seconds. 
In the game itself the player can (using arrows and the ENTER key) select the place where the formula goes in the next step, using BACKSPACE, ha can return a step back.
There is a one AI controlled formula competing against the player.
The game ends when the player goes off the track (or, in rare cases, if he was to leave the entire screen due to a very high speed), or when he reaches the finish. 
The players reaches the finish not only when he ends on one of the finish tiles, he can also just drive through the finish. 
After reaching the finish (or crashing), the results are shown and the player has the option to choose a new track or play the same one once again. ESCAPE ends the game. 

The requirements (also in requirements.txt) to run the game are:
pip install --user numpy pillow matplotlib pyglet scikit-image

The player can create his own track (for example in GIMP), the game can load a map of any size and adjusts the window size correspondingly. 
However, the map should not be larger than 128x128 pixels - computing the AI route could take a long time and all icons would look very small. 
The racetrack needs to have a white color (rgb [210-255, 210-255, 210-255]), the start green (rgb [0-40, 210-255, 0-40]) and the finish red (rgb [210-255,0-40,0-40]). 
The space surrounding the racetrack can have any color not colliding with the previous three, for example black [255, 255, 255]. 

In the added .pdf file, there is a report on how the AI controlled formula works, and the code can be tested using pytest module and the pytesting_racetrack.py code. 
