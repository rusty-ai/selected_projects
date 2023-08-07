#Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
import random
import pyglet
import os

from racetrack_helpers import *

#Lets run the application
window = pyglet.window.Window(1344, 755, "Racetrack")

#LABELS AND NAMES
state = "choose map"  #in which "part" of the application we are right now
rt_name = "tracks/draha_01.png"  #Defaultní dráha
rt_name_label = pyglet.text.Label("", x=window.width//2 - 160, y=window.height//2)  #name under which is the racetrack stored in the folder
comment_during_choice_map = pyglet.text.Label("", x=window.width//2, y=window.height//2 - 40, anchor_x='center', anchor_y='center')
write_name_of_the_racetrack = pyglet.text.Label("Napiš název obrázku dráhy (včetně přípony):",
                                                x=window.width//2, y=window.height//2 + 40, anchor_x='center', anchor_y='center')

final_results = pyglet.text.Label("", x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')
new_map_or_old_map = pyglet.text.Label("", x=window.width//2, y=window.height//2 - 40, anchor_x='center', anchor_y='center')


#INNER VARIABLES NECCESARY FOR THE PROGRAM ITSELF
wall_value = 10**9
max_speed = 10
draha, eval_matrix, fm_path_AI, start_player = None, None, None, None
current_tile, possible_tiles, fm_path = None, None, None

#Let's set the formula icon:
formula_icon = pyglet.resource.image("others/formule_2.png")  #formula_icon
window.set_icon(formula_icon)

#Let's load the car engine sound and other sounds:
engine_sound = pyglet.media.load('others/menu_selection.wav', streaming=False)  
#originally true engine sound, but somehow it sounded weirdly in pyglet
play_engine_sound = True
menu_selection_sound = pyglet.media.load('others/menu_selection.wav', streaming=False)
win_sound = pyglet.media.load("others/winfretless.ogg", streaming=False)
game_over_sound = pyglet.media.load("others/GameOver.wav", streaming=False)


@window.event
def on_key_press(key_code, modifier):
    global state, rt_name, draha, eval_matrix, fm_path_AI, start_player, current_tile, possible_tiles, fm_path, wall_value, play_engine_sound

    if key_code == pyglet.window.key.M:
        if play_engine_sound == True:
            play_engine_sound = False
        else:
            play_engine_sound = True
        menu_selection_sound.play()

    if state == "choose map":
        #remove text when in "choose map"
        if key_code == pyglet.window.key.BACKSPACE:
            rt_name = rt_name[:-1]

        #check if the typed-in-name is correct, if so, start the game and prepare everything
        if key_code == pyglet.window.key.ENTER:
            if os.path.isfile(rt_name):
                state = "game"

                draha = load_racetrack(rt_name)
                list_starts = np.asarray(np.where(draha == 2)).T.tolist()
                list_finishes = np.asarray(np.where(draha == 3)).T.tolist()
                start = random.choice(list_starts)
                _, astar_path, astar_distances = astar(draha, start=start,
                                                       finish=random.choice(list_finishes), symbol_wall=1)
                eval_matrix, wall_value = create_evaluation_matrix_v01(draha, astar_path, astar_distances, list_finishes)

                fm_path_AI, final_speed = give_formula_path_v01(list_starts, eval_matrix, wall_value=10**9, max_speed=10)
                if fm_path_AI == []:
                    fm_path_AI = astar_path  #if pathfinding fails completely, the formula will move with speed 1, same as astar path

                start_player = random.choice(list_starts)
                possible_tiles = tiles_to_move_to(current_position=start_player, vector=(0, 0), max_speed=max_speed, eval_matrix=eval_matrix).tolist()
                current_tile = random.choice(possible_tiles)
                fm_path = [start_player]

            else:
                comment_during_choice_map.text = "Daný obrázek dráhy nenalezen. Zkus např. draha_01.png"

    elif state == "game":
        if key_code == pyglet.window.key.LEFT:
            current_tile = move_current_tile(current_tile=current_tile, possible_tiles=possible_tiles, key="left")
        elif key_code == pyglet.window.key.RIGHT:
            current_tile = move_current_tile(current_tile=current_tile, possible_tiles=possible_tiles, key="right")
        elif key_code == pyglet.window.key.UP:
            current_tile = move_current_tile(current_tile=current_tile, possible_tiles=possible_tiles, key="up")
        elif key_code == pyglet.window.key.DOWN:
            current_tile = move_current_tile(current_tile=current_tile, possible_tiles=possible_tiles, key="down")

        elif key_code == pyglet.window.key.BACKSPACE:
            #Let's us undo the last move
            if len(fm_path) >= 3:
                fm_path = fm_path[:-1]
                vector = (- fm_path[-2][0] + fm_path[-1][0], - fm_path[-2][1] + fm_path[-1][1])
                possible_tiles = tiles_to_move_to(current_position=fm_path[-1],
                                                  vector=vector, max_speed=max_speed,
                                                  eval_matrix=eval_matrix).tolist()
                current_tile = random.choice(possible_tiles)
            elif len(fm_path) >= 2:
                fm_path = fm_path[:-1]
                possible_tiles = tiles_to_move_to(current_position=fm_path[0],
                                                  vector=(0, 0), max_speed=max_speed,
                                                  eval_matrix=eval_matrix).tolist()
                current_tile = random.choice(possible_tiles)

        elif key_code == pyglet.window.key.ENTER:
            #Lets check if it's a crash or we found the finish or it's just a normal move.
            prev_tile = fm_path[-1]
            crash = False
            finish_found = False
            crash_unavoidable = False
            vector_points = np.array(line(prev_tile[0], prev_tile[1], current_tile[0], current_tile[1])).T
            for (x, y) in vector_points:
                #We can evaluate both crash and finish simultaneously bcs it iterates from tiles close to prev_tine
                #and ends with tiles near current_tile. So finish won't be found if we crash before it (bcs break).
                if eval_matrix[x, y] == wall_value:
                    crash = True
                    break
                if eval_matrix[x, y] == 0:
                    finish_found = True
                    break
            if not crash and not finish_found:
                #We made a valid move, let's make a new step
                fm_path.append(current_tile)
                vector = (current_tile[0] - prev_tile[0], current_tile[1] - prev_tile[1])
                possible_tiles = tiles_to_move_to(current_position=current_tile,
                                                  vector=vector, max_speed=max_speed,
                                                  eval_matrix=eval_matrix).tolist()

                #There is a last way to fail the race - possible_tiles is an empty array -> crash unavoidable (or out of map bounds)
                if len(possible_tiles) == 0:
                    crash_unavoidable = True
                else:
                    #Let's choose randomly the next current_tile
                    current_tile = random.choice(possible_tiles)
            if crash:
                fm_path.append(current_tile)
                state = "crash"
                game_over_sound.play()
            elif finish_found:
                fm_path.append(current_tile)
                state = "finish found"
                win_sound.play()
            elif crash_unavoidable:
                state = "crash unavoidable"
                game_over_sound.play()
                if crash != False or finish_found != False:  #Just so that we wouldn't append it twice
                    fm_path.append(current_tile)

            if play_engine_sound == True:
                engine_sound.play()

    elif state == "crash" or state == "finish found" or state == "crash unavoidable":
        if key_code == pyglet.window.key.ENTER:
            #Let's play the same racetrack again
            state = "game"

            #Let's try to make a new AI path. If it fails, use the old one:
            list_starts = np.asarray(np.where(draha == 2)).T.tolist()
            #AI_start = random.choice(list_starts)
            new_fm_path_AI, final_speed = give_formula_path_v01(list_starts, eval_matrix, wall_value=wall_value, max_speed=max_speed)
            if new_fm_path_AI != []:
                fm_path_AI = new_fm_path_AI

            start_player = random.choice(list_starts)
            possible_tiles = tiles_to_move_to(current_position=start_player, vector=(0, 0), max_speed=max_speed, eval_matrix=eval_matrix).tolist()
            current_tile = random.choice(possible_tiles)
            fm_path = [start_player]

        elif key_code == pyglet.window.key.SPACE:
            state = "choose map"


@window.event
def on_text(text):
    global state, rt_name
    if state == "choose map" and text != ("\n") and text != ("\r"):  #pressing ENTER == "\r", "\n" is unnecessary I guess
        #racetrack_name.text = racetrack_name.text + text
        rt_name = rt_name + text
        comment_during_choice_map.text = ""


@window.event
def on_draw():
    global state, rt_name, draha, eval_matrix, fm_path_AI, start_player, current_tile, possible_tiles, fm_path
    window.clear()
    if state == "choose map":
        rt_name_label.text = "Název: " + rt_name
        rt_name_label.draw()
        comment_during_choice_map.draw()
        write_name_of_the_racetrack.draw()
    elif state == "game":

        #fig, ax = plt.subplots(figsize=(15, 15))
        J = display_rt(racetrack=draha, fm_path_AI=fm_path_AI, fm_path=fm_path, possible_tiles=possible_tiles,
                       current_tile=current_tile)
        plt.axis('off')
        plt.savefig('current_situation.png', dpi=100, transparent=True, bbox_inches="tight", pad_inches=0)
        current_situation = pyglet.image.load('current_situation.png')

        #Now we need to scale the plotted image so that it would fit nicely into the window
        im_height, im_width = draha.shape[0], draha.shape[1]
        if (window.height/im_height < window.width/im_width) == True:
            py_image_height = window.height
            py_image_width = window.height / im_height * im_width
        else:
            py_image_width = window.width
            py_image_height = window.width / im_width * im_height
        #current_situation.anchor_x = current_situation.width // 2    #x=window.width//2
        current_situation.blit(x=0, y=0, width=py_image_width, height=py_image_height)
        plt.close('all')  #See https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

    elif state == "crash" or state == "finish found" or state == "crash unavoidable":
        if state == "crash":
            final_results.text = "Vyjetí z dráhy. Hra skončila po " + str(len(fm_path)) + " tazích."
        if state == "finish found":
            final_results.text = "Gratulace k úspěšnému dojetí dráhy! Oponentovi trvala cesta " + str(len(fm_path_AI)) + ", Vám " + str(len(fm_path)) + " tahů."
        if state == "crash unavoidable":
            final_results.text = "Příliš velká rychlost způsobila kompletní opuštění dráhy. Hra skončila po " + str(len(fm_path)) + " tazích."

        new_map_or_old_map.text = "SPACE bar: výběr nové mapy, ENTER: opakování stejné mapy"
        final_results.draw()
        new_map_or_old_map.draw()

pyglet.app.run()
