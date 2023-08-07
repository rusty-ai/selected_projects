#Necessary imports
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.draw import line
import random
import pyglet


def load_racetrack(name, upper_bound=210, lower_bound=40):
    """Loads rgb picture of the racetrack, returns 2D np.array of the racetrack
    start needs to be green [0,255,0], finish red [255,0,0], space white [255,255,255] and wall anything else (ideally black [0,0,0])
    the colors need not to have those exact colors, any element lower than lower_bound is treated as 0 and upper than upper_bound as 255
    Args:
        name(str): name in string of the file of the racetrack
        upper_bound(int, optional): any element of [r,g,b] upper than upper_bound is treated as 255. Defaults to 210.
        lower_bound(int, optional): any element of [r,g,b] lower than lower_bound is treated as 0. Defaults to 40.
    Returns:
        2D np.array of the racetrack. 0 - space, 1 - wall, 2 - start, 3 -finish
    """
    rgb_draha = Image.open(name)
    rgb_draha = np.array(rgb_draha)
    draha = np.ones(shape=(rgb_draha.shape[0], rgb_draha.shape[1]))  #wall
    draha[(rgb_draha[:, :, 0] > upper_bound) & (rgb_draha[:, :, 1] > upper_bound) & (rgb_draha[:, :, 2] > upper_bound)] = 0  #space
    draha[(rgb_draha[:, :, 0] < lower_bound) & (rgb_draha[:, :, 1] > upper_bound) & (rgb_draha[:, :, 2] < lower_bound)] = 2  #start
    draha[(rgb_draha[:, :, 0] > upper_bound) & (rgb_draha[:, :, 1] < lower_bound) & (rgb_draha[:, :, 2] < lower_bound)] = 3  #finish
    return draha


#Lets do A* algorithm to find the path from start to finish
def list_8_tiles_sourrounding_the_middle_one(tile):
    """Recieves [x, y] coordinates of a tile, returns a list of eight tiles surrounding it, used in astar"""
    desired_list = [[tile[0], tile[1] + 1], [tile[0], tile[1] - 1], [tile[0] - 1, tile[1]], [tile[0] + 1, tile[1]],
                    [tile[0] + 1, tile[1] + 1], [tile[0] + 1, tile[1] - 1], [tile[0] - 1, tile[1] + 1], [tile[0] - 1, tile[1] - 1]]
    return desired_list


def astar(mapa, start, finish, symbol_wall=1):
    """A* algorithm adapted from my BI-ZUM assignment. note - unlike in that assignment, start and finish are both (x, y), not (y, x)
    and we look at 8 tiles surrounding the middle one instead of 4 and take into account that the corner tiles are further from (x,y)

    Args:
        mapa (2D np array): map of the racetrack, the only relevant part is the wall
        start (tupe of two ints): start coordinates (x, y)
        finish (tuple of two ints): finish coordinates (x, y)
        symbol_wall (int, optional): number describing wall in the map, defaults to 1.

    Returns:
        map_open_closed: map of tiles that were open/closed. Not necessary for racetrack
        path: list of (x, y) coordinates of the found route. in order from start to the back. If not found, the list is empty
        distances: list of distances of the found route to the finish. Corresponds to path_backwards. If not found, empty
    """
    list_open = [[start[0], start[1], np.sqrt(np.abs(start[0]-finish[0])**2 + np.abs(start[1]-finish[1])**2), 0]]
    #<--besides coordinates, there is also a the aerial distance tile-finish + distance of the current path to the start
    symbol_open, symbol_closed = 3, 8
    map_open_closed = np.zeros(mapa.shape, dtype=int)  # 3 ... symbol_open, 9 ... symbol_closed
    map_open_closed[start[0], start[1]] = symbol_open
    map_path_back = np.empty((mapa.shape[0], mapa.shape[1], 2))
    map_path_back[start[0], start[1]] = np.array(start)
    #We need to monitor distances of the path from the start to the current tile
    map_path_dist_to_start = np.empty(mapa.shape)
    map_path_dist_to_start[start[0], start[1]] = 0

    finish_found = False
    while finish_found == False:

        #Lets choose random tile for expansion out of the best ones
        tile_for_expansion = random.choice([open_tile for open_tile in list_open if (open_tile[2] + open_tile[3]) == min((open_tile[2] + open_tile[3]) for open_tile in list_open)])

        #Remove chosen tile from the list and the map of open ones, write down the way to the previous tile
        list_open.remove(tile_for_expansion)
        map_open_closed[tile_for_expansion[0], tile_for_expansion[1]] = symbol_closed

        #Look at eight tiles surrounding the chosen one
        for tile in list_8_tiles_sourrounding_the_middle_one(tile_for_expansion):

            #If we are right next to the edge of the map, there is a problem. So:
            if tile[0] < 0 or tile[0] >= mapa.shape[0] or tile[1] < 0 or tile[1] >= mapa.shape[1] == True:
                continue

            #This is newly defined
            distance = np.sqrt((tile[0] - tile_for_expansion[0])**2 + (tile[1] - tile_for_expansion[1])**2)

            #Mark openable tiles around chosen tile as open ones, put them into the list
            if mapa[tile[0], tile[1]] != symbol_wall:
                if map_open_closed[tile[0], tile[1]] != symbol_open and map_open_closed[tile[0], tile[1]] != symbol_closed:
                    map_open_closed[tile[0], tile[1]] = symbol_open
                    map_path_back[tile[0], tile[1]] = np.array([tile_for_expansion[0], tile_for_expansion[1]])

                    #This needs to be modified so that if we are moving diagonally, we add sqrt(2), if horizontally/vertically, we add 1
                    list_open.append([tile[0], tile[1], np.sqrt(np.abs(tile[0]-finish[0])**2 + np.abs(tile[1]-finish[1])**2), tile_for_expansion[3] + distance])
                    map_path_dist_to_start[tile[0], tile[1]] = tile_for_expansion[3] + distance

                #If the tile is already open, check, whether the newly found route is faster. Update it if it's the case
                if map_open_closed[tile[0], tile[1]] == symbol_open:
                    if map_path_dist_to_start[tile[0], tile[1]] >= tile_for_expansion[3] + distance:
                        list_open.remove([tile[0], tile[1], np.sqrt(np.abs(tile[0] - finish[0])**2 + np.abs(tile[1] - finish[1])**2), map_path_dist_to_start[tile[0], tile[1]]])
                        list_open.append([tile[0], tile[1], np.sqrt(np.abs(tile[0] - finish[0])**2 + np.abs(tile[1] - finish[1])**2), tile_for_expansion[3] + distance])
                        map_path_back[tile[0], tile[1]] = np.array([tile_for_expansion[0], tile_for_expansion[1]])
                        map_path_dist_to_start[tile[0], tile[1]] = tile_for_expansion[3] + distance

                #Lets check if we found the finish, if so, reconstruct the path
                if tile[0] == finish[0] and tile[1] == finish[1]:
                    finish_found = True
                    reconstructed = False
                    path_backwards = [[int(tile[0]), int(tile[1])]]
                    distance_backwards = [map_path_dist_to_start[tile[0], tile[1]]]
                    while reconstructed == False:
                        next_tile = map_path_back[int(path_backwards[-1][0]), int(path_backwards[-1][1])].tolist()
                        path_backwards.append([int(next_tile[0]), int(next_tile[1])])  #could do just .append(next_tile), but it's elements would not be integers
                        distance_backwards.append(map_path_dist_to_start[int(next_tile[0]), int(next_tile[1])])
                        if next_tile[0] == start[0] and next_tile[1] == start[1]:
                            reconstructed = True

                    #In path_backwards, the distances are from finish to start -> for "start", it is 0, for "finish", it is a lot. I need it the other way around
                    #distance_backwards = distance_backwards[::-1] #reverses the list
                    #-> what I did was just reversing the path so it won't go from finish to start but from start to finish, now it corresponds to distances_backwards
                    #(because they are no longer backwards)
                    path = path_backwards[::-1]
                    distances = distance_backwards

        #Check if there are any open tiles left - if not, end the search
        if list_open == []:
            finish_found = True
            path = []
            distances = []

    return map_open_closed, path, distances


def create_evaluation_matrix_v01(mapa, path, distances, list_finishes, symbol_wall=1, wall_value=10**9):
    """recieves mapa, path and distances, returns an evaluation matrix of shape=mapa.shape,
    where in each element is written the lowest distance of this element to some point of path + it's distance to finish
    This time, walls are taken into account -> matrix element MUST be in the clear line of sight of the path element
    requires "from skimage.draw import line"
    wall_value is the constant with which will be filled all elements that are "wall" according to mapa
    all coordinates in list_finishes will be set to 0
    """

    evaluation_matrix = np.empty(shape=mapa.shape)
    for row in range(mapa.shape[0]):
        for column in range(mapa.shape[1]):
            #If the element is wall, let's just put there an astronomical evaluation given by wall_value
            if mapa[row, column] == symbol_wall:
                evaluation_matrix[row, column] = wall_value
            else:
                #let's get a 1D array of 1/0 values whether a given path point is in LOS
                LOS_path = np.ones(shape=(len(path)))
                for index, (x, y) in enumerate(path):
                    LOS_points = np.array(line(row, column, x, y)).T
                    for (x2, y2) in LOS_points:
                        if mapa[x2, y2] == symbol_wall:
                            LOS_path[index] = 0
                            break

                #Now that we have LOS_path, we need to find the actual lowest valid distance
                evaluations = np.sqrt(np.sum((path - np.array([row, column]))**2, axis=1)) + distances
                #Let's change those elements of evaluations which are not in LOS to an astronomical value
                evaluations[LOS_path == 0] = wall_value  #can be any other constant than wall_value tho
                evaluation_matrix[row, column] = np.amin(evaluations)

    for (x, y) in list_finishes:
        evaluation_matrix[x, y] = 0
    return evaluation_matrix, wall_value


#Greedy search to find formula path
def tiles_to_move_to(current_position, vector, max_speed, eval_matrix):
    """returns np.array of 8 tiles surrounding the one it recieved + it's vector,
       if the value of the norm would be bigger than max_speed, it won't make it into the array
       (of the norm of the surrounding vector - current_position)
       all values need to be in bounds given by the shape of eval_matrix
    Args:
        tile (tuple of ints (x,y)): current position on the map
        vector (tuple of ints (x,y)): current speed vector
        max_speed (int): maximum value of the speed vector
        eval_matrix (2D np.ndarray): evaluation matrix of the position, here we only need it's shape
    Returns:
        array of coordinates of tiles according to above mentioned conditions
    """
    vector_tiles = np.array([[vector[0], vector[1] + 1], [vector[0], vector[1] - 1], [vector[0] - 1, vector[1]], [vector[0] + 1, vector[1]],
                             [vector[0] + 1, vector[1] + 1], [vector[0] + 1, vector[1] - 1], [vector[0] - 1, vector[1] + 1],
                             [vector[0] - 1, vector[1] - 1]])
    lower_than_norm = np.where(np.sqrt(vector_tiles[:, 0]**2 + vector_tiles[:, 1]**2) > max_speed)[0]
    vector_tiles = np.delete(vector_tiles, lower_than_norm, axis=0)
    array_of_tiles = vector_tiles + current_position[0:2]
    array_of_tiles = array_of_tiles.astype(int)

    #now we also need to delete those tiles which are out of bounds of mapa/eval_matrix
    out_of_bounds = np.where((array_of_tiles[:, 0] < 0) | (array_of_tiles[:, 0] >= eval_matrix.shape[0]) |
                             (array_of_tiles[:, 1] < 0) | (array_of_tiles[:, 1] >= eval_matrix.shape[1]))[0]
    array_of_tiles = np.delete(array_of_tiles, out_of_bounds, axis=0)
    return array_of_tiles


def formula_greedy_v03(start, eval_matrix, wall_value=10**9, max_speed=10):
    """Greedy search v03, uses evaluation matrix derived from create_evaluation_matrix_v01 (derived form astar),
       max_speed limits max norm of the speed vector

    Args:
        start (tuple of ints (x,y)): coordinates of start (chosen randomly from all possible starts before this algorithm)
        eval_matrix (2D np.ndarray): same shape as mapa, gives each element evaluation how "close" to finish it is.
        wall_value (int, optional): describes which value do elements in evaluation_matrix that correspond to "symbol_wall" in mapa have.
        max_speed (int, optional): describes what is the maximum norm of speed vector the car can reach.

    Depreciated Args:
        mapa (2D np.ndarray): draha, walls are marked with a symbol_wall
        symbol_wall (int, optional): number describing wall in mapa. Defaults to 1.
        finish (tuple of (x,y)): instead of finish, all possible finishes (list_of_finishes) are 0 in eval_matrix, if we reach any of these, it returns the path

    Returns:
        formula_path: list of (x,y) describing steps necessary to go from start to finish. Maybe gonna be np.ndarray
    """
    array_open = np.empty(shape=(0, 3), dtype=float)  # instead of list I will use array, hopefully it's faster
    map_open = np.zeros(shape=eval_matrix.shape, dtype=int)
    map_closed = np.zeros(shape=(eval_matrix.shape[0], eval_matrix.shape[1], max_speed*2, max_speed*2), dtype=int)
    map_vectors = np.zeros(shape=(eval_matrix.shape[0], eval_matrix.shape[1], 2), dtype=int)
    map_path_back = np.zeros(shape=(eval_matrix.shape[0], eval_matrix.shape[1], 2), dtype=int)
    map_steps = np.zeros(shape=eval_matrix.shape, dtype=int)

    # Lets initialize some necessary things:
    # nothing = 0, open = 1, closed positions are only in map_closed with their corresponding speed vectors
    map_open[tuple(start)] = 1
    # and I will look at that closed tile if and only if the vector leading to that tile differs
    array_open = np.append(array_open, np.array([[start[0], start[1], eval_matrix[start[0], start[1]]]]), axis=0)
    map_path_back[tuple(start)] = np.array(start)
    # map_vectors is already "zeros", that that is ok at the beginning.

    counter = 0

    finish_found = False
    while finish_found == False:

        indices = np.where(array_open[:, 2] == np.amin(array_open[:, 2]))[0]
        current_position = array_open[np.random.choice(indices)]

        #tile (x,y) corresponding to the current position, and corresponding vector. Named orig..original so as to not to confuse it with the expanded ones
        orig_tile = current_position[0:2]
        orig_tile = orig_tile.astype(int)
        orig_vector = map_vectors[tuple(current_position[0:2].astype(int))]

        cursed_position = 1  # 0 if at least one tile is opened, if still cursed after this for cycle, the orig_tile + vector will be closed

        for tile in tiles_to_move_to(orig_tile, orig_vector, max_speed, eval_matrix):

            tile = tile.astype(int)
            vector = tile - orig_tile
            vector = vector.astype(int)

            #Let's check of the vector is not going through a wall -> this checks even the tile itself
            #You could incorporate finish_found similarily (if we went right through the finish), but you gotta make sure you didn't go through wall
            in_LOS = 1
            vector_points = np.array(line(orig_tile[0], orig_tile[1], tile[0], tile[1])).T
            for (x2, y2) in vector_points:
                if eval_matrix[x2, y2] == wall_value:
                    in_LOS = 0
                    break

            if (in_LOS == 1 and map_open[tuple(tile)] != 1 and map_closed[tile[0], tile[1], vector[0], vector[1]] != 1):
                #eval_matrix[tuple(tile)] != wall_value and

                # Need to edit map_open/closed, array_open, map_path_back, map_vectors, map_steps
                map_open[tuple(tile)] = 1
                array_open = np.append(array_open, np.array([[tile[0], tile[1], eval_matrix[tile[0], tile[1]]]]), axis=0)
                map_path_back[tuple(tile)] = orig_tile
                map_vectors[tuple(tile)] = vector
                map_steps[tuple(tile)] = map_steps[tuple(orig_tile)] + 1

                cursed_position = 0

            # no need to check if it's a wall - wall tiles can never be open
            elif (map_open[tuple(tile)] == 1 and map_closed[tile[0], tile[1], vector[0], vector[1]] != 1 and in_LOS == 1):

                # if it works I can put this if into the previous elif
                if map_steps[tuple(tile)] > map_steps[tuple(orig_tile)] + 1:

                    # No need to change map_open/closed, only array_open
                    map_path_back[tuple(tile)] = orig_tile
                    map_vectors[tuple(tile)] = vector
                    map_steps[tuple(tile)] = map_steps[tuple(orig_tile)] + 1
                    array_open = np.append(array_open, np.array([[tile[0], tile[1], eval_matrix[tile[0], tile[1]]]]), axis=0)

                    cursed_position = 0

            # If we found the finish
            if eval_matrix[tuple(tile)] == 0:
                finish_found = True
                reconstructed = False
                path_backwards = [[tile[0], tile[1]]]
                while reconstructed == False:
                    next_tile = map_path_back[int(path_backwards[-1][0]), int(path_backwards[-1][1])].tolist()
                    path_backwards.append(next_tile)
                    if next_tile[0] == start[0] and next_tile[1] == start[1]:
                        reconstructed = True

        if cursed_position == 1:
            #map_closed[tile[0], tile[1], tile[0] - current_position[0], tile[1] - current_position[1]] = 1
            map_closed[orig_tile[0], orig_tile[1], orig_vector[0], orig_vector[1]] = 1
            index_cursed_position = np.where(np.all(array_open == current_position, axis=1))[0]
            array_open = np.delete(array_open, index_cursed_position, axis=0)

        counter += 1
        # If we did not find the finish -> no more tiles to expand -> array_open is empty
        if array_open.shape[0] == 0 or counter == 10000000:
            path_backwards = []
            break

    formula_path = path_backwards[::-1]

    return formula_path


#To display current situation on the racetrack
def display_rt(racetrack, fm_path_AI, fm_path, possible_tiles, current_tile):
    """Recieves internal formula_path and racetrack apod, returns figure to plot
    Args:
        racetracK(2D np.array): 2D map of the racetrack
        fm_path_AI(list of (x,y)):coordinates of the formula path for AI
        fm_path(list of (x,y)):coordinates of the formula path of the player
        possible_tiles(list of (x,y)):possible tiles the player can choose to move to with arrows
        current_tile(tuple (x,y)): the tile player is just on (from possible tiles)
    Returns:
        figure to plot
    """
    WALL_COLOR = np.array([0, 0, 0])
    RT_COLOR = np.array([255, 255, 255])
    START_COLOR = np.array([0, 255, 0])
    FINISH_COLOR = np.array([255, 0, 0])
    AI_LINE_COLOR = "red"  #np.array([100, 100, 0])
    PLAYER_LINE_COLOR = "yellow"  #np.array([0, 100, 100])
    PLAYER_CURRENT_LINE_COLOR = "orange"
    POSSIBLE_TILES_COLOR = "blue"  #np.array([200, 200, 0])
    CURRENT_TILE_COLOR = np.array([200, 100, 0])

    rgb_racetrack = np.zeros(shape=(racetrack.shape[0], racetrack.shape[1], 3), dtype=int)
    rgb_racetrack[racetrack == 0] = RT_COLOR
    rgb_racetrack[racetrack == 1] = WALL_COLOR  #kinda not necessary since it's already full of zeros
    rgb_racetrack[racetrack == 2] = START_COLOR
    rgb_racetrack[racetrack == 3] = FINISH_COLOR

    #Lets transpose fm_path and fm_path_AI (imshow shows transposed image, normal plt.plot does not, I work with it transposed)
    fm_path = np.array(fm_path)
    fm_path_AI = np.array(fm_path_AI)
    fm_path = np.concatenate((np.expand_dims(fm_path[:, 1], 1), np.expand_dims(fm_path[:, 0], 1)), axis=1)
    fm_path_AI = np.concatenate((np.expand_dims(fm_path_AI[:, 1], 1), np.expand_dims(fm_path_AI[:, 0], 1)), axis=1)

    #Lets make the figure. Now we need to display the path of the player and the AI, and the path the player might choose to do
    fig, ax = plt.subplots(figsize=(10, 10))
    J = ax.imshow(rgb_racetrack)

    ax.scatter([x[1] for x in possible_tiles], [y[0] for y in possible_tiles], marker=".", color="blue", alpha=0.5)

    ax.plot([fm_path[-1, 0], current_tile[1]], [fm_path[-1, 1], current_tile[0]], marker=".", color=PLAYER_CURRENT_LINE_COLOR, alpha=0.7)
    if len(fm_path) >= 2:
        for i in range(1, len(fm_path)):
            ax.plot([fm_path[i, 0], fm_path[i - 1, 0]], [fm_path[i, 1], fm_path[i - 1, 1]], marker=".", color=PLAYER_LINE_COLOR)
            if i < len(fm_path_AI):
                ax.plot([fm_path_AI[i, 0], fm_path_AI[i - 1, 0]], [fm_path_AI[i, 1], fm_path_AI[i - 1, 1]], marker=".", color=AI_LINE_COLOR, alpha=0.6)
    else:
            ax.plot([fm_path[0, 0], fm_path[0, 0]], [fm_path[0, 1], fm_path[0, 1]], marker=".", color=PLAYER_LINE_COLOR)
            ax.plot([fm_path_AI[0, 0], fm_path_AI[0, 0]], [fm_path_AI[0, 1], fm_path_AI[0, 1]], marker=".", color=AI_LINE_COLOR, alpha=0.6)
    return J


def move_current_tile(current_tile, possible_tiles, key):
    """Just moving that current_tile around with key presses
    possible_tiles MUST BE A LIST, NOT NDARRAY, OTHERWISE IT WONT FUNCTION!
    Args:
        current_tile(tuple of ints (x,y)): the tile where is player's cursor
        possible_tiles(list of tuples of ints (x,y)): those tiles where the player can move to
        key(str): "left", "right", "up", "down" -> which key player pressed
    Returns:
        new_current_tile -> only if it is a legal movement
    """
    new_current_tile = current_tile
    if key == "right":  #[0,1]
        if [current_tile[0], current_tile[1] + 1] in possible_tiles:
            new_current_tile = [current_tile[0], current_tile[1] + 1]
        elif [current_tile[0], current_tile[1] + 2] in possible_tiles:
            new_current_tile = [current_tile[0], current_tile[1] + 2]
    elif key == "left":  #[0,-1]
        if [current_tile[0], current_tile[1] - 1] in possible_tiles:
            new_current_tile = [current_tile[0], current_tile[1] - 1]
        elif [current_tile[0], current_tile[1] - 2] in possible_tiles:
            new_current_tile = [current_tile[0], current_tile[1] - 2]
    if key == "down":  #[1,0]
        if [current_tile[0] + 1, current_tile[1]] in possible_tiles:
            new_current_tile = [current_tile[0] + 1, current_tile[1]]
        elif [current_tile[0] + 2, current_tile[1]] in possible_tiles:
            new_current_tile = [current_tile[0] + 2, current_tile[1]]
    if key == "up":  #[-1,0]
        if [current_tile[0] - 1, current_tile[1]] in possible_tiles:
            new_current_tile = [current_tile[0] - 1, current_tile[1]]
        elif [current_tile[0] - 2, current_tile[1]] in possible_tiles:
            new_current_tile = [current_tile[0] - 2, current_tile[1]]

    return new_current_tile


def give_formula_path_v01(list_starts, eval_matrix, wall_value, max_speed):
    """formula_greedy_v03 sometimes fails to find a path when confronted with some very difficult racetracks.
    This function will ensure that we (hopefully) always recieve a path when confronted with any racetrack.
    We run formula_greedy_v03 for many epochs until solution is found and gradually reduce max_speed.
    Args:
        everything required by formula_greedy_v03
        list_starts so that we try a different starting tile each time
    Returns:
        formula_path
        final_speed
    """
    for speed in range(max_speed, 0, -1):
        for epoch in range(20):
            start = random.choice(list_starts)
            formula_path = formula_greedy_v03(start, eval_matrix=eval_matrix, wall_value=wall_value, max_speed=speed)
            if formula_path != []:
                break
        if formula_path != []:
            final_speed = speed
            break

    return formula_path, final_speed
