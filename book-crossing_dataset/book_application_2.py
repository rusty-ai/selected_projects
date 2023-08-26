import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
import pyglet

import urllib.request
import os
import matplotlib.pyplot as plt
from skimage.transform import resize

evaluation_matrix = pd.read_pickle("my_dataframes/evaluation_matrix.pkl")
df_users_reduced = pd.read_pickle("my_dataframes/df_users_reduced.pkl")
df_books_reduced = pd.read_pickle("my_dataframes/df_books_reduced.pkl")


########################################################################
# Variables and constants
########################################################################

# STATE describes in which mode the application is now - whether in the mode where we input what we wish to find or in the mode when results are being shown
STATE = "MENU" # can be also "RESULTS"
# BOOK_NAME is what the user typed in so far when in MENU when selecting the book
BOOK_NAME = ""
# SUGGESTED_BOOK is used for autocompleting BOOK_NAME when selecting the book. When None instead of str, it means no book can be suggested. 
SUGGESTED_BOOK = None
# MENU_SELECTION is for when we are in MENU, so that we can select between what we wish to type in
MENU_SELECTION = 0 # 0 ... "BOOK_NAME", 1 ... "AGE", 2 ... "COUNTRY"
# AGE is for inputing the Age in MENU
AGE = ""
# COUNTRY is for inputing the country in MENU
COUNTRY = ""
# SUGGESTED_COUONTRY is for autocompleting COUNTRY when selecting the country. When None instead of str, it means no country can be suggested. 
SUGGESTED_COUNTRY = None
# NUM_PRED is the number of books we want the model to recommend (-1, because it always recommends the book we look for itself)
NUM_PRED = 5
# BOOK_TO_SHOW is the number denoting which book from the list of (NUM_PRED - 1) books we want to currently show, when in RESULTS
BOOK_TO_SHOW = 0


key_press_sound = pyglet.media.load('resources/key_press_sound.wav', streaming=False)

window = pyglet.window.Window(1280, 720, "Book Recommendation Model")

########################################################################
# Pyglet - what happens on text
########################################################################

@window.event
def on_text(text):
    global STATE, BOOK_NAME, MENU_SELECTION, SUGGESTED_BOOK, AGE, COUNTRY, SUGGESTED_COUNTRY, BOOK_TO_SHOW

    if STATE == "MENU" and MENU_SELECTION == 0 and text != ("\n") and text != ("\r"):  #pressing ENTER == "\r", "\n" is unnecessary maybe
        BOOK_NAME = BOOK_NAME + text

        # Now, let's find books that begin with those letters (we could search a tree instead of that dataframe, but the speed difference won't be noticeable)
        def find_beginning_string(text, beginning_string=BOOK_NAME):
            """If the input text begins with beginning_string, returns True, else False"""
            return text[:len(beginning_string)] == beginning_string

        suggested_books = df_books_reduced[df_books_reduced["Book-Title"].apply(find_beginning_string)]
        if len(suggested_books) != 0:
            SUGGESTED_BOOK = suggested_books.iloc[0]["Book-Title"]
        else:
            SUGGESTED_BOOK = None
        
    if STATE == "MENU" and MENU_SELECTION == 1 and text in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] and len(AGE) < 2:
        AGE = AGE + text

    if STATE == "MENU" and MENU_SELECTION == 2 and text != ("\n") and text != ("\r"):
        COUNTRY = COUNTRY + text

        # Now, let's find countries that begin with those letters
        def find_beginning_string(text, beginning_string=COUNTRY):
            """If the input text begins with beginning_string, returns True, else False"""
            return text[:len(beginning_string)] == beginning_string

        suggested_countries = df_users_reduced[df_users_reduced["Country"].apply(find_beginning_string)]
        if len(suggested_countries) != 0:
            SUGGESTED_COUNTRY = suggested_countries.iloc[0]["Country"]
        else:
            SUGGESTED_COUNTRY = None


########################################################################
# Pyglet - what happens on key press
########################################################################

@window.event
def on_key_press(key_code, modifier):
    global STATE, BOOK_NAME, MENU_SELECTION, SUGGESTED_BOOK, AGE, COUNTRY, SUGGESTED_COUNTRY, BOOK_TO_SHOW

    # Changing MENU_SELECTION - changing input between BOOK_NAME, AGE, COUNTRY
    if STATE == "MENU":
        if key_code == pyglet.window.key.UP:
            MENU_SELECTION = np.mod(MENU_SELECTION - 1, 3)
            key_press_sound.play()

        if key_code == pyglet.window.key.DOWN:
            MENU_SELECTION = np.mod(MENU_SELECTION + 1, 3)
            key_press_sound.play()

    # Show the RESULTS or come back to MENU and erase all inputs
    if key_code == pyglet.window.key.ENTER:
        if STATE == "MENU" and SUGGESTED_BOOK != None:
            STATE = "RESULTS"
            key_press_sound.play()

        elif STATE == "RESULTS":
            STATE = "MENU"
            BOOK_NAME, AGE, COUNTRY = "", "", ""
            SUGGESTED_BOOK, SUGGESTED_COUNTRY, MENU_SELECTION = None, None, 0
            key_press_sound.play()

    # Deleting characters of BOOK_NAME, AGE, COUNTRY
    elif STATE == "MENU" and key_code == pyglet.window.key.BACKSPACE:
        if MENU_SELECTION == 0:
            BOOK_NAME = BOOK_NAME[:-1]

            def find_beginning_string(text, beginning_string=BOOK_NAME):
                """If the input text begins with beginning_string, returns True, else False"""
                return text[:len(beginning_string)] == beginning_string

            suggested_books = df_books_reduced[df_books_reduced["Book-Title"].apply(find_beginning_string)]
            if len(suggested_books) != 0:
                SUGGESTED_BOOK = suggested_books.iloc[0]["Book-Title"]
            else:
                SUGGESTED_BOOK = None

        if MENU_SELECTION == 1:
            AGE = AGE[:-1]

        if MENU_SELECTION == 2:
            COUNTRY = COUNTRY[:-1]

            # Unlike when editing SUGGESTED_BOOK when BACKSPACE is pressed, we need to make sure that if COUNTRY = "", SUGGESTED_COUNTRY = None
            if len(COUNTRY) == 0:
                SUGGESTED_COUNTRY = None
            else:
                def find_beginning_string(text, beginning_string=COUNTRY):
                    """If the input text begins with beginning_string, returns True, else False"""
                    return text[:len(beginning_string)] == beginning_string

                suggested_countries = df_users_reduced[df_users_reduced["Country"].apply(find_beginning_string)]
                if len(suggested_countries) != 0:
                    SUGGESTED_COUNTRY = suggested_countries.iloc[0]["Country"]
                else:
                    SUGGESTED_COUNTRY = None

    # When it RESULTS, change which book is displayed
    if STATE == "RESULTS" and key_code == pyglet.window.key.RIGHT:
        BOOK_TO_SHOW = np.mod(BOOK_TO_SHOW + 1, NUM_PRED - 1)
        key_press_sound.play()
    if STATE == "RESULTS" and key_code == pyglet.window.key.LEFT:
        BOOK_TO_SHOW = np.mod(BOOK_TO_SHOW - 1, NUM_PRED - 1)
        key_press_sound.play()


########################################################################
# Pyglet - what is being drawn
########################################################################

@window.event
def on_draw():
    global STATE, BOOK_NAME, SUGGESTED_BOOK, AGE, COUNTRY, SUGGESTED_COUNTRY, BOOK_TO_SHOW
    window.clear()
    pyglet.shapes.Rectangle(x=0, y=0, width=1280, height=720, color=(200, 200, 200)).draw() # background color

    if STATE == "MENU":

        # BOOK_NAME label and background

        if SUGGESTED_BOOK == None and MENU_SELECTION == 0:
            label_book = "<font face='Calibri'><b> Title: </b><font color='red'>" + BOOK_NAME + "</font>|"
        elif SUGGESTED_BOOK == None and MENU_SELECTION != 0:
            label_book = "<font face='Calibri'><b> Title: </b><font color='red'>" + BOOK_NAME + "</font>"
        elif SUGGESTED_BOOK != None and MENU_SELECTION == 0:
            label_book = "<font face='Calibri'><b> Title:</b> " + BOOK_NAME + "|<i>" + SUGGESTED_BOOK[len(BOOK_NAME):] + "<i></font>"
        else:
            label_book = "<font face='Calibri'><b> Title:</b> " + BOOK_NAME + "<i>" + SUGGESTED_BOOK[len(BOOK_NAME):] + "<i></font>"

        pyglet.shapes.Rectangle(x=window.width//2 - 200, y=window.height//2 + 100, width=600, height=30, color=(180, 180, 180)).draw() # color under the text input
        pyglet.text.HTMLLabel(label_book, x=window.width//2 - 190, y=window.height//2 + 108).draw()

        # AGE label and background

        if MENU_SELECTION == 1:
            label_age = "<font face='Calibri'><b>Age:</b> " + AGE + "|</font>"
        else:
            label_age = "<font face='Calibri'><b>Age:</b> " + AGE + "</font>"

        pyglet.shapes.Rectangle(x=window.width//2 - 200, y=window.height//2 + 30, width=70, height=30, color=(180, 180, 180)).draw() # color under the age input
        pyglet.text.HTMLLabel(label_age, x=window.width//2 - 190, y=window.height//2 + 38).draw()

        # COUNTRY label and background

        if SUGGESTED_COUNTRY == None and MENU_SELECTION == 2:
            label_country = "<font face='Calibri'><b>Country: </b><font color='red'>" + COUNTRY + "</font>|"
        elif SUGGESTED_COUNTRY == None and MENU_SELECTION != 2:
            label_country = "<font face='Calibri'><b>Country: </b><font color='red'>" + COUNTRY + "</font>"
        elif SUGGESTED_COUNTRY != None and MENU_SELECTION == 2:
            label_country = "<font face='Calibri'><b>Country: </b>" + COUNTRY + "|<i>" + SUGGESTED_COUNTRY[len(COUNTRY):] + "<i></font>"
        else:
            label_country = "<font face='Calibri'><b>Country: </b>" + COUNTRY + "<i>" + SUGGESTED_COUNTRY[len(COUNTRY):] + "<i></font>"

        pyglet.shapes.Rectangle(x=window.width//2 - 200, y=window.height//2 - 40, width=200, height=30, color=(180, 180, 180)).draw() # color under the country input
        pyglet.text.HTMLLabel(label_country, x=window.width//2 - 190, y=window.height//2 - 32).draw()

    if STATE == "RESULTS":

        pyglet.shapes.Rectangle(x=window.width//2 - 200, y=window.height//2 + 100, width=600, height=30, color=(180, 180, 180)).draw() # box under the SUGGESTED_BOOK
        label_book = "<font face='Calibri'><b>Title: </b>" + SUGGESTED_BOOK + "</font>"
        pyglet.text.HTMLLabel(label_book, x=window.width//2 - 190, y=window.height//2 + 108).draw()

        pyglet.shapes.Rectangle(x=window.width//2 - 200, y=window.height//2 - 230, width=600, height=300, color=(180, 180, 180)).draw() # box under the results

        # First, let's find those columns that, according to AGE and COUNTRY, should be evaluated with a higher importance
        target_columns = np.zeros(shape=(len(evaluation_matrix.columns),))
        if AGE != "" and SUGGESTED_COUNTRY != None:
            target_users = df_users_reduced[((df_users_reduced["Age"] <= int(AGE) + 5) & 
                                             (df_users_reduced["Age"] >= int(AGE) - 5)) | (df_users_reduced["Country"] == SUGGESTED_COUNTRY)]
            target_columns = evaluation_matrix.columns.isin(target_users["User-ID"])
        elif AGE != "" and SUGGESTED_COUNTRY == None:
            target_users = df_users_reduced[((df_users_reduced["Age"] <= int(AGE) + 5) & (df_users_reduced["Age"] >= int(AGE) - 5))]
            target_columns = evaluation_matrix.columns.isin(target_users["User-ID"])
        elif AGE == "" and SUGGESTED_COUNTRY != None:
            target_users = df_users_reduced[(df_users_reduced["Country"] == SUGGESTED_COUNTRY)]
            target_columns = evaluation_matrix.columns.isin(target_users["User-ID"])


        # Now let's use KNN to get those recommendations
        def custom_metric(v1, v2, target_columns=target_columns, factor=3, factor_2=15):
            """
            Used when evaluating the nearest neighbors. Using a custom one because we want to increase the influence of chosen users (AGE, COUNTRY)
            The metric used will be Minkowski for p=2 (Euclidean)
            We also want to increase the influence of users that rated this book (NON-zero values in v1) drastically, that is done using factor_2
            Recieves: 
                v1, v2          ... vectors from the evaluation_matrix space
                target_columns  ... columns we want to increase the influence of ()
                factor          ... the factor by which will the NON-target columns will be multiplied
                factor_2        ... this number multiplies NON-zero values in v1
            Returns:
                the weighted distance between v1 and v2
            """
            weight_vector = 1 + (factor - 1) * (1 - target_columns)
            weight_vector_2 = 1 + (factor_2 - 1) * np.array([v1 != 0.])
            return np.sqrt(np.sum(np.abs(v1 - v2)**2 * weight_vector * weight_vector_2)) # tbh np.sqrt is not needed, but let's keep it consistent

        model = neighbors.NearestNeighbors(n_neighbors=NUM_PRED, algorithm="brute", metric=custom_metric)
        model.fit(evaluation_matrix)

        # Let's find results for the evaluation_matrix row corresponding to the book name
        corresponding_ISBN = df_books_reduced[df_books_reduced["Book-Title"] == SUGGESTED_BOOK]["ISBN"].iloc[0]
        prediction = model.kneighbors(evaluation_matrix.loc[[corresponding_ISBN]])

        # Now we have the distances and the vector indices, let's find the corresponding Book-Titles
        def get_prediction_names(matrix_indices, evaluation_matrix=evaluation_matrix, df_books=df_books_reduced):
            """
            Recieves:
                a list of indices of nearest neighbors of the input vector (including the input vector index itself)
                evaluation_matrix - to get "ISBN" corresponding to the indices
                df_books dataframe - to get "Book-Title corresponding to those "ISBN"s
            Returns a dataframe with names of those books
            """
            isbns = evaluation_matrix.iloc[matrix_indices].index
            return df_books[df_books["ISBN"].isin(isbns)] # ["Book-Title"].tolist()

        df_recommendations = get_prediction_names(prediction[1][0][1:])
        pyglet.text.HTMLLabel("<font face='Calibri'>" + df_recommendations["Book-Title"].tolist()[BOOK_TO_SHOW], 
                              x=window.width//2 - 190, y=window.height//2 - 222 + 270).draw()
        pyglet.text.HTMLLabel("<font face='Calibri'><b>Book Author: </b><i>" + df_recommendations["Book-Author"].tolist()[BOOK_TO_SHOW], 
                              x=window.width//2 - 190, y=window.height//2 - 222 + 270 - 30).draw()
        pyglet.text.HTMLLabel("<font face='Calibri'><b>Year of publication: </b><i>" + str(int(df_recommendations["Year-Of-Publication"].tolist()[BOOK_TO_SHOW])), 
                              x=window.width//2 - 190, y=window.height//2 - 222 + 270 - 60).draw()


        SHOW_IMG = True
        if SHOW_IMG == True:

            # Seems like we need to build a custom opener to access images on Amazon servers
            opener=urllib.request.build_opener()
            opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            urllib.request.install_opener(opener)

            urllib.request.urlretrieve(df_recommendations["Image-URL-L"].tolist()[BOOK_TO_SHOW], "resources/book_image.jpg")

            if os.path.exists("resources/book_image.jpg"):

                # Now, let's load it, resize it and display it
                book_image = pyglet.resource.image("resources/book_image.jpg")
                book_image.width = 120
                book_image.height = 200
                book_image.blit(x=window.width//2 - 190, y=window.height//2 - 220, z=0)


pyglet.app.run()