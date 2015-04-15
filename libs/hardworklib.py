# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:31:18 2014

@author: pedro.correia
"""

from __future__ import division
import random, time, pygame, sys
import pygcurse as pygcurse
from pygame.locals import *

def launch_pygcurses_textris():
    """
    This code aint ours. You can get it with pygcurse libaray at
    http://inventwithpython.com/pygcurse/
    Our deepest thanks guys...
    """
    FPS = 25
    WINDOWWIDTH = 26
    WINDOWHEIGHT = 27
    BOARDWIDTH = 10
    BOARDHEIGHT = 20
    BLANK = None
    
    LEFTMARGIN = 4
    TOPMARGIN = 4
    
    MOVESIDEWAYSFREQ = 0.15
    MOVEDOWNFREQ = 0.15
    
    
    #               R    G    B
    WHITE       = (255, 255, 255)
    GRAY        = (185, 185, 185)
    BLACK       = (  0,   0,   0)
    RED         = (155,   0,   0)
    GREEN       = (  0, 155,   0)
    BLUE        = (  0,   0, 155)
    YELLOW      = (155, 155,   0)
    
    BORDERCOLOR = BLUE
    BGCOLOR = BLACK
    TEXTCOLOR = WHITE
    COLORS = (BLUE, GREEN, RED, YELLOW)
    
    
    TEMPLATEWIDTH = 5
    TEMPLATEHEIGHT = 5
    
    S_PIECE_TEMPLATE = [['.....',
                         '.....',
                         '..OO.',
                         '.OO..',
                         '.....'],
                        ['.....',
                         '..O..',
                         '..OO.',
                         '...O.',
                         '.....']]
    
    Z_PIECE_TEMPLATE = [['.....',
                         '.....',
                         '.OO..',
                         '..OO.',
                         '.....'],
                        ['.....',
                         '..O..',
                         '.OO..',
                         '.O...',
                         '.....']]
    
    I_PIECE_TEMPLATE = [['..O..',
                         '..O..',
                         '..O..',
                         '..O..',
                         '.....'],
                        ['.....',
                         '.....',
                         'OOOO.',
                         '.....',
                         '.....']]
    
    O_PIECE_TEMPLATE = [['.....',
                         '.....',
                         '.OO..',
                         '.OO..',
                         '.....']]
    
    J_PIECE_TEMPLATE = [['.....',
                         '.O...',
                         '.OOO.',
                         '.....',
                         '.....'],
                        ['.....',
                         '..OO.',
                         '..O..',
                         '..O..',
                         '.....'],
                        ['.....',
                         '.....',
                         '.OOO.',
                         '...O.',
                         '.....'],
                        ['.....',
                         '..O..',
                         '..O..',
                         '.OO..',
                         '.....']]
    
    L_PIECE_TEMPLATE = [['.....',
                         '...O.',
                         '.OOO.',
                         '.....',
                         '.....'],
                        ['.....',
                         '..O..',
                         '..O..',
                         '..OO.',
                         '.....'],
                        ['.....',
                         '.....',
                         '.OOO.',
                         '.O...',
                         '.....'],
                        ['.....',
                         '.OO..',
                         '..O..',
                         '..O..',
                         '.....']]
    
    T_PIECE_TEMPLATE = [['.....',
                         '..O..',
                         '.OOO.',
                         '.....',
                         '.....'],
                        ['.....',
                         '..O..',
                         '..OO.',
                         '..O..',
                         '.....'],
                        ['.....',
                         '.....',
                         '.OOO.',
                         '..O..',
                         '.....'],
                        ['.....',
                         '..O..',
                         '.OO..',
                         '..O..',
                         '.....']]
    
    PIECES = {'S': S_PIECE_TEMPLATE,
              'Z': Z_PIECE_TEMPLATE,
              'J': J_PIECE_TEMPLATE,
              'L': L_PIECE_TEMPLATE,
              'I': I_PIECE_TEMPLATE,
              'O': O_PIECE_TEMPLATE,
              'T': T_PIECE_TEMPLATE}
    
    # Convert the shapes in PIECES from our text-friendly format above to a
    # programmer-friendly format.
    for p in PIECES: # loop through each piece
        for r in range(len(PIECES[p])): # loop through each rotation of the piece
            shapeData = []
            for x in range(TEMPLATEWIDTH): # loop through each column of the rotation of the piece
                column = []
                assert len(PIECES[p][r]) == TEMPLATEWIDTH, 'Malformed shape given for piece %s, rotation %s' % (p, r)
                for y in range(TEMPLATEHEIGHT): # loop through each character in the column of the rotation of the piece
                    assert len(PIECES[p][r][y]) == TEMPLATEHEIGHT, 'Malformed shape given for piece %s, rotation %s' % (p, r)
                    if PIECES[p][r][y][x] == '.':
                        column.append(BLANK)
                    else:
                        column.append(1)
                shapeData.append(column)
            PIECES[p][r] = shapeData
    
    
    def main():
        global FPSCLOCK, WINDOWSURF, BOARDBOX
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        WINDOWSURF = pygcurse.PygcurseWindow(WINDOWWIDTH, WINDOWHEIGHT, 'Textris', font=pygame.font.Font(None, 24))
        WINDOWSURF.autoupdate = False
        BOARDBOX = pygcurse.PygcurseTextbox(WINDOWSURF, (LEFTMARGIN-1, TOPMARGIN-1, BOARDWIDTH+2, BOARDHEIGHT+2))
    
        showTextScreen('Textris')
        while True: # main game loop
            #if random.randint(0, 1) == 0:
            #    pygame.mixer.music.load('tetrisb.mid')
            #else:
            #    pygame.mixer.music.load('tetrisc.mid')
            #pygame.mixer.music.play(-1, 0.0)
            runGame()
            #pygame.mixer.music.stop()
            showTextScreen('Game Over')
    
    
    def showTextScreen(text):
        # This function displays large text in the
        # center of the screen until a key is pressed.
        WINDOWSURF.setscreencolors('white', 'black', True)
        WINDOWSURF.cursor = WINDOWSURF.centerx - int(len(text)/2), WINDOWSURF.centery
        WINDOWSURF.write(str(text), fgcolor=WHITE)
    
        WINDOWSURF.cursor = WINDOWSURF.centerx - int(len('Press a key to continue.')/2), WINDOWSURF.centery + 4
        WINDOWSURF.write('Press a key to continue.', fgcolor='gray')
    
        while checkForKeyPress() == None:
            WINDOWSURF.update()
            FPSCLOCK.tick()
    
    
    def terminate():
        pygame.quit()
        #sys.exit()
    
    
    def runGame():
        # setup variables for the start of the game
        board = getNewBoard()
        lastMoveDownTime = time.time()
        lastMoveSidewaysTime = time.time()
        lastFallTime = time.time()
        movingDown = False # note: there is no movingUp variable
        movingLeft = False
        movingRight = False
        score = 0
        level, fallFreq = calculateLevelAndFallFreq(score)
    
        currentPiece = getNewPiece()
        nextPiece = getNewPiece()
    
        while True: # main game loop
            if currentPiece == None:
                # No current piece in play, so start a new piece at the top
                currentPiece = nextPiece
                nextPiece = getNewPiece()
                lastFallTime = time.time() # reset lastFallTime
    
                if not isValidPosition(board, currentPiece):
                    return # can't fit a new piece on the board, so game over
    
            checkForQuit()
            for event in pygame.event.get(): # event handling loop
                if event.type == KEYUP:
                    if (event.key == K_p):
                        # Pausing the game
                        WINDOWSURF.fill(BGCOLOR)
                        #pygame.mixer.music.stop()
                        showTextScreen('Paused') # pause until a key press
                        #pygame.mixer.music.play(-1, 0.0)
                        lastFallTime = time.time()
                        lastMoveDownTime = time.time()
                        lastMoveSidewaysTime = time.time()
                    elif (event.key == K_LEFT or event.key == K_a):
                        movingLeft = False
                    elif (event.key == K_RIGHT or event.key == K_d):
                        movingRight = False
                    elif (event.key == K_DOWN or event.key == K_s):
                        movingDown = False
    
                elif event.type == KEYDOWN:
                    # moving the block sideways
                    if (event.key == K_LEFT or event.key == K_a) and isValidPosition(board, currentPiece, adjX=-1):
                        currentPiece['x'] -= 1
                        lastMoveSidewaysTime = time.time()
                        movingLeft = True
                        movingRight = False
                        lastMoveSidewaysTime = time.time()
                    elif (event.key == K_RIGHT or event.key == K_d) and isValidPosition(board, currentPiece, adjX=1):
                        currentPiece['x'] += 1
                        movingRight = True
                        movingLeft = False
                        lastMoveSidewaysTime = time.time()
    
                    # rotating the block (if there is room to rotate)
                    elif (event.key == K_UP or event.key == K_w):
                        currentPiece['rotation'] = (currentPiece['rotation'] + 1) % len(PIECES[currentPiece['shape']])
                        if not isValidPosition(board, currentPiece):
                            currentPiece['rotation'] = (currentPiece['rotation'] - 1) % len(PIECES[currentPiece['shape']])
                    elif (event.key == K_q): # rotate the other direction
                        currentPiece['rotation'] = (currentPiece['rotation'] - 1) % len(PIECES[currentPiece['shape']])
                        if not isValidPosition(board, currentPiece):
                            currentPiece['rotation'] = (currentPiece['rotation'] + 1) % len(PIECES[currentPiece['shape']])
    
                    # making the block fall faster with the down key
                    elif (event.key == K_DOWN or event.key == K_s):
                        movingDown = True
                        if isValidPosition(board, currentPiece, adjY=1):
                            currentPiece['y'] += 1
                        lastMoveDownTime = time.time()
    
                    # move the current block all the way down
                    elif event.key == K_SPACE:
                        movingDown = False
                        movingLeft = False
                        movingRight = False
                        for i in range(1, BOARDHEIGHT):
                            if not isValidPosition(board, currentPiece, adjY=i):
                                break
                        currentPiece['y'] += (i-1)
    
            # handle moving the block because of user input
            if (movingLeft or movingRight) and time.time() - lastMoveSidewaysTime > MOVESIDEWAYSFREQ:
                if movingLeft and isValidPosition(board, currentPiece, adjX=-1):
                    currentPiece['x'] -= 1
                elif movingRight and isValidPosition(board, currentPiece, adjX=1):
                    currentPiece['x'] += 1
                lastMoveSidewaysTime = time.time()
    
            if movingDown and time.time() - lastMoveDownTime > MOVEDOWNFREQ and isValidPosition(board, currentPiece, adjY=1):
                currentPiece['y'] += 1
                lastMoveDownTime = time.time()
    
            # let the piece fall if it is time to fall
            if time.time() - lastFallTime > fallFreq:
                # see if the piece has hit the bottom
                if hasHitBottom(board, currentPiece):
                    # set the piece on the board, and update game info
                    addToBoard(board, currentPiece)
                    score += removeCompleteLines(board)
                    level, fallFreq = calculateLevelAndFallFreq(score)
                    currentPiece = None
                else:
                    # just move the block down
                    currentPiece['y'] += 1
                    lastFallTime = time.time()
    
            # drawing everything on the screen
            WINDOWSURF.setscreencolors(clear=True)
            drawBoard(board)
            drawStatus(score, level)
            drawNextPiece(nextPiece)
            if currentPiece != None:
                drawPiece(currentPiece)
    
            WINDOWSURF.update()
            FPSCLOCK.tick(FPS)
    
    
    def checkForQuit():
        for event in pygame.event.get(QUIT): # get all the QUIT events
            terminate() # terminate if any QUIT events are present
        for event in pygame.event.get(KEYUP): # get all the KEYUP events
            if event.key == K_ESCAPE:
                terminate() # terminate if the KEYUP event was for the Esc key
            pygame.event.post(event) # put the other KEYUP event objects back
    
    
    def calculateLevelAndFallFreq(score):
        # Based on the score, return the level the player is on and
        # how many seconds pass until a falling piece falls one space.
        level = int(score / 10) + 1
        return level, 0.27 - (level * 0.02)
    
    
    def getNewPiece():
        # return a random new piece in a random rotation and color
        shape = random.choice(list(PIECES.keys()))
        newPiece = {'shape': shape,
                    'rotation': random.randint(0, len(PIECES[shape]) - 1),
                    'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                    'y': -2, # start it above the board (i.e. less than 0)
                    'color': random.randint(0, len(COLORS)-1)}
        return newPiece
    
    
    def addToBoard(board, piece):
        # fill in the board based on piece's location, shape, and rotation
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[piece['shape']][piece['rotation']][x][y] != BLANK:
                    board[x + piece['x']][y + piece['y']] = piece['color']
    
    
    def checkForKeyPress():
        # Go through event queue looking for a KEYUP event.
        # Grab KEYDOWN events to remove them from the event queue.
        for event in pygame.event.get([KEYDOWN, KEYUP]):
            if event.type == KEYDOWN:
                continue
            if event.key == K_ESCAPE:
                terminate()
            return event.key
        return None
    
    
    def getNewBoard():
        # create and return a new blank board data structure
        board = []
        for i in range(BOARDWIDTH):
            board.append([BLANK] * BOARDHEIGHT)
        return board
    
    
    def hasHitBottom(board, piece):
        # Returns True if the piece's bottom is currently on top of something
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[piece['shape']][piece['rotation']][x][y] == BLANK or y + piece['y'] + 1 < 0:
                    continue # box is above the board or this box is blank
                if y + piece['y'] + 1 == BOARDHEIGHT:
                    return True # box is on bottom of the board
                if board[x + piece['x']][y + piece['y'] + 1] != BLANK:
                    return True # box is on top of another box on the board
        return False
    
    
    def isOnBoard(x, y):
        return x >= 0 and x < BOARDWIDTH and y < BOARDHEIGHT
    
    
    def isValidPosition(board, piece, adjX=0, adjY=0):
        # Return True if the piece is within the board and not colliding
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if y + piece['y'] + adjY < 0 or PIECES[piece['shape']][piece['rotation']][x][y] == BLANK:
                    continue
                if not isOnBoard(x + piece['x'] + adjX, y + piece['y'] + adjY):
                    return False
                if board[x + piece['x'] + adjX][y + piece['y'] + adjY] != BLANK:
                    return False
        return True
    
    
    def isCompleteLine(board, y):
        # Return True if the line filled with boxes with no gaps.
        for x in range(BOARDWIDTH):
            if board[x][y] == BLANK:
                return False
        return True
    
    
    def removeCompleteLines(board):
        # Remove any completed lines on the board, move everything above them down, and return the number of complete lines.
        numLinesRemoved = 0
        y = BOARDHEIGHT - 1 # start y at the bottom of the board
        while y >= 0:
            if isCompleteLine(board, y):
                # Remove the line and pull boxes down by one line.
                for pullDownY in range(y, 0, -1):
                    for x in range(BOARDWIDTH):
                        board[x][pullDownY] = board[x][pullDownY-1]
                # Set very top line to blank.
                for x in range(BOARDWIDTH):
                    board[x][0] = BLANK
                numLinesRemoved += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y -= 1 # move on to check next row up
        return numLinesRemoved
    
    
    def drawBoard(board):
        # draw the border around the board
        BOARDBOX.update()
    
        # draw the individual boxes on the board
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                drawBox(x, y, board[x][y])
    
    
    def drawBox(x, y, color, cellx=None, celly=None):
        # draw a single box (each tetromino piece has four boxes)
        # at xy coordinates on the board. Or, if cellx & celly
        # are specified, draw to the pixel coordinates stored in
        # cellx & celly (this is used for the "Next" piece.)
        if color == BLANK:
            return
        if cellx is None and celly is None:
            cellx, celly = x + LEFTMARGIN, y + TOPMARGIN
        WINDOWSURF.putchar('O', x=cellx, y=celly, fgcolor=COLORS[color], bgcolor='black')
    
    
    def drawStatus(score, level):
        # draw the score text
        WINDOWSURF.putchars('Score:', x=WINDOWWIDTH-8, y=2, fgcolor='gray', bgcolor='black')
        WINDOWSURF.putchars(str(score), x=WINDOWWIDTH-7, y=3, fgcolor='white', bgcolor='black')
        # draw the level text
        WINDOWSURF.putchars('Level:', x=WINDOWWIDTH-8, y=5, fgcolor='gray', bgcolor='black')
        WINDOWSURF.putchars(str(level), x=WINDOWWIDTH-7, y=6, fgcolor='white', bgcolor='black')
    
    def drawNextPiece(piece):
        # draw the "next" text
        WINDOWSURF.putchars('Next:', x=WINDOWWIDTH-8, y=8, fgcolor='gray', bgcolor='black')
    
        # draw the "next" piece
        drawPiece(piece, cellx=WINDOWWIDTH-8, celly=9)
    
    
    def drawPiece(piece, cellx=None, celly=None):
        shapeToDraw = PIECES[piece['shape']][piece['rotation']]
        if cellx == None and celly == None:
            # if cellx & celly hasn't been specified, use the location stored in the piece data structure
            cellx, celly = piece['x'] + LEFTMARGIN, piece['y'] + TOPMARGIN
    
        # draw each of the blocks that make up the piece
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if shapeToDraw[x][y] != BLANK:
                    drawBox(None, None, piece['color'], cellx + x, celly + y)
                    
    main()
    
def launch_pygcurse_dodger():
    """
    This code aint ours. You can get it with pygcurse libaray at
    http://inventwithpython.com/pygcurse/
    Our deepest thanks guys...
    """
    GREEN = (0, 255, 0)
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    RED = (255,0,0)
    
    WINWIDTH = 40
    WINHEIGHT = 50
    TEXTCOLOR = WHITE
    BACKGROUNDCOLOR = (0, 0, 0)
    FPS = 40
    BADDIEMINSIZE = 1
    BADDIEMAXSIZE = 5
    BADDIEMINSPEED = 4
    BADDIEMAXSPEED = 1
    ADDNEWBADDIERATE = 3
    
    win = pygcurse.PygcurseWindow(WINWIDTH, WINHEIGHT, fullscreen=False)
    pygame.display.set_caption('Pygcurse Dodger')
    win.autoupdate = False
    
    def main():
        showStartScreen()
        pygame.mouse.set_visible(False)
        mainClock = pygame.time.Clock()
        gameOver = False
    
        newGame = True
        while True:
            if gameOver and time.time() - 4 > gameOverTime:
                newGame = True
            if newGame:
                newGame = False
                pygame.mouse.set_pos(win.centerx * win.cellwidth, (win.bottom - 4) * win.cellheight)
                mousex, mousey = pygame.mouse.get_pos()
                cellx, celly = win.getcoordinatesatpixel(mousex, mousey)
                baddies = []
                baddieAddCounter = 0
                gameOver = False
                score = 0
    
            win.fill(bgcolor=BLACK)
    
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    terminate()
                if event.type == MOUSEMOTION and not gameOver:
                    mousex, mousey = event.pos
                    cellx, celly = win.getcoordinatesatpixel(mousex, mousey)
    
            # add new baddies if needed
            if baddieAddCounter == ADDNEWBADDIERATE:
                speed = random.randint(BADDIEMAXSPEED, BADDIEMINSPEED)
                baddies.append({'size': random.randint(BADDIEMINSIZE, BADDIEMAXSIZE),
                                'speed': speed,
                                'x': random.randint(0, win.width),
                                'y': -BADDIEMAXSIZE,
                                'movecounter': speed})
                baddieAddCounter = 0
            else:
                baddieAddCounter += 1
    
    
            # move baddies down, remove if needed
            for i in range(len(baddies)-1, -1, -1):
                if baddies[i]['movecounter'] == 0:
                    baddies[i]['y'] += 1
                    baddies[i]['movecounter'] = baddies[i]['speed']
                else:
                    baddies[i]['movecounter'] -= 1
    
                if baddies[i]['y'] > win.height:
                    del baddies[i]
    
    
            # check if hit
            if not gameOver:
                for baddie in baddies:
                    if cellx >= baddie['x'] and celly >= baddie['y'] and cellx < baddie['x']+baddie['size'] and celly < baddie['y']+baddie['size']:
                        gameOver = True
                        gameOverTime = time.time()
                        break
                score += 1
    
            # draw baddies to screen
            for baddie in baddies:
                win.fill('#', GREEN, BLACK, (baddie['x'], baddie['y'], baddie['size'], baddie['size']))
    
            if not gameOver:
                playercolor = WHITE
            else:
                playercolor = RED
                win.putchars('GAME OVER', win.centerx-4, win.centery, fgcolor=RED, bgcolor=BLACK)
    
            win.putchar('@', cellx, celly, playercolor)
            win.putchars('Score: %s' % (score), win.width - 14, 1, fgcolor=WHITE)
            win.update()
            mainClock.tick(FPS)
    
    
    def showStartScreen():
        while checkForKeyPress() is None:
            win.fill(bgcolor=BLACK)
            win.putchars('Pygcurse Dodger', win.centerx-8, win.centery, fgcolor=TEXTCOLOR)
            if int(time.time() * 2) % 2 == 0: # flashing
                win.putchars('Press a key to start!', win.centerx-11, win.centery+1, fgcolor=TEXTCOLOR)
            win.update()
    
    
    def checkForKeyPress():
        # Go through event queue looking for a KEYUP event.
        # Grab KEYDOWN events to remove them from the event queue.
        for event in pygame.event.get([KEYDOWN, KEYUP]):
            if event.type == KEYDOWN:
                continue
            if event.key == K_ESCAPE:
                terminate()
            return event.key
        return None
    
    
    def terminate():
        pygame.quit()
        #sys.exit()
    
    main()
    
def launch_pygcurse_maze():
    """
    This code aint ours. You can get it with pygcurse libaray at
    http://inventwithpython.com/pygcurse/
    Our deepest thanks guys...
    """
    BLUE = (0, 0, 128)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLACK = (0,0,0)
    RED = (255,0,0)
    
    MAZE_WIDTH  = 41
    MAZE_HEIGHT = 41
    FPS = 40
    
    win = pygcurse.PygcurseWindow(MAZE_WIDTH, MAZE_HEIGHT, fullscreen=False)
    pygame.display.set_caption('Pygcurse Maze')
    win.autowindowupdate = False
    win.autoupdate = False
    
    
    class JoeWingMaze():
        # Maze generator in Python
        # Joe Wingbermuehle
        # 2010-10-06
        # http://joewing.net/programs/games/python/maze.py
    
        def __init__(self, width=21, height=21):
            if width % 2 == 0:
                width += 1
            if height % 2 == 0:
                height += 1
    
            # The size of the maze (must be odd).
            self.width  = width
            self.height = height
    
            # The maze.
            self.maze = dict()
    
            # Generate and display a random maze.
            self.init_maze()
            self.generate_maze()
            #self.display_maze() # prints out the maze to stdout
    
        # Display the maze.
        def display_maze(self):
           for y in range(0, self.height):
              for x in range(0, self.width):
                 if self.maze[x][y] == 0:
                    sys.stdout.write(" ")
                 else:
                    sys.stdout.write("#")
              sys.stdout.write("\n")
    
        # Initialize the maze.
        def init_maze(self):
           for x in range(0, self.width):
              self.maze[x] = dict()
              for y in range(0, self.height):
                 self.maze[x][y] = 1
    
        # Carve the maze starting at x, y.
        def carve_maze(self, x, y):
           dir = random.randint(0, 3)
           count = 0
           while count < 4:
              dx = 0
              dy = 0
              if   dir == 0:
                 dx = 1
              elif dir == 1:
                 dy = 1
              elif dir == 2:
                 dx = -1
              else:
                 dy = -1
              x1 = x + dx
              y1 = y + dy
              x2 = x1 + dx
              y2 = y1 + dy
              if x2 > 0 and x2 < self.width and y2 > 0 and y2 < self.height:
                 if self.maze[x1][y1] == 1 and self.maze[x2][y2] == 1:
                    self.maze[x1][y1] = 0
                    self.maze[x2][y2] = 0
                    self.carve_maze(x2, y2)
              count = count + 1
              dir = (dir + 1) % 4
    
        # Generate the maze.
        def generate_maze(self):
           random.seed()
           #self.maze[1][1] = 0
           self.carve_maze(1, 1)
           #self.maze[1][0] = 0
           #self.maze[self.width - 2][self.height - 1] = 0
    
           # maze generator modified to have randomly placed entrance/exit.
           startx = starty = endx = endy = 0
           while self.maze[startx][starty]:
               startx = random.randint(1, self.width-2)
               starty = random.randint(1, self.height-2)
           while self.maze[endx][endy] or endx == 0 or abs(startx - endx) < int(self.width / 3) or abs(starty - endy) < int(self.height / 3):
               endx = random.randint(1, self.width-2)
               endy = random.randint(1, self.height-2)
    
           self.maze[startx][starty] = 0
           self.maze[endx][endy] = 0
    
           self.startx = startx
           self.starty = starty
           self.endx = endx
           self.endy = endy
    
    
    
    
    
    def main():
        newGame = True
        solved = False
        moveLeft = moveRight = moveUp = moveDown = False
        lastmovetime = sys.maxsize
        mainClock = pygame.time.Clock()
        while True:
            if newGame:
                newGame = False # if you want to see something cool, change the False to True
                jwmaze = JoeWingMaze(MAZE_WIDTH, MAZE_HEIGHT)
                maze = jwmaze.maze
                solved = False
                playerx, playery = jwmaze.startx, jwmaze.starty
                endx, endy = jwmaze.endx, jwmaze.endy
                breadcrumbs = {}
    
            if (playerx, playery) not in breadcrumbs:
                breadcrumbs[(playerx, playery)] = True
    
            # handle input
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    #sys.exit()
    
                elif event.type == KEYDOWN:
                    if solved or event.key == K_BACKSPACE:
                        newGame = True
                    elif event.key == K_ESCAPE:
                        pygame.quit()
                        #sys.exit()
                    elif event.key == K_UP:
                        moveUp = True
                        moveDown = False
                    elif event.key == K_DOWN:
                        moveDown = True
                        moveUp = False
                    elif event.key == K_LEFT:
                        moveLeft = True
                        moveRight = False
                    elif event.key == K_RIGHT:
                        moveRight = True
                        moveLeft = False
                    lastmovetime = time.time() - 1
    
                elif event.type == KEYUP:
                    if event.key == K_UP:
                        moveUp = False
                    elif event.key == K_DOWN:
                        moveDown = False
                    elif event.key == K_LEFT:
                        moveLeft = False
                    elif event.key == K_RIGHT:
                        moveRight = False
    
            # move the player (if allowed)
            if time.time() - 0.05 > lastmovetime:
                if moveUp and isOnBoard(playerx, playery-1) and maze[playerx][playery-1] == 0:
                    playery -= 1
                elif moveDown and isOnBoard(playerx, playery+1) and maze[playerx][playery+1] == 0:
                    playery += 1
                elif moveLeft and isOnBoard(playerx-1, playery) and maze[playerx-1][playery] == 0:
                    playerx -= 1
                elif moveRight and isOnBoard(playerx+1, playery) and maze[playerx+1][playery] == 0:
                    playerx += 1
    
                lastmovetime = time.time()
                if playerx == endx and playery == endy:
                    solved = True
    
            # display maze
            drawMaze(win, maze, breadcrumbs)
            if solved:
                win.cursor = (win.centerx - 4, win.centery)
                win.write('Solved!', fgcolor=YELLOW, bgcolor=RED)
                moveLeft = moveRight = moveUp = moveDown = False
            win.putchar('@', playerx, playery, RED, BLACK)
            win.putchar('O', jwmaze.endx, jwmaze.endy, GREEN, BLACK)
            win.update()
            pygame.display.update()
            mainClock.tick(FPS)
    
    
    def isOnBoard(x, y):
        return x >= 0 and y >= 0 and x < MAZE_WIDTH and y < MAZE_HEIGHT
    
    
    def drawMaze(win, maze, breadcrumbs):
        for x in range(MAZE_WIDTH):
            for y in range(MAZE_HEIGHT):
                if maze[x][y] != 0:
                    win.paint(x, y, BLUE)
                else:
                    win.paint(x, y, BLACK)
                if (x, y) in breadcrumbs:
                    win.putchar('.', x, y, RED, BLACK)
    
    main()
