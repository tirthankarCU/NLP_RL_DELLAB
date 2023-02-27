import numpy as np
import pygame 

pygame.init()
'''
Global Variables.
'''
MX_NO_OF_DIGITS=3
FPS=30
rm=0
WIDTH, HEIGHT = 900, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NLP_RL_GAME")
constructArrElement=[[] for _ in range(MX_NO_OF_DIGITS)]
'''
COLORS.
'''
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 225)
GREEN = (0,255,0)
YELLOW = (200, 200, 0)
CYAN = (0,200,200)
MAGENTA = (200,0,200)
NEUTRAL = (128,128,128)
'''
BORDERS
'''
width_do,height_do=450,150
DIGIT_OUTER=pygame.Rect(WIDTH/2-width_do/2,HEIGHT/2-height_do/2,width_do,height_do)
width_di,height_di=100,100
'''
FONT
'''
FONT_SIZE=50
FONT = pygame.font.SysFont('Arial',FONT_SIZE)
'''
CONSTRUCTION_BOX
'''
w_b,h_b=32,32
w_m,h_m=16,16
w_s,h_s=8,8
'''
CONSTRUCT BOX
'''
w_cb,h_cb=100,200
big_block=[];medium_block=[];small_block=[]

class RectNode:
    def __init__(self,x,y,width,height,isEmpty) -> None:
        self.rect=pygame.Rect(x,y,width,height)
        self.isEmpty=isEmpty
    
def drawRectangleBorder(rect_,color,width):
    a=(rect_.left,rect_.top)
    b=(rect_.left+rect_.width-width,rect_.top)
    c=(rect_.left,rect_.top+rect_.height-width)
    pygame.draw.rect(WIN,color,pygame.Rect(a[0],a[1],rect_.width,width))
    pygame.draw.rect(WIN,color,pygame.Rect(a[0],a[1],width,rect_.height))
    pygame.draw.rect(WIN,color,pygame.Rect(b[0],b[1],width,rect_.height))
    pygame.draw.rect(WIN,color,pygame.Rect(c[0],c[1],rect_.width,width))

def drawWindowOneTime(no_list):
    global constructArrElement,big_block,medium_block,small_block
    WIN.fill(WHITE)
    drawRectangleBorder(DIGIT_OUTER,BLACK,2)
    gap_x=(width_do/3-width_di)/2
    gap_y=(height_do-height_di)/2
    rectArr=[]
    rx,ry=WIDTH/2-width_do/2,HEIGHT/2-height_do/2
    for _ in range(MX_NO_OF_DIGITS):
        digit_inner=pygame.Rect(rx+gap_x,ry+gap_y,width_di,height_di)
        rectArr.append(digit_inner)
        drawRectangleBorder(digit_inner,BLUE,2)
        rx+=2*gap_x+width_di
    f_indx=0
    for indx in range(len(rectArr)-1,-1,-1):
        x_co=rectArr[indx][0]+(rectArr[indx].width-FONT_SIZE/2)/2
        y_co=rectArr[indx][1]+(rectArr[indx].height-FONT_SIZE)/2
        WIN.blit(FONT.render(str(no_list[f_indx]),True,BLACK),(x_co,y_co))
        f_indx+=1
        if f_indx>=len(no_list):
            break
    offset=10
# BIG BOX
    init_x=offset
    y_co=offset
    for i in range(MX_NO_OF_DIGITS):
        x_co=init_x
        for j in range(MX_NO_OF_DIGITS):
            big_block.append(RectNode(x_co,y_co,w_b,h_b,False))
            x_co+=w_b+offset
        y_co+=h_b+offset
# MEDIUM BOX
    init_x+=MX_NO_OF_DIGITS*offset+MX_NO_OF_DIGITS*(offset+w_b)
    y_co=offset
    for i in range(MX_NO_OF_DIGITS):
        x_co=init_x
        for j in range(MX_NO_OF_DIGITS):
            medium_block.append(RectNode(x_co,y_co,w_m,h_m,False))
            x_co+=w_m+offset
        y_co+=h_m+offset
# SMALL BOX
    init_x+=MX_NO_OF_DIGITS*offset+MX_NO_OF_DIGITS*(offset+w_m)
    y_co=offset
    for i in range(MX_NO_OF_DIGITS):
        x_co=init_x
        for j in range(MX_NO_OF_DIGITS):
            small_block.append(RectNode(x_co,y_co,w_s,h_s,False))
            x_co+=w_s+offset
        y_co+=h_s+offset
# Construct Box
    gap_x=(width_do/3-w_cb)/2
    gap_y=(height_do-h_cb)/2
    rx,ry=WIDTH/2-width_do/2,HEIGHT/1.15-h_cb/2
    c_gap=4
    for _ in range(MX_NO_OF_DIGITS):
        digit_inner=pygame.Rect(rx+gap_x,ry+gap_y,w_cb,h_cb)
        drawRectangleBorder(digit_inner,BLUE,2)
        y_co=digit_inner[1]+c_gap
        for i in range(5):
            x_co=digit_inner[0]+c_gap
            for j in range(2):
                if _%3==0:
                    obj=RectNode(x_co,y_co,w_b,h_b,True)
                    x_co+=w_b+c_gap
                    y_temp=h_b
                elif _%3==1:
                    obj=RectNode(x_co,y_co,w_m,h_m,True)
                    x_co+=w_m+c_gap
                    y_temp=h_b
                else:
                    obj=RectNode(x_co,y_co,w_s,h_s,True)   
                    x_co+=w_s+c_gap     
                    y_temp=h_b            
                constructArrElement[_].append(obj)
            y_co+=c_gap+y_temp
        rx+=2*gap_x+width_di 
    rgb_array = pygame.surfarray.array3d(WIN)
    if rm=='human':
        pygame.display.update()
    return rgb_array

def drawAgain():
    for __ in big_block:
        if __.isEmpty==False:
            pygame.draw.rect(WIN,CYAN,__.rect)
        else:
            pygame.draw.rect(WIN,WHITE,__.rect)
    for __ in medium_block:
        if __.isEmpty==False:
            pygame.draw.rect(WIN,MAGENTA,__.rect) 
        else:
            pygame.draw.rect(WIN,WHITE,__.rect)
    for __ in small_block:
        if __.isEmpty==False:
            pygame.draw.rect(WIN,YELLOW,__.rect)
        else:
            pygame.draw.rect(WIN,WHITE,__.rect)
    color=[CYAN,MAGENTA,YELLOW]
    for id,c in enumerate(constructArrElement):
        for __ in c:
            if __.isEmpty==False:
                pygame.draw.rect(WIN,color[id],__.rect)
    rgb_array = pygame.surfarray.array3d(WIN)
    if rm=='human':
        pygame.display.update()
    return rgb_array

def draw_main(render_mode,fps,no):
    global FPS,rm,big_block,medium_block,small_block,constructArrElement
    big_block=[];medium_block=[];small_block=[]
    constructArrElement=[[] for _ in range(MX_NO_OF_DIGITS)]
    FPS,rm=fps,render_mode
    clock=pygame.time.Clock()
    no_list=[]
    while no!=0:
        no_list.append(no%10)
        no=no//10
    drawWindowOneTime(no_list)
    if True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
            return drawAgain()
