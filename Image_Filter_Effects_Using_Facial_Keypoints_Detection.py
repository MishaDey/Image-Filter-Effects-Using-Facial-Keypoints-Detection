from Utils import facial_keypoints_detection_model
from HarryPotterInvisibilityCloak import harry_potter_invisibility_cloak
from ipynb.fs.full.SantaClausFilter import Santa_Claus_Filter
from ipynb.fs.full.MoustacheAndGlasses import Moustache_and_Glasses
from ipynb.fs.full.EasterBunnyFace import EasterBunnyFace
from ipynb.fs.full.DogFaceFilter import DogFaceFilter
from ipynb.fs.full.ScaryJawFilter import ScaryJawFilter
from ipynb.fs.full.PopOutEyesFilter import PopOutEyesFilter
from ipynb.fs.full.DevilHornsFilter import DevilHornsFilter

model=facial_keypoints_detection_model()

def main():
    #facial keypoints Detection
    stop=False
    while(stop!=True):
        option=int(input("Choose the Filter Effect :\n\n1.POPPING EYES FILTER\n\n2.DOG FACE FILTER\n\n3.SANTA CLAUS FILTER\n\n4.EASTER BUNNY FACE\n\n5.HARRY POTTER INVISIBILITY CLOAK\n\n6.SCARY JAW FILTER\n\n7.DEVIL HORNS FILTER\n\n8.MOUSTACHE AND GLASSES\n\n9. EXIT\n\n"))
        if(option==1):
            PopOutEyesFilter(model)
        if(option==2):
            DogFaceFilter(model)
        if(option==3):
            Santa_Claus_Filter(model)
        if(option==4):
            EasterBunnyFace(model)
        if(option==5):
            harry_potter_invisibility_cloak()
        if(option==6):
            ScaryJawFilter(model)
        if(option==7):
            DevilHornsFilter(model)
        if(option==8):
            Moustache_and_Glasses(model)
        if(option==9):
            stop=True
    print("\n\nTHANK YOU !!\n\n")

if __name__=='__main__':
    main()