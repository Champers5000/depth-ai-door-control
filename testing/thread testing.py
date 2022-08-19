from threading import Thread, Lock
import signal 
name = "unnamed"

  

def keyboardInterruptHandler(signal, frame):

    #print("call your function here".format(signal))
    print("piece of crap why you terminate")
    exit(1)

signal.signal(signal.SIGINT, keyboardInterruptHandler)


def getinput():
    global name
    lock = Lock()
    print("this is the getinput method")
    while True:
            userin = input("You gonna press a key?")
            if(userin == "i"):
                userin = input("Putin a new name")
                lock.acquire()
                name = userin
                lock.release()
        

def main():
    lastname = name
    while True:
        if not lastname == name:
            print(name)
        lastname = name
        if name == "quit" :
            exit()

t1= Thread(target=getinput)
t2 = Thread(target = main)

t1.start()
t2.start()