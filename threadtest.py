from threading import Thread, Lock
import signal
import time
name = "unnamed"



def keyboardInterruptHandler(signal, frame):

    #print("call your function here".format(signal))
    print("piece of crap why you terminate")
    exit(1)

signal.signal(signal.SIGINT, keyboardInterruptHandler)


def getinput():
    global name
    print("this is the getinput method")
    while True:
            if name == "quit" :
                exit()
            userin = input("You gonna press a key?")
            if(userin == "i"):
                userin = input("Putin a new name")
                lock.acquire()
                name = userin
                lock.release()


def main():
    lastname = name
    while True:
        time.sleep(1)
        print(name)
        lastname = name
        if name == "quit" :
            exit()

lock = Lock()
t1= Thread(target=getinput)
t2 = Thread(target = main)

t1.start()
t2.start()
