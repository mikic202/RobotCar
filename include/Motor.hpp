#ifndef Motor_H
#define Motor_H

#include <Arduino.h>

class Motor
{
public:
    Motor(){};
    Motor(int forwardPin, int backwardPin) : forwardPin_(forwardPin), backwardPin_(backwardPin)
    {
        pinMode(forwardPin_, OUTPUT);
        pinMode(backwardPin_, OUTPUT);
    };

    void go_forward()
    {
        digitalWrite(forwardPin_, HIGH);
        digitalWrite(backwardPin_, LOW);
    };

    void go_backwards()
    {
        digitalWrite(forwardPin_, LOW);
        digitalWrite(backwardPin_, HIGH);
    };

    void stop()
    {
        digitalWrite(forwardPin_, LOW);
        digitalWrite(backwardPin_, LOW);
    };

private:
    int forwardPin_;
    int backwardPin_;
};

#endif