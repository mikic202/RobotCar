#ifndef Motor_H
#define Motor_H

#include <Arduino.h>

class Motor
{
public:
    Motor(){};
    Motor(int forwardPin, int backwardPin, int pwmPin, int pwmChannel) : forwardPin_(forwardPin), backwardPin_(backwardPin), pwmPin_(pwmPin)
    {
        pinMode(forwardPin_, OUTPUT);
        pinMode(backwardPin_, OUTPUT);
        pinMode(pwmPin_, OUTPUT);
        ledcSetup(pwmChannel, 500, 8);
        ledcAttachPin(pwmPin_, pwmChannel);
        pwmChannel_ = pwmChannel;
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
    void setSpeed(float pwmValue)
    {
        Serial.println((pwmValue / 100) * 255);
        ledcWrite(pwmChannel_, (pwmValue / 100) * 255);
    };

private:
    int pwmChannel_;
    int forwardPin_;
    int backwardPin_;
    int pwmPin_;
};

#endif