#ifndef Robot_H
#define Robot_H

#include <Arduino.h>
#include "Motor.hpp"
#include "DistanceMapper.hpp"

class Robot
{
public:
    Robot(const int first_motor_pins[2], const int second_motor_pins[2])
    {
        right_motor = Motor(first_motor_pins[0], first_motor_pins[1]);
        left_motor = Motor(second_motor_pins[0], second_motor_pins[1]);
    };

    void go_right(int moveTime)
    {
        right_motor.go_backwards();
        left_motor.go_forward();
        delay(moveTime);
        right_motor.stop();
        left_motor.stop();
    };

    void go_left(int moveTime)
    {
        right_motor.go_forward();
        left_motor.go_backwards();
        delay(moveTime);
        right_motor.stop();
        left_motor.stop();
    };

    void go_forward(int moveTime)
    {
        right_motor.go_forward();
        left_motor.go_forward();
        delay(moveTime);
        right_motor.stop();
        left_motor.stop();
    };

    void go_backwards(int moveTime)
    {
        right_motor.go_backwards();
        left_motor.go_backwards();
        delay(moveTime);
        right_motor.stop();
        left_motor.stop();
    };

    void setup_robot()
    {
        pinMode(LED_BUILTIN, OUTPUT);
    };

    void goToTheWall(DistanceMapper const &mapper)
    {
        right_motor.go_forward();
        left_motor.go_forward();
        auto distance = mapper.measureDistance();
        while (distance > 20 || distance < 4)
        {
            distance = mapper.measureDistance();
        }
        right_motor.stop();
        left_motor.stop();
    }

    void go(String message, int time)
    {
        if (message == "r")
        {
            go_right(time);
        }
        else if (message == "l")
        {
            go_left(time);
        }
        else if (message == "f")
        {
            go_forward(time);
        }
        else if (message == "b")
        {
            go_backwards(time);
        }
    };

private:
    Motor right_motor;
    Motor left_motor;
    int control_pins_[2];
};

#endif