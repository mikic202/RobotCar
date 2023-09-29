#pragma once

#include <ESP32Servo.h>
#include <vector>
#include "Vector.hpp"
#include <chrono>

#define SOUND_SPEED 0.034

class DistanceMapper
{
private:
    int servoPin;
    Servo rotationController;
    int triggerPin;
    int echoPin;

public:
    DistanceMapper(int servoPin, int triggerPin, int echoPin) : triggerPin(triggerPin), echoPin(echoPin), servoPin(servoPin) { rotationController = Servo(); }
    void initDistanceMapper()
    {
        pinMode(triggerPin, OUTPUT);
        pinMode(echoPin, INPUT);

        ESP32PWM::allocateTimer(0);
        ESP32PWM::allocateTimer(1);
        ESP32PWM::allocateTimer(2);
        ESP32PWM::allocateTimer(3);
        rotationController.setPeriodHertz(50);
        rotationController.attach(servoPin, 500, 2400);
    }
    float measureDistance() const
    {
        constexpr auto RETRIES{1.0};
        float distnce = 0;
        float read = 0;
        int skipped = 0;
        for (int i = 0; i < RETRIES; i++)
        {
            digitalWrite(triggerPin, LOW);
            delay(1);
            digitalWrite(triggerPin, HIGH);
            delay(3);
            digitalWrite(triggerPin, LOW);
            read = pulseIn(echoPin, HIGH) * SOUND_SPEED / 2;
            if (read >= 300)
            {
                skipped++;
                continue;
            }
            distnce += read;
        }
        if (skipped == RETRIES)
            return 0;
        return distnce / (RETRIES - skipped);
    }

    std::vector<Vector> mapSoroundings()
    {
        constexpr auto lowerRotationBound{40};
        constexpr auto upperRotationBound{140};
        constexpr auto rotationStep{1};
        std::vector<Vector> measuredValues;
        auto start = std::chrono::high_resolution_clock::now();
        for (int pos = lowerRotationBound; pos <= upperRotationBound; pos += rotationStep)
        {
            rotationController.write(pos);
            delay(3);
            float distance = measureDistance();
            measuredValues.emplace_back(Vector(distance, pos));
            Serial.println(distance);
        }
        rotationController.write(90);
        Serial.print("Took: ");
        Serial.println(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
        return measuredValues;
    }
};
