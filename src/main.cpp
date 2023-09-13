#include <Arduino.h>
#include <WiFi.h>
#include "Motor.hpp"
#include "Robot.hpp"
#include "Connector.hpp"
#include <ESP32Servo.h>
#include "DistanceMapper.hpp"

#define SOUND_SPEED 0.034

const int motorOne[] = {27, 26};
const int motorTwo[] = {33, 25};

int echo = 32;
int trigger = 12;
int servoPin = 21;

Motor firstMotor(motorOne[0], motorOne[1]);
Motor secondMotor(motorTwo[0], motorTwo[1]);
Robot robot(motorOne, motorTwo);
Connector connector;
DistanceMapper mapper(servoPin, trigger, echo);

String parseDistanceData(std::vector<Vector> data)
{
  String output = "";
  for (auto measurement : data)
  {
    output += String(measurement.r) + "," + String(measurement.fi) + ";";
  }
  return output;
}

void setup()
{
  Serial.begin(115200);
  Serial.print("Setting AP (Access Point)…");
  Serial.print("AP IP address: ");
  Serial.println(connector.initServer());
  mapper.initDistanceMapper();
}

void loop()
{
  connector.waitForClient();
  WiFiClient client = connector.getClient();

  if (client)
  {
    Serial.println("New Client.");
    String currentLine = "";
    while (client.connected())
    {
      if (client.available())
      {
        auto messages = connector.getClientMessage();
        Serial.println(messages[0]);
        if (messages[0] == "end")
        {
          break;
        }
        if (messages[0] == "m")
        {
          client.println(parseDistanceData(mapper.mapSoroundings()));
          continue;
        }
        else if (messages[0] == "w")
        {
          robot.goToTheWall(mapper);
        }
        robot.go(messages[0], messages[1].toInt());
        client.print(mapper.measureDistance());
        client.println(123);
      }
    }
    Serial.println("Client disconnected.");
    Serial.println("");
  }
}

int myFunction(int x, int y)
{
  return x + y;
}