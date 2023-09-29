#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <vector>

class Connector
{
public:
    WiFiServer server;
    Connector()
    {
        server = WiFiServer(port);
    }

    IPAddress initServer()
    {
        WiFi.softAP(ssid, password);
        IP = WiFi.softAPIP();
        server.begin();
        return IP;
    }

    void waitForClient()
    {
        client = server.available();
    }

    WiFiClient const &getClient()
    {
        return client;
    }

    std::vector<String> getClientMessage()
    {
        std::vector<String> messages;
        if (client.available())
        {
            String line = client.readStringUntil('\n');
            messages.emplace_back(line.substring(0, line.indexOf(";")));
            messages.emplace_back(line.substring(line.indexOf(";") + 1));
        }
        return messages;
    }

private:
    const int port = 8080;
    const char *ssid = "ESP32-Access-Point";
    const char *password = "123456789";
    IPAddress IP;
    WiFiClient client;
};