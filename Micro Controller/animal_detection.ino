#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "SONY-S28K 8346";
const char* password = "Z641z<24";

WebServer server(80);

const int ledPin1 = 2;
const int ledPin2 = 4; // D2 pin

void setup() {
  pinMode(ledPin1, OUTPUT);
  pinMode(ledPin2, OUTPUT);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..");
  }

  Serial.println(WiFi.localIP());

  server.on("/", HTTP_GET, handleRoot);
  server.on("/on", HTTP_GET, handleTurnOn); // New endpoint for turning the LED on
  server.on("/off", HTTP_GET, handleTurnOff); // New endpoint for turning the LED off
  server.begin();
}

void loop() {
  server.handleClient();
}

void handleRoot() {
  server.send(200, "text/plain", "Hello from ESP32!");
}

void handleTurnOn() {
  Serial.println("LED turned on");
  digitalWrite(ledPin1, HIGH);
  digitalWrite(ledPin2, HIGH); // Turn the LED on
  server.send(200, "text/plain", "LED turned on");
}

void handleTurnOff() {
  Serial.println("LED turned on");
  digitalWrite(ledPin1, LOW); // Turn the LED off
  digitalWrite(ledPin2, LOW);
  server.send(200, "text/plain", "LED turned off");
}
