#include "Arduino_RouterBridge.h"
#define LED_PIN             13
#define BAUD_RATE         9600

int consecutiveStressed = 0;

void setup() {
    Serial1.begin(BAUD_RATE);       // ESP32 UART
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Register handler for predictions coming back from Python
    Bridge.provide("prediction", onPrediction);

    Bridge.begin();
    delay(2000);
}

void loop() {
    Bridge.update();

    // Read from ESP32
    if (Serial1.available()) {
        String line = Serial1.readStringUntil('\n');
        line.trim();

        if (line.startsWith("T:")) {
            // Parse T:<temp>,H:<hum>
            float temp = 0, hum = 0;
            int commaIdx = line.indexOf(',');
            if (commaIdx > 0) {
                temp = line.substring(2, commaIdx).toFloat();
                hum  = line.substring(commaIdx + 3).toFloat();
            }
            // Send to Python for inference
            Bridge.notify("sensor_reading", temp, hum);
        }
    }
}

// Called by Bridge when Python sends back a prediction
void onPrediction(String label, float prob) {
    Serial1.println(label + ":" + String(prob, 4));  // relay to ESP32
    handleResponse(label);
}

void handleResponse(const String &resp) {
    if (resp == "WAIT") {
        blink(1, 400);
    } else if (resp == "HEALTHY") {
        blink(2, 200);
    } else if (resp == "STRESSED") {
        blink(5, 300);
    }
}

void blink(int times, int ms) {
    for (int i = 0; i < times; i++) {
        digitalWrite(LED_PIN, HIGH); delay(ms);
        digitalWrite(LED_PIN, LOW);  delay(ms);
    }
}