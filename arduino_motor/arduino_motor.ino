#include <Stepper.h>

const float STEPS_PER_REV = 32;

const float GEAR_RED = 64;

const float STEPS_PER_OUT_REV = STEPS_PER_REV * GEAR_RED;

int StepsRequired;

const int buttonPin = 2;

int buttonState = 0;

Stepper steppermotor(STEPS_PER_REV, 8, 10, 9, 11);


void setup() {
  // put your setup code here, to run once:

  pinMode(buttonPin, INPUT);

}

void loop() {

  buttonState = digitalRead(buttonPin);

  if (buttonState == HIGH) {
    steppermotor.setSpeed(1);
    StepsRequired = 4;
    steppermotor.step(StepsRequired);
    delay(2000);
  
    StepsRequired = STEPS_PER_OUT_REV / 2;
    steppermotor.setSpeed(100);
    steppermotor.step(StepsRequired);
    delay(1000);
    }

  else {
    ;
    }

}
