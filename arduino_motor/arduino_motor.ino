#include <Stepper.h>

const float STEPS_PER_REV = 32;

const float GEAR_RED = 64;

const float STEPS_PER_OUT_REV = STEPS_PER_REV * GEAR_RED;

int StepsRequired;


Stepper steppermotor(STEPS_PER_REV, 8, 10, 9, 11);


void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);

}

void start_kinect() {
  // Turn on Kinect
  Serial.println("k_on");
  delay(3000);

  // Start the motor
  }

void start_classification() {
  Serial.println("class");
  delay(10000);
  }
  
void loop() {

  
  steppermotor.setSpeed(1);
  StepsRequired = 4;
  steppermotor.step(StepsRequired);
  delay(2000);
  start_kinect();

  StepsRequired = 1000;
  steppermotor.setSpeed(100);
  steppermotor.step(StepsRequired);
  delay(1000);
  start_classification();

  StepsRequired = -1000;
  steppermotor.setSpeed(100);
  steppermotor.step(StepsRequired);
  delay(1000);
  

}
