#include <Stepper.h>

const float STEPS_PER_REV = 32;
const float GEAR_RED = 64;
const float STEPS_PER_OUT_REV = STEPS_PER_REV * GEAR_RED;
int StepsRequired;
Stepper steppermotor(STEPS_PER_REV, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);
}

void start_kinect() {
  // Turn on Kinect
  Serial.println("k_on");
  delay(3000);
  }

void start_classification() {
  Serial.println("class");
  delay(10000);
  }
  
void loop() {

  // 1. Start kinect
  delay(2000);
  start_kinect();

  // 2. Move the waste
  StepsRequired = 1000;
  steppermotor.setSpeed(100);
  steppermotor.step(StepsRequired);
  delay(1000);
  start_classification();

  // 3. Wait for classification to be complete
  // Arduino will receive the input: 'plstc' or 'rsdl'
  while (Serial.available() == 0){
    // Receive the input
    classifiedMaterial = Serial.read();

    // If plastic:
    if (classifiedMaterial == 'plstc'){
        // Notify of the input
        Serial.print("Received: ");
        Serial.println(classifiedMaterial);
      
        StepsRequired = -1000;
        steppermotor.setSpeed(100);
        steppermotor.step(StepsRequired);
        delay(1000);
      
      }
     else if(classifiedMaterial == 'rsdl'){
        // Notify of the input
        Serial.print("Received: ");
        Serial.println(classifiedMaterial);
        
        StepsRequired = -1000;
        steppermotor.setSpeed(1000);
        steppermotor.step(StepsRequired);
        delay(1000);
      }
      else {
        ;
        }
    }


  

}
