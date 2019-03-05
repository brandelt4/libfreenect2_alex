int yBUTTON = 4;
int rBUTTON = 5;
int last_rubbish_location = 0;
int num_compartments = 8;



void setup() {
  pinMode(yBUTTON, INPUT);
  pinMode(rBUTTON, INPUT);
  
  Serial.begin(9600);

}



void loop() {

  if (digitalRead(yBUTTON) == HIGH) {
    move_one();
    
    } 

  if (digitalRead(rBUTTON) == HIGH) {
    move_all();
    }
    
}

void move_one() {
  last_rubbish_location = last_rubbish_location + 1;

  if (last_rubbish_location == num_compartments) {
    // Move one and call start_linear()
    
    }

  // Move the stepper motor some angle
  
  
  }

void move_all() {

  // Move stepper motor (num_compartments - last_rubbish_location) steps

  }


void start_linear() {
  // Turn on Kinect
  Serial.print("k_on")
  delay(1000);

  // Start the motor

  
  }
