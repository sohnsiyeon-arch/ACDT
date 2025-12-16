#include <SoftwareSerial.h>
#include <DFRobotDFPlayerMini.h>

const int MP3_RX = 10;
const int MP3_TX = 11;
SoftwareSerial mp3Serial(MP3_RX, MP3_TX);
DFRobotDFPlayerMini dfp;
void setup() {
  Serial.begin(9600); // Raspberry Pi와 통신
  mp3Serial.begin(9600); // DFPlayer와 통신
  dfp.begin(mp3Serial);
  dfp.volume(30);
}
void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      dfp.play(1);
      delay(500);// ser.write(b'1') #focus mp3 디렉토리의 001.mp3 재생
    }
    if (cmd == '2') {
      dfp.play(2);
      delay(500);// ser.write(b'2') #stop mp3 디렉토리의 002.mp3 재생
    }
    if (cmd == '3') {
      dfp.play(3);
      delay(500);// ser.write(b'3') #wasting mp3 디렉토리의 003.mp3 재생
    }
    if (cmd == '4') {
      dfp.play(4);
      delay(500);// ser.write(b'4') #your life mp3 디렉토리의 004.mp3 재생
    }
    if (cmd == '5') {
      dfp.play(5);
      delay(500);// ser.write(b'5') #wake up mp3 디렉토리의 005.mp3 재생
    }
    if (cmd == '6') {
      dfp.play(6);
      delay(500);// ser.write(b'6') #point deduction mp3 디렉토리의 006.mp3 재생
    }
    if (cmd == '7') {
      dfp.play(7);
      delay(500);// ser.write(b'7') #you failed mp3 디렉토리의 007.mp3 재생
    }
    if (cmd == '8') {
      dfp.play(8);
      delay(500);//ser.write(b'8') #you will be F mp3 디렉토리의 008.mp3 재생
    }
  }
}