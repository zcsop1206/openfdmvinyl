// adc_capture.ino - record a ceramic cartridge through an Arduino UNO (Elegoo kit) ADC.
//
// Wiring (one channel is enough: a lateral groove moves L and R together):
//   5V  --[1 MOhm]--+--[1 MOhm]-- GND      bias A0 at 2.5 V, 500 kOhm load
//                   |
//                   A0 ---- cartridge L (hot)
//   GND ---------------- cartridge L (ground)
// Keep the cartridge leads short and twisted; mains hum is the main enemy at this impedance.
// 500 kOhm against a ~1 nF ceramic gives a first-order high-pass near 300 Hz, the same
// corner as the encoder's HPF, so nothing below it was cut into the groove anyway.
//
// The ADC free-runs at 16 MHz / 128 / 13 = 9615.4 samples/s (crystal-locked).
// Each 10-bit sample goes out as two bytes: 1hhhhh (top 5 bits), 0lllll (low 5 bits).
// Serial at 500000 baud. Record with capture.py.

volatile uint16_t buf[256];
volatile uint8_t head = 0;
uint8_t tail = 0;
volatile bool overflow = false;

ISR(ADC_vect) {
  uint8_t lo = ADCL;
  uint8_t hi = ADCH;
  uint8_t next = head + 1;
  if (next == tail) { overflow = true; return; }
  buf[head] = ((uint16_t)hi << 8) | lo;
  head = next;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(500000);
  ADMUX = _BV(REFS0);                                   // AVcc reference, channel A0, right-adjusted
  ADCSRB = 0;                                           // free-running trigger
  DIDR0 = _BV(ADC0D);                                   // disable digital input buffer on A0
  ADCSRA = _BV(ADEN) | _BV(ADATE) | _BV(ADIE) | _BV(ADPS2) | _BV(ADPS1) | _BV(ADPS0);  // /128
  ADCSRA |= _BV(ADSC);
}

void loop() {
  while (tail != head) {
    uint16_t v = buf[tail];
    tail++;
    Serial.write(0x80 | (v >> 5));
    Serial.write(v & 0x1F);
  }
  digitalWrite(LED_BUILTIN, overflow);                  // LED on = samples were dropped
}
