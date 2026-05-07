# STM32F407VET6 Firmware

## Overview

The STM32 microcontroller serves as the lower-level controller, receiving commands from Jetson Nano via UART (serial) and executing motor control, servo control, and sensor data acquisition in real time.

## Architecture

- **RTOS**: FreeRTOS-based task scheduler
- **MCU**: STM32F407VET6 (ARM Cortex-M4, 168MHz)
- **Peripherals**:
  - UART3: Communication with Jetson Nano (baud rate: 115200)
  - Timer PWM: Motor speed control (AT8236 driver)
  - Timer PWM: Servo angle control (MG996)
  - Timer Encoder: Hall encoder pulse counting (wheel odometry)
  - I2C/SPI: IMU (gyroscope/accelerometer) data reading
  - GPIO: OLED display, push buttons

## Task Priority (RTOS)

```
Priority  Task              Description
──────────────────────────────────────────
Highest   UART3_ISR         Receive commands from Jetson Nano
↑         Motor_Control     PID speed control loop
│         Servo_Control     Steering servo angle update
│         Encoder_Read      Wheel encoder pulse counting
│         IMU_Read          Gyroscope data acquisition
Lowest    Display_Update    OLED status display refresh
```

## Communication Protocol

Jetson Nano sends commands via UART3 as comma-separated values:

```
<linear_speed>,<angular_z>\n
```

Example: `0.35,0.12\n`

## Motor Control

- 4-wheel differential drive chassis
- Speed control via PID with encoder feedback
- PWM frequency: 20kHz

## Building & Flashing

1. Open the project in Keil MDK-ARM or STM32CubeIDE
2. Build the project
3. Flash via SWD (ST-Link) or USB DFU
